"""
vision.camera
=============

ESP32-CAM video acquisition interface for the
Multimodal Acoustic Camera project.

This module provides a low-latency abstraction over the
ESP32-CAM MJPEG HTTP stream.

Responsibilities
----------------
- Connect to the ESP32-CAM MJPEG stream
- Continuously read the camera stream in a background thread
- Keep ONLY the newest frame
- Discard stale buffered frames
- Return OpenCV frames
- Provide frame timestamps
- Handle stream lifecycle safely

Important
---------
The camera reader thread is the ONLY component that calls
cv2.VideoCapture.read().

The public read() method returns the latest already-decoded
frame instead of reading directly from the network stream.

This prevents old ESP32-CAM frames from accumulating and
causing multi-second visual latency.

This module intentionally does NOT perform:
- YOLO inference
- image segmentation
- acoustic processing
- synchronization
- multimodal fusion

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2

from config.settings import (
    STREAM_URL,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CAMERA_MODEL,
)


# ============================================================
# Camera Frame
# ============================================================


class CameraFrame:
    """
    Container for one camera frame.

    Attributes
    ----------
    frame : numpy.ndarray
        OpenCV BGR image.

    timestamp : float
        Local monotonic timestamp corresponding to when the
        frame was acquired from the camera stream.

    frame_id : int
        Sequential frame number.
    """

    def __init__(
        self,
        frame,
        timestamp: float,
        frame_id: int,
    ) -> None:

        self.frame = frame
        self.timestamp = timestamp
        self.frame_id = frame_id


# ============================================================
# Camera Acquisition
# ============================================================


class CameraAcquisition:
    """
    Low-latency ESP32-CAM MJPEG stream acquisition.

    Architecture
    ------------

        ESP32-CAM
             |
             v
        VideoCapture
             |
             v
       Reader Thread
             |
             v
       Latest Frame
             |
             v
          read()
             |
             v
       Main Pipeline

    Only the newest decoded frame is retained.

    This deliberately avoids a FIFO frame queue because a
    real-time multimodal camera should prefer a slightly newer
    frame over processing an increasingly old frame.

    Example
    -------

    camera = CameraAcquisition()

    camera.start()

    packet = camera.read()

    if packet is not None:
        frame = packet.frame
        timestamp = packet.timestamp

    camera.stop()
    """

    def __init__(
        self,
        stream_url: str = STREAM_URL,
        expected_width: int = FRAME_WIDTH,
        expected_height: int = FRAME_HEIGHT,
        camera_model: str = CAMERA_MODEL,
    ) -> None:

        self.stream_url = stream_url
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.camera_model = camera_model

        # ----------------------------------------------------
        # OpenCV capture
        # ----------------------------------------------------

        self.capture: Optional[cv2.VideoCapture] = None

        # ----------------------------------------------------
        # Acquisition state
        # ----------------------------------------------------

        self.frame_count = 0
        self.started = False
        self.start_time: Optional[float] = None

        # ----------------------------------------------------
        # Latest-frame storage
        # ----------------------------------------------------

        self._latest_frame: Optional[CameraFrame] = None

        # Lock only protects the reference to the latest frame.
        # We never perform image processing while holding it.
        self._frame_lock = threading.Lock()

        # ----------------------------------------------------
        # Reader thread
        # ----------------------------------------------------

        self._reader_thread: Optional[threading.Thread] = None

        self._reader_running = threading.Event()

        # ----------------------------------------------------
        # Reader status
        # ----------------------------------------------------

        self._reader_error: Optional[Exception] = None

    # ========================================================
    # Stream control
    # ========================================================

    def start(self) -> None:
        """
        Connect to the ESP32-CAM MJPEG stream and start the
        background frame reader.
        """

        if self.started:
            return

        print("[Camera] Connecting to ESP32-CAM...")
        print(f"[Camera] Model : {self.camera_model}")
        print(f"[Camera] URL   : {self.stream_url}")

        capture = cv2.VideoCapture(self.stream_url)

        if not capture.isOpened():

            capture.release()

            raise RuntimeError(
                f"Unable to connect to ESP32-CAM stream: "
                f"{self.stream_url}"
            )

        # ----------------------------------------------------
        # Low-latency capture configuration
        # ----------------------------------------------------
        #
        # These properties are best-effort. MJPEG HTTP streams
        # may ignore them depending on the backend.
        # ----------------------------------------------------

        try:
            capture.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1,
            )
        except Exception:
            pass

        # ----------------------------------------------------
        # Reset state
        # ----------------------------------------------------

        self.capture = capture

        self.frame_count = 0
        self.start_time = time.perf_counter()

        self._latest_frame = None
        self._reader_error = None

        self.started = True
        self._reader_running.set()

        # ----------------------------------------------------
        # Start ONE reader thread.
        #
        # IMPORTANT:
        # The reader thread is now the ONLY place where
        # capture.read() is called.
        # ----------------------------------------------------

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="CameraFrameReader",
            daemon=True,
        )

        self._reader_thread.start()

        print("[Camera] Connected successfully")

    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------

    def stop(self) -> None:
        """
        Stop the camera reader thread and release the stream.

        The reader thread is stopped before VideoCapture is
        released to avoid OpenCV access from two threads.
        """

        # Signal reader thread to stop.
        self._reader_running.clear()

        reader_thread = self._reader_thread

        # Wait briefly for the reader thread to exit.
        if (
            reader_thread is not None
            and reader_thread.is_alive()
            and reader_thread is not threading.current_thread()
        ):

            reader_thread.join(timeout=1.0)

        self._reader_thread = None

        # ----------------------------------------------------
        # Release VideoCapture only after reader exits.
        # ----------------------------------------------------

        capture = self.capture
        self.capture = None

        if capture is not None:

            try:
                capture.release()
            except Exception:
                pass

        # ----------------------------------------------------
        # Clear latest frame
        # ----------------------------------------------------

        with self._frame_lock:
            self._latest_frame = None

        if self.started:
            print("[Camera] Stream stopped")

        self.started = False

    # ========================================================
    # Background reader
    # ========================================================

    def _reader_loop(self) -> None:
        """
        Continuously read frames from the ESP32-CAM.

        This is the ONLY method allowed to call
        VideoCapture.read().

        The loop intentionally keeps only the newest frame.
        If the processing pipeline is slower than the camera,
        intermediate frames are discarded automatically.
        """

        while self._reader_running.is_set():

            capture = self.capture

            if capture is None:
                break

            try:
                success, frame = capture.read()

            except Exception as exc:

                self._reader_error = exc

                print(
                    f"[Camera] Reader error: {exc}"
                )

                break

            if not success or frame is None:

                # Do not immediately spin at maximum CPU usage
                # if the stream temporarily fails.
                time.sleep(0.005)

                continue

            # Timestamp as close as possible to successful
            # frame acquisition.
            timestamp = time.perf_counter()

            self.frame_count += 1

            camera_frame = CameraFrame(
                frame=frame,
                timestamp=timestamp,
                frame_id=self.frame_count,
            )

            # ------------------------------------------------
            # Replace the previous frame.
            #
            # No queue.
            #
            # Old frame becomes eligible for garbage collection.
            # ------------------------------------------------

            with self._frame_lock:

                self._latest_frame = camera_frame

    # ========================================================
    # Frame acquisition
    # ========================================================

    def read(self) -> Optional[CameraFrame]:
        """
        Return the newest available camera frame.

        IMPORTANT
        ---------
        This method does NOT call cv2.VideoCapture.read().

        The background reader continuously acquires frames and
        discards stale frames. Therefore this method returns
        the newest frame available at the time of the call.

        Returns
        -------
        CameraFrame or None
            Latest decoded frame, or None if no frame has been
            received yet.

        Raises
        ------
        RuntimeError
            If the camera has not been started.
        """

        if not self.started:
            raise RuntimeError(
                "Camera is not started. "
                "Call camera.start() first."
            )

        # ----------------------------------------------------
        # Get newest frame.
        # ----------------------------------------------------

        with self._frame_lock:

            camera_frame = self._latest_frame

        # ----------------------------------------------------
        # No frame received yet.
        # ----------------------------------------------------

        if camera_frame is None:
            return None

        # ----------------------------------------------------
        # Return the latest frame.
        #
        # We intentionally do NOT clear _latest_frame here.
        #
        # If the pipeline temporarily calls read() faster than
        # the camera produces frames, returning the latest frame
        # again is preferable to blocking.
        # ----------------------------------------------------

        return camera_frame

    # ========================================================
    # Camera information
    # ========================================================

    def get_frame_count(self) -> int:
        """
        Return the number of successfully acquired frames.
        """

        return self.frame_count

    # --------------------------------------------------------

    def get_elapsed_time(self) -> float:
        """
        Return elapsed acquisition time in seconds.
        """

        if self.start_time is None:
            return 0.0

        return time.perf_counter() - self.start_time

    # --------------------------------------------------------

    def get_fps(self) -> float:
        """
        Estimate average camera acquisition FPS.
        """

        elapsed = self.get_elapsed_time()

        if elapsed <= 0:
            return 0.0

        return self.frame_count / elapsed

    # --------------------------------------------------------

    def get_resolution(
        self,
    ) -> Optional[tuple[int, int]]:
        """
        Return the actual resolution reported by the stream.

        Returns
        -------
        tuple[int, int] or None
            (width, height)
        """

        capture = self.capture

        if capture is None:
            return None

        try:

            width = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

        except Exception:
            return None

        if width <= 0 or height <= 0:
            return None

        return width, height

    # ========================================================
    # Reader diagnostics
    # ========================================================

    def has_reader_error(self) -> bool:
        """
        Return True if the background reader encountered an error.
        """

        return self._reader_error is not None

    # --------------------------------------------------------

    def get_reader_error(self) -> Optional[Exception]:
        """
        Return the last reader exception, if any.
        """

        return self._reader_error

    # ========================================================
    # Context manager
    # ========================================================

    def __enter__(self) -> "CameraAcquisition":

        self.start()

        return self

    # --------------------------------------------------------

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:

        self.stop()