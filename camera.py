"""
vision.camera
=============

ESP32-CAM video acquisition interface for the
Multimodal Acoustic Camera project.

This module provides a reusable abstraction over the
ESP32-CAM MJPEG HTTP stream.

Responsibilities
----------------
- Connect to the ESP32-CAM MJPEG stream
- Decode incoming JPEG frames
- Return OpenCV frames
- Provide frame timestamps
- Handle stream lifecycle

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

import time
from typing import Optional

import cv2

from config.settings import (
    STREAM_URL,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CAMERA_MODEL,
)


class CameraFrame:
    """
    Container for one camera frame.

    Attributes
    ----------
    frame : numpy.ndarray
        OpenCV BGR image.
    timestamp : float
        Acquisition timestamp from the local system clock.
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


class CameraAcquisition:
    """
    ESP32-CAM MJPEG stream acquisition.

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

        self.capture: Optional[cv2.VideoCapture] = None

        self.frame_count = 0
        self.started = False

        self.start_time: Optional[float] = None

    # ---------------------------------------------------------
    # Stream control
    # ---------------------------------------------------------

    def start(self) -> None:
        """
        Connect to the ESP32-CAM MJPEG stream.
        """

        if self.started:
            return

        print("[Camera] Connecting to ESP32-CAM...")
        print(f"[Camera] Model : {self.camera_model}")
        print(f"[Camera] URL   : {self.stream_url}")

        self.capture = cv2.VideoCapture(self.stream_url)

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None

            raise RuntimeError(
                f"Unable to connect to ESP32-CAM stream: "
                f"{self.stream_url}"
            )

        self.started = True
        self.start_time = time.perf_counter()
        self.frame_count = 0

        print("[Camera] Connected successfully")

    def stop(self) -> None:
        """
        Stop the camera stream and release resources.
        """

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        if self.started:
            print("[Camera] Stream stopped")

        self.started = False

    # ---------------------------------------------------------
    # Frame acquisition
    # ---------------------------------------------------------

    def read(self) -> Optional[CameraFrame]:
        """
        Read one frame from the ESP32-CAM.

        Returns
        -------
        CameraFrame or None
            Decoded frame with timestamp and frame ID.

        Raises
        ------
        RuntimeError
            If the camera has not been started.
        """

        if not self.started or self.capture is None:
            raise RuntimeError(
                "Camera is not started. Call camera.start() first."
            )

        success, frame = self.capture.read()

        if not success or frame is None:
            print("[Camera] Failed to read frame")
            return None

        timestamp = time.perf_counter()

        self.frame_count += 1

        return CameraFrame(
            frame=frame,
            timestamp=timestamp,
            frame_id=self.frame_count,
        )

    # ---------------------------------------------------------
    # Camera information
    # ---------------------------------------------------------

    def get_frame_count(self) -> int:
        """
        Return the number of successfully acquired frames.
        """

        return self.frame_count

    def get_elapsed_time(self) -> float:
        """
        Return elapsed acquisition time in seconds.
        """

        if self.start_time is None:
            return 0.0

        return time.perf_counter() - self.start_time

    def get_fps(self) -> float:
        """
        Estimate average acquisition FPS.

        Returns
        -------
        float
            Average FPS since acquisition started.
        """

        elapsed = self.get_elapsed_time()

        if elapsed <= 0:
            return 0.0

        return self.frame_count / elapsed

    def get_resolution(self) -> Optional[tuple[int, int]]:
        """
        Return the actual resolution of the most recently
        available camera stream.

        Returns
        -------
        tuple[int, int] or None
            (width, height)
        """

        if self.capture is None:
            return None

        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            return None

        return width, height

    # ---------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------

    def __enter__(self) -> "CameraAcquisition":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()