"""
acoustic.audio
==============

UMA-16 v2 audio acquisition interface for the
Multimodal Acoustic Camera project.

This module uses a callback-based PortAudio stream because
the Windows WDM-KS backend used by the UMA-16 does not reliably
support blocking InputStream.read() operation.

Responsibilities
----------------
- Open the UMA-16 USB audio interface
- Configure multichannel audio acquisition
- Receive audio through a PortAudio callback
- Buffer incoming samples
- Return fixed-size timestamped audio frames
- Provide basic stream statistics
- Cleanly start and stop the stream

This module intentionally does NOT perform:
- MUSIC localization
- FFT processing
- beamforming
- steering-vector calculation
- acoustic localization
- synchronization
- multimodal fusion

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd

from config.settings import (
    SAMPLE_RATE,
    NUM_CHANNELS,
    FRAME_SIZE,
    HOP_SIZE,
)


# ============================================================
# Audio Frame
# ============================================================


@dataclass
class AudioFrame:
    """
    Container for one acquired UMA-16 audio frame.

    data
        Audio samples with shape:

            (frame_size, channels)

    timestamp
        Timestamp from the local monotonic clock.

    frame_id
        Sequential audio frame identifier.
    """

    data: np.ndarray
    timestamp: float
    frame_id: int

    @property
    def sample_count(self) -> int:
        """Return number of samples per channel."""

        return self.data.shape[0]

    @property
    def channel_count(self) -> int:
        """Return number of channels."""

        return self.data.shape[1]


# ============================================================
# Audio Acquisition
# ============================================================


class AudioAcquisition:
    """
    Callback-based UMA-16 multichannel audio acquisition.

    Example
    -------

    audio = AudioAcquisition(device=28)

    audio.start()

    packet = audio.read(timeout=5.0)

    if packet is not None:
        samples = packet.data

    audio.stop()
    """

    def __init__(
        self,
        device: Optional[int | str] = None,
        sample_rate: int = SAMPLE_RATE,
        channels: int = NUM_CHANNELS,
        frame_size: int = FRAME_SIZE,
        hop_size: int = HOP_SIZE,
    ) -> None:

        if sample_rate <= 0:
            raise ValueError(
                "sample_rate must be positive."
            )

        if channels <= 0:
            raise ValueError(
                "channels must be positive."
            )

        if frame_size <= 0:
            raise ValueError(
                "frame_size must be positive."
            )

        if hop_size <= 0:
            raise ValueError(
                "hop_size must be positive."
            )

        self.device = device
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)

        self.stream: Optional[sd.InputStream] = None

        self.started = False

        self.frame_count = 0

        self.start_time: Optional[float] = None

        # ----------------------------------------------------
        # Internal sample buffer
        # ----------------------------------------------------

        self._buffer = np.empty(
            (0, self.channels),
            dtype=np.float32,
        )

        self._buffer_lock = threading.Lock()

        # ----------------------------------------------------
        # Callback state
        # ----------------------------------------------------

        self._callback_error: Optional[Exception] = None

        self._input_samples_received = 0

        self._overflow_count = 0

    # ========================================================
    # PortAudio callback
    # ========================================================

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        """
        PortAudio callback.

        IMPORTANT:
        Do not perform expensive processing here.

        The callback only copies incoming samples into the
        internal buffer. MUSIC and other processing happen
        outside the callback thread.
        """

        if status:

            if getattr(status, "input_overflow", False):

                self._overflow_count += 1

        try:

            block = np.asarray(
                indata,
                dtype=np.float32,
            ).copy()

            if block.ndim != 2:

                self._callback_error = RuntimeError(
                    "UMA-16 callback returned unexpected "
                    f"data dimensions: {block.shape}"
                )

                return

            if block.shape[1] != self.channels:

                self._callback_error = RuntimeError(
                    "UMA-16 callback returned "
                    f"{block.shape[1]} channels; "
                    f"expected {self.channels}."
                )

                return

            with self._buffer_lock:

                self._buffer = np.concatenate(
                    (
                        self._buffer,
                        block,
                    ),
                    axis=0,
                )

                self._input_samples_received += (
                    block.shape[0]
                )

        except Exception as exc:

            self._callback_error = exc

    # ========================================================
    # Stream control
    # ========================================================

    def start(self) -> None:
        """
        Start the UMA-16 callback stream.
        """

        if self.started:
            return

        print(
            f"[Audio] Starting UMA-16 stream "
            f"(device {self.device}) "
            f"at {self.sample_rate} Hz"
        )

        self._buffer = np.empty(
            (0, self.channels),
            dtype=np.float32,
        )

        self.frame_count = 0

        self._callback_error = None

        self._input_samples_received = 0

        self._overflow_count = 0

        try:

            self.stream = sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.hop_size,
                callback=self._audio_callback,
            )

            self.stream.start()

        except Exception:

            if self.stream is not None:

                try:
                    self.stream.close()
                except Exception:
                    pass

            self.stream = None

            raise

        self.started = True

        self.start_time = time.perf_counter()

        print("[Audio] Stream started")

    def stop(self) -> None:
        """
        Stop and release the UMA-16 stream.
        """

        if self.stream is not None:

            try:
                self.stream.stop()
            except Exception:
                pass

            try:
                self.stream.close()
            except Exception:
                pass

            self.stream = None

        if self.started:

            print("[Audio] Stream stopped")

        self.started = False

    # ========================================================
    # Frame acquisition
    # ========================================================

    def read(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[AudioFrame]:
        """
        Return one fixed-size audio frame.

        The callback continuously fills the internal buffer.
        This method waits until enough samples are available.

        Returned shape:

            (frame_size, channels)
        """

        if not self.started or self.stream is None:

            raise RuntimeError(
                "Audio stream is not started. "
                "Call audio.start() first."
            )

        start_wait = time.perf_counter()

        while True:

            # ------------------------------------------------
            # Check callback errors
            # ------------------------------------------------

            if self._callback_error is not None:

                raise RuntimeError(
                    "Audio callback failed: "
                    f"{self._callback_error}"
                )

            # ------------------------------------------------
            # Check buffer
            # ------------------------------------------------

            with self._buffer_lock:

                if (
                    self._buffer.shape[0]
                    >= self.frame_size
                ):

                    frame_data = self._buffer[
                        :self.frame_size
                    ].copy()

                    # Advance by hop size.
                    #
                    # This gives overlapping frames when:
                    #
                    # FRAME_SIZE > HOP_SIZE
                    #
                    remaining_start = self.hop_size

                    if (
                        remaining_start
                        < self._buffer.shape[0]
                    ):

                        self._buffer = self._buffer[
                            remaining_start:
                        ].copy()

                    else:

                        self._buffer = np.empty(
                            (0, self.channels),
                            dtype=np.float32,
                        )

                    break

            # ------------------------------------------------
            # Timeout
            # ------------------------------------------------

            if timeout is not None:

                elapsed = (
                    time.perf_counter()
                    - start_wait
                )

                if elapsed >= timeout:

                    return None

            # ------------------------------------------------
            # Do not busy-spin
            # ------------------------------------------------

            time.sleep(0.001)

        timestamp = time.perf_counter()

        self.frame_count += 1

        return AudioFrame(
            data=frame_data,
            timestamp=timestamp,
            frame_id=self.frame_count,
        )

    # ========================================================
    # Information
    # ========================================================

    def get_frame_count(self) -> int:
        """Return number of successfully acquired frames."""

        return self.frame_count

    def get_elapsed_time(self) -> float:
        """Return elapsed stream time."""

        if self.start_time is None:
            return 0.0

        return (
            time.perf_counter()
            - self.start_time
        )

    def get_fps(self) -> float:
        """Return average acquired frame rate."""

        elapsed = self.get_elapsed_time()

        if elapsed <= 0.0:
            return 0.0

        return self.frame_count / elapsed

    def get_buffered_samples(self) -> int:
        """Return currently buffered samples per channel."""

        with self._buffer_lock:

            return int(
                self._buffer.shape[0]
            )

    def get_input_samples_received(self) -> int:
        """Return total samples received from PortAudio."""

        return self._input_samples_received

    def get_overflow_count(self) -> int:
        """Return number of reported input overflows."""

        return self._overflow_count

    def get_device_info(self) -> dict:
        """
        Return acquisition configuration.
        """

        return {
            "device": self.device,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frame_size": self.frame_size,
            "hop_size": self.hop_size,
        }

    # ========================================================
    # Context manager
    # ========================================================

    def __enter__(self) -> "AudioAcquisition":

        self.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:

        self.stop()