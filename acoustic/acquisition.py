"""
acoustic.acquisition
====================

Real-time audio acquisition interface for the miniDSP UMA-16 v2 microphone array.

This module provides a reusable abstraction over the sounddevice InputStream and
implements the same sliding-window buffering strategy used in the validated
`uma_live_camera.py` prototype.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import queue
from typing import Optional

import numpy as np
import sounddevice as sd

from config.settings import (
    AUDIO_DEVICE_NAME,
    SAMPLE_RATE,
    NUM_CHANNELS,
    FRAME_SIZE,
    HOP_SIZE,
)


class AudioAcquisition:
    """
    Real-time audio acquisition for the UMA-16 microphone array.

    The class automatically discovers the microphone array by name, opens a
    multi-channel input stream, and maintains a sliding analysis buffer.

    Example
    -------
    audio = AudioAcquisition()
    audio.start()

    frame = audio.read()

    audio.stop()
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = NUM_CHANNELS,
        frame_size: int = FRAME_SIZE,
        hop_size: int = HOP_SIZE,
        device_name: str = AUDIO_DEVICE_NAME,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.device_name = device_name

        self.device_index: Optional[int] = None
        self.stream: Optional[sd.InputStream] = None

        # Thread-safe queue filled by the audio callback
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        # Sliding analysis buffer
        self.audio_buffer = np.zeros(
            (self.frame_size, self.channels),
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # Device discovery
    # ---------------------------------------------------------

    def find_device(self) -> int:
        """
        Locate the UMA-16 device by partial name match.

        Returns
        -------
        int
            PortAudio device index.

        Raises
        ------
        RuntimeError
            If the device cannot be found.
        """

        devices = sd.query_devices()

        for index, device in enumerate(devices):
            name = device["name"]

            if (
                self.device_name.lower() in name.lower()
                and device["max_input_channels"] >= self.channels
            ):
                return index

        raise RuntimeError(
            f"Audio device containing '{self.device_name}' not found."
        )

    # ---------------------------------------------------------
    # Audio callback
    # ---------------------------------------------------------

    def _callback(self, indata, frames, time_info, status) -> None:
        """
        Background callback executed by sounddevice.

        Incoming audio blocks are copied into the thread-safe queue.
        """

        if status:
            print(f"[Audio] {status}")

        self.audio_queue.put(indata.copy())

    # ---------------------------------------------------------
    # Stream control
    # ---------------------------------------------------------

    def start(self) -> None:
        """
        Open and start the input stream.
        """

        self.device_index = self.find_device()

        self.stream = sd.InputStream(
            device=self.device_index,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.hop_size,
            callback=self._callback,
        )

        self.stream.start()

        print(
            f"[Audio] Started UMA-16 stream (device {self.device_index}) "
            f"at {self.sample_rate} Hz"
        )

    def stop(self) -> None:
        """
        Stop and close the input stream.
        """

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

            print("[Audio] Stream stopped")

    # ---------------------------------------------------------
    # Frame acquisition
    # ---------------------------------------------------------

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Return one raw audio frame directly from the callback queue.

        This method is intended for calibration and low-level processing where
        the newest available audio block is required.

        Parameters
        ----------
        timeout : float
            Maximum time to wait for a frame in seconds.

        Returns
        -------
        numpy.ndarray | None
            Audio frame with shape (HOP_SIZE, NUM_CHANNELS), or None if no
            frame is received within the timeout period.
        """

        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read(self) -> np.ndarray:
        """
        Return the latest sliding analysis window.

        This preserves the exact buffering strategy from the original
        prototype: each callback provides HOP_SIZE samples, which are
        appended to the end of the analysis buffer while the oldest
        samples are discarded from the front.

        Returns
        -------
        numpy.ndarray
            Audio frame with shape (FRAME_SIZE, NUM_CHANNELS).
        """

        try:
            while not self.audio_queue.empty():
                indata = self.audio_queue.get_nowait()

                self.audio_buffer[:-self.hop_size] = self.audio_buffer[self.hop_size :]
                self.audio_buffer[-self.hop_size :] = indata

        except queue.Empty:
            pass

        return self.audio_buffer.copy()

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def rms_db(self) -> float:
        """
        Compute the approximate sound level of the current buffer.

        Returns
        -------
        float
            Approximate dB level using the same method as the original
            prototype.
        """

        rms = np.sqrt(np.mean(self.audio_buffer ** 2))

        return float(20 * np.log10(rms + 1e-9) + 90)

    def channel_levels(self) -> np.ndarray:
        """
        Compute the RMS level of each microphone channel in dB.

        Returns
        -------
        numpy.ndarray
            Array of length NUM_CHANNELS containing approximate dB levels.
        """

        rms = np.sqrt(np.mean(self.audio_buffer ** 2, axis=0))
        levels = 20 * np.log10(rms + 1e-9) + 90

        return levels.astype(np.float32)

    def print_channel_status(self) -> None:
        """
        Print a formatted 4x4 table of microphone signal levels.

        The table layout matches the physical UMA-16 microphone arrangement.
        """

        levels = self.channel_levels()

        print("\\nUMA-16 Channel Levels (dB)")
        print("-" * 36)

        for row in range(4):
            values = []
            for col in range(4):
                idx = row * 4 + col
                values.append(f"CH{idx + 1:02d}: {levels[idx]:6.1f}")

            print("  ".join(values))

        print("-" * 36)

    def check_microphone_health(self, threshold_db: float = -40.0) -> list[int]:
        """
        Detect microphones that appear inactive or disconnected.

        Parameters
        ----------
        threshold_db : float
            Channels below this level are considered inactive.

        Returns
        -------
        list[int]
            List of channel indices (0-based) that appear inactive.
        """

        levels = self.channel_levels()
        inactive = [i for i, level in enumerate(levels) if level < threshold_db]

        return inactive

    # ---------------------------------------------------------
    # Context manager support
    # ---------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()