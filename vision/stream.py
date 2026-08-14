"""
vision.stream
=============

ESP32-CAM streaming interface for the Multimodal Acoustic Camera project.

This module provides a reusable camera interface that connects to the
ESP32-CAM MJPEG stream and exposes a simple API for reading frames.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera

=============
"""

from __future__ import annotations

import time
from typing import Optional

import cv2

from config.settings import STREAM_URL


class ESP32Camera:
    """
    Reusable interface for the ESP32-CAM MJPEG stream.

    Example
    -------
    camera = ESP32Camera()
    camera.connect()

    frame = camera.read()

    camera.release()
    """

    def __init__(
        self,
        stream_url: str = STREAM_URL,
        reconnect_delay: float = 1.0,
    ) -> None:
        self.stream_url = stream_url
        self.reconnect_delay = reconnect_delay
        self.capture: Optional[cv2.VideoCapture] = None

    def connect(self) -> bool:
        """
        Connect to the ESP32-CAM stream.

        Returns
        -------
        bool
            True if the connection succeeds.
        """
        self.capture = cv2.VideoCapture(self.stream_url)

        if not self.capture.isOpened():
            self.capture = None
            return False

        return True

    def is_connected(self) -> bool:
        """Return True if the stream is currently open."""
        return self.capture is not None and self.capture.isOpened()

    def read(self):
        """
        Read a single frame from the camera.

        Returns
        -------
        frame : numpy.ndarray | None
            The captured frame, or None if reading fails.
        """
        if not self.is_connected():
            if not self.connect():
                return None

        ret, frame = self.capture.read()

        if not ret:
            self._reconnect()

            if not self.is_connected():
                return None

            ret, frame = self.capture.read()

            if not ret:
                return None

        return frame

    def _reconnect(self) -> None:
        """Attempt to reconnect to the ESP32 stream."""
        self.release()
        time.sleep(self.reconnect_delay)
        self.connect()

    def release(self) -> None:
        """Release the camera stream."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None