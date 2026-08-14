"""
vision.pipeline
===============

Unified vision pipeline for the Multimodal Acoustic Camera project.

This module combines the ESP32 camera interface and the YOLO11 segmentation
engine into a single reusable processing pipeline.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

from vision.stream import ESP32Camera
from vision.segmentation import YOLOSegmenter


class VisionPipeline:
    """
    Unified camera + segmentation pipeline.

    Example
    -------
    vision = VisionPipeline()
    vision.connect()

    frame, annotated, detections = vision.process_frame()

    vision.release()
    """

    def __init__(self) -> None:
        self.camera = ESP32Camera()
        self.segmenter = YOLOSegmenter()

    def connect(self) -> bool:
        """
        Connect to the ESP32 camera stream.

        Returns
        -------
        bool
            True if the camera connection succeeds.
        """
        return self.camera.connect()

    def process_frame(self):
        """
        Acquire a frame and run YOLO11 segmentation.

        Returns
        -------
        tuple
            (frame, annotated_frame, detections)

            frame : Original camera frame
            annotated_frame : Frame with masks and labels
            detections : Structured detection list
        """

        frame = self.camera.read()

        if frame is None:
            return None, None, []

        results = self.segmenter.segment(frame)
        annotated = self.segmenter.annotate(frame, results)
        detections = self.segmenter.get_detections(results)

        return frame, annotated, detections

    def release(self) -> None:
        """
        Release all vision resources.
        """
        self.camera.release()