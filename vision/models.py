"""
vision.models
=============

Project-level data models for the vision pipeline.

These models provide a stable interface between the vision subsystem
and downstream modules such as synchronization and multimodal fusion.

The models intentionally do NOT depend on Ultralytics.

Pipeline
--------
ESP32-CAM
    ↓
CameraAcquisition
    ↓
CameraFrame
    ↓
YOLOSegmenter
    ↓
VisionDetection
    ↓
VisionFrame
    ↓
Fusion / Synchronization

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================
# Vision Detection
# ============================================================

@dataclass
class VisionDetection:
    """
    Representation of one detected visual object.

    This is the project-level representation of a YOLO detection.

    It deliberately contains no Ultralytics-specific objects.

    Attributes
    ----------
    class_id : int
        Numeric class identifier.

    class_name : str
        Human-readable class name.

    confidence : float
        Detection confidence in the range [0, 1].

    bbox : tuple[float, float, float, float]
        Bounding box in pixel coordinates:

        (x1, y1, x2, y2)

    bbox_centroid : tuple[float, float]
        Center of the bounding box in pixel coordinates:

        (cx, cy)

    mask : numpy.ndarray or None
        Binary segmentation mask.

    mask_centroid : tuple[float, float] or None
        Centroid of the segmentation mask in pixel coordinates.
    """

    class_id: int
    class_name: str
    confidence: float

    bbox: tuple[float, float, float, float]

    bbox_centroid: tuple[float, float]

    mask: Optional[np.ndarray] = None

    mask_centroid: Optional[tuple[float, float]] = None


# ============================================================
# Vision Frame
# ============================================================

@dataclass
class VisionFrame:
    """
    Project-level representation of one processed camera frame.

    A VisionFrame contains the timestamp and frame identity inherited
    from the camera acquisition stage together with the detections
    produced by the vision pipeline.

    The raw camera image is intentionally NOT stored here.

    This keeps the data model lightweight and prevents downstream
    modules from becoming dependent on OpenCV image processing.

    Attributes
    ----------
    timestamp : float
        Camera acquisition timestamp.

    frame_id : int
        Sequential camera frame identifier.

    detections : list[VisionDetection]
        Objects detected in this frame.

    width : int
        Width of the processed camera frame in pixels.

    height : int
        Height of the processed camera frame in pixels.

    inference_time_ms : float or None
        YOLO inference time for this frame in milliseconds.
    """

    timestamp: float
    frame_id: int

    detections: list[VisionDetection]

    width: int
    height: int

    inference_time_ms: Optional[float] = None

    # --------------------------------------------------------
    # Convenience properties
    # --------------------------------------------------------

    @property
    def detection_count(self) -> int:
        """
        Return the number of detections in this frame.
        """

        return len(self.detections)

    def get_detections_by_class(
        self,
        class_name: str,
    ) -> list[VisionDetection]:
        """
        Return all detections matching a class name.

        Parameters
        ----------
        class_name : str
            Class name to search for.

        Returns
        -------
        list[VisionDetection]
            Matching detections.
        """

        return [
            detection
            for detection in self.detections
            if detection.class_name == class_name
        ]

    def get_highest_confidence_detection(
        self,
    ) -> Optional[VisionDetection]:
        """
        Return the highest-confidence detection.

        Returns
        -------
        VisionDetection or None
            Highest-confidence detection, or None if no objects
            were detected.
        """

        if not self.detections:
            return None

        return max(
            self.detections,
            key=lambda detection: detection.confidence,
        )