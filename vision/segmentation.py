"""
vision.segmentation
===================

YOLO11 instance segmentation interface for the Multimodal Acoustic Camera.

This module provides a reusable abstraction around the Ultralytics YOLO11
segmentation model.

Responsibilities
----------------
- Load the YOLO11 segmentation model.
- Run inference on OpenCV BGR frames.
- Convert YOLO results into project-level Detection objects.
- Extract bounding boxes, centroids, masks, classes, and confidence.
- Provide optional annotated frames for debugging.

This module does NOT:
- Acquire camera frames.
- Perform acoustic localization.
- Perform camera/audio synchronization.
- Perform multimodal fusion.
- Determine physical 3-D source position.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config.settings import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    YOLO_DEVICE,
    YOLO_VERBOSE,
)


# ============================================================
# Detection Data Model
# ============================================================


@dataclass
class Detection:
    """
    Project-level representation of a single YOLO detection.

    Coordinates are expressed in camera-image pixel coordinates.

    Attributes
    ----------
    class_id : int
        Numeric YOLO class ID.

    class_name : str
        Human-readable class name.

    confidence : float
        YOLO confidence score.

    bbox : tuple[float, float, float, float]
        Bounding box in the form:
        (x1, y1, x2, y2)

    bbox_centroid : tuple[float, float]
        Center of the bounding box:
        (cx, cy)

    mask_centroid : Optional[tuple[float, float]]
        Centroid calculated from the segmentation mask.
        None when a valid mask is unavailable.

    mask : Optional[np.ndarray]
        Binary segmentation mask at the original camera-frame
        resolution.

    """

    class_id: int
    class_name: str
    confidence: float

    bbox: tuple[float, float, float, float]

    bbox_centroid: tuple[float, float]

    mask_centroid: Optional[tuple[float, float]]

    mask: Optional[np.ndarray]


# ============================================================
# YOLO Segmentation Engine
# ============================================================


class YOLOSegmenter:
    """
    Reusable YOLO11 instance segmentation engine.

    Example
    -------
    segmenter = YOLOSegmenter()

    results = segmenter.segment(frame)

    detections = segmenter.get_detections(
        results,
        frame.shape,
    )

    annotated = segmenter.annotate(frame, results)
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        confidence: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        device: str = YOLO_DEVICE,
    ) -> None:
        """
        Initialize the YOLO11 segmentation engine.

        Parameters
        ----------
        model_path : str
            Path to the YOLO11 segmentation model.

        confidence : float
            Minimum detection confidence.

        iou : float
            Non-maximum suppression IoU threshold.

        device : str
            Inference device.

            Supported values:
            - "cpu"
            - "cuda"
            - "auto"
        """

        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou

        # ----------------------------------------------------
        # Device selection
        # ----------------------------------------------------

        requested_device = device.lower()

        if requested_device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        elif requested_device == "cuda":
            if not torch.cuda.is_available():
                print(
                    "[YOLO] CUDA requested but unavailable. "
                    "Falling back to CPU."
                )
                self.device = "cpu"
            else:
                self.device = "cuda"

        else:
            self.device = requested_device

        print(f"[YOLO] Loading model: {self.model_path}")
        print(f"[YOLO] Using device: {self.device}")

        # ----------------------------------------------------
        # Model loading
        # ----------------------------------------------------

        self.model = YOLO(self.model_path)

        print("[YOLO] Model loaded successfully.")

    # ========================================================
    # Input Validation
    # ========================================================

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """
        Validate an OpenCV camera frame.
        """

        if frame is None:
            raise ValueError(
                "Input frame is None."
            )

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "Input frame must be a numpy.ndarray."
            )

        if frame.ndim != 3:
            raise ValueError(
                "Input frame must have shape "
                "(height, width, channels)."
            )

        if frame.shape[2] != 3:
            raise ValueError(
                "Input frame must contain 3 channels."
            )

        if frame.dtype != np.uint8:
            raise ValueError(
                "Input frame must use dtype uint8."
            )

    # ========================================================
    # Inference
    # ========================================================

    def segment(self, frame: np.ndarray):
        """
        Run YOLO11 instance segmentation on one camera frame.

        Parameters
        ----------
        frame : numpy.ndarray
            OpenCV BGR image.

        Returns
        -------
        ultralytics.engine.results.Results
            YOLO result object for the frame.
        """

        self._validate_frame(frame)

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=YOLO_VERBOSE,
        )

        if not results:
            return None

        return results[0]

    # ========================================================
    # Mask Processing
    # ========================================================

    @staticmethod
    def _extract_mask(
        results: Any,
        detection_index: int,
        frame_shape: tuple[int, ...],
    ) -> Optional[np.ndarray]:
        """
        Extract one YOLO segmentation mask and resize it to
        the original camera-frame resolution.

        Returns
        -------
        numpy.ndarray or None
            Binary uint8 mask with shape (height, width).
        """

        if results is None:
            return None

        if results.masks is None:
            return None

        if detection_index >= len(results.masks.data):
            return None

        frame_height = frame_shape[0]
        frame_width = frame_shape[1]

        mask_tensor = results.masks.data[detection_index]

        mask = mask_tensor.detach().cpu().numpy()

        # Convert model mask into binary mask.
        mask = (mask > 0.5).astype(np.uint8)

        # Resize to original camera resolution.
        mask = cv2.resize(
            mask,
            (frame_width, frame_height),
            interpolation=cv2.INTER_NEAREST,
        )

        return mask

    # ========================================================
    # Centroid Calculation
    # ========================================================

    @staticmethod
    def _bbox_centroid(
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        """
        Calculate bounding-box centroid.
        """

        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        return cx, cy

    @staticmethod
    def _mask_centroid(
        mask: Optional[np.ndarray],
    ) -> Optional[tuple[float, float]]:
        """
        Calculate the centroid of a binary segmentation mask.

        Returns
        -------
        tuple or None
            (cx, cy) in image pixel coordinates.
        """

        if mask is None:
            return None

        moments = cv2.moments(mask, binaryImage=True)

        if moments["m00"] <= 0:
            return None

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        return float(cx), float(cy)

    # ========================================================
    # Structured Detection Extraction
    # ========================================================

    def get_detections(
        self,
        results: Any,
        frame_shape: tuple[int, ...],
    ) -> list[Detection]:
        """
        Convert YOLO results into project-level Detection objects.

        Parameters
        ----------
        results
            YOLO Results object returned by segment().

        frame_shape
            Shape of the original camera frame.

        Returns
        -------
        list[Detection]
            Structured project-level detections.
        """

        detections: list[Detection] = []

        if results is None:
            return detections

        if results.boxes is None:
            return detections

        for index, box in enumerate(results.boxes):

            # ------------------------------------------------
            # Class
            # ------------------------------------------------

            class_id = int(
                box.cls.item()
            )

            class_name = str(
                results.names[class_id]
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = float(
                box.conf.item()
            )

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

            bbox = (
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            )

            # ------------------------------------------------
            # Bounding-box centroid
            # ------------------------------------------------

            bbox_centroid = self._bbox_centroid(
                bbox
            )

            # ------------------------------------------------
            # Segmentation mask
            # ------------------------------------------------

            mask = self._extract_mask(
                results,
                index,
                frame_shape,
            )

            # ------------------------------------------------
            # Mask centroid
            # ------------------------------------------------

            mask_centroid = self._mask_centroid(
                mask
            )

            # ------------------------------------------------
            # Detection object
            # ------------------------------------------------

            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                bbox_centroid=bbox_centroid,
                mask_centroid=mask_centroid,
                mask=mask,
            )

            detections.append(detection)

        return detections

    # ========================================================
    # Convenience Method
    # ========================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[Detection]:
        """
        Run segmentation and directly return structured detections.

        This is the primary high-level interface for future
        vision and fusion modules.

        Example
        -------
        detections = segmenter.detect(frame)
        """

        results = self.segment(frame)

        if results is None:
            return []

        return self.get_detections(
            results,
            frame.shape,
        )

    # ========================================================
    # Visualization
    # ========================================================

    def annotate(
        self,
        frame: np.ndarray,
        results: Any,
    ) -> np.ndarray:
        """
        Generate an annotated visualization frame.

        This function is intended only for debugging and
        visualization.

        It does not modify the original frame.
        """

        self._validate_frame(frame)

        if results is None:
            return frame.copy()

        return results.plot(
            img=frame.copy()
        )

    # ========================================================
    # Utility
    # ========================================================

    def class_names(self) -> dict[int, str]:
        """
        Return the class-name mapping used by the loaded model.
        """

        return dict(self.model.names)

    def __repr__(self) -> str:
        return (
            f"YOLOSegmenter("
            f"model='{self.model_path}', "
            f"device='{self.device}', "
            f"confidence={self.confidence}, "
            f"iou={self.iou}"
            f")"
        )