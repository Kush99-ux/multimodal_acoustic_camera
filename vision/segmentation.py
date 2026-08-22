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
- Convert YOLO results into project-level VisionDetection objects.
- Extract bounding boxes, centroids, masks, classes, and confidence.
- Provide optional annotated frames for debugging.

This module does NOT:
- Acquire camera frames.
- Perform acoustic localization.
- Perform camera/audio synchronization.
- Perform multimodal fusion.
- Determine physical 3-D source position.

Architecture
------------
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
Synchronization / Fusion

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

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

from vision.models import VisionDetection


class YOLOSegmenter:
    """
    Reusable YOLO11 instance segmentation engine.

    The class isolates all Ultralytics-specific functionality inside
    the vision subsystem.

    Downstream modules should work with VisionDetection objects rather
    than directly accessing Ultralytics Results objects.

    Example
    -------
    segmenter = YOLOSegmenter()

    results = segmenter.segment(frame)

    detections = segmenter.get_detections(
        results,
        frame.shape,
    )

    annotated = segmenter.annotate(
        frame,
        results,
    )
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

        # ====================================================
        # Device selection
        # ====================================================

        requested_device = device.lower().strip()

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

        elif requested_device == "cpu":

            self.device = "cpu"

        else:

            raise ValueError(
                f"Unsupported YOLO device: '{device}'. "
                "Use 'cpu', 'cuda', or 'auto'."
            )

        # ====================================================
        # Model loading
        # ====================================================

        print(
            f"[YOLO] Loading model: {self.model_path}"
        )

        print(
            f"[YOLO] Using device: {self.device}"
        )

        self.model = YOLO(self.model_path)

        print(
            "[YOLO] Model loaded successfully."
        )

    # ========================================================
    # Input validation
    # ========================================================

    @staticmethod
    def _validate_frame(
        frame: np.ndarray,
    ) -> None:
        """
        Validate an OpenCV camera frame.

        Expected format
        ----------------
        dtype  : uint8
        shape  : (height, width, 3)
        format : BGR
        """

        if frame is None:

            raise ValueError(
                "Input frame is None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):

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
                "Input frame must contain exactly "
                "3 color channels."
            )

        if frame.dtype != np.uint8:

            raise ValueError(
                "Input frame must use dtype uint8."
            )

    # ========================================================
    # Inference
    # ========================================================

    def segment(
        self,
        frame: np.ndarray,
    ) -> Optional[Any]:
        """
        Run YOLO11 instance segmentation on one camera frame.

        Parameters
        ----------
        frame : numpy.ndarray
            OpenCV BGR image.

        Returns
        -------
        Results or None
            Ultralytics Results object for the frame.

        Notes
        -----
        This is the only method downstream code should need to call
        before converting the result into VisionDetection objects.
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
    # Mask extraction
    # ========================================================

    @staticmethod
    def _extract_mask(
        results: Any,
        detection_index: int,
        frame_shape: tuple[int, ...],
    ) -> Optional[np.ndarray]:
        """
        Extract one YOLO segmentation mask.

        The mask is converted into a binary uint8 mask and resized
        to the original camera-frame resolution.

        Parameters
        ----------
        results :
            Ultralytics Results object.

        detection_index : int
            Index of the detection.

        frame_shape : tuple
            Shape of the original OpenCV frame.

        Returns
        -------
        numpy.ndarray or None
            Binary mask with shape:

            (frame_height, frame_width)

            Values are 0 or 1.
        """

        if results is None:
            return None

        if results.masks is None:
            return None

        if detection_index >= len(
            results.masks.data
        ):
            return None

        frame_height = int(
            frame_shape[0]
        )

        frame_width = int(
            frame_shape[1]
        )

        mask_tensor = (
            results
            .masks
            .data[detection_index]
        )

        mask = (
            mask_tensor
            .detach()
            .cpu()
            .numpy()
        )

        # Convert probability mask to binary mask.
        mask = (
            mask > 0.5
        ).astype(
            np.uint8
        )

        # Resize to original camera resolution.
        mask = cv2.resize(
            mask,
            (
                frame_width,
                frame_height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        return mask

    # ========================================================
    # Bounding-box centroid
    # ========================================================

    @staticmethod
    def _bbox_centroid(
        bbox: tuple[
            float,
            float,
            float,
            float,
        ],
    ) -> tuple[float, float]:
        """
        Calculate the bounding-box centroid.

        Parameters
        ----------
        bbox :
            (x1, y1, x2, y2)

        Returns
        -------
        tuple[float, float]
            (cx, cy)
        """

        x1, y1, x2, y2 = bbox

        cx = (
            x1 + x2
        ) / 2.0

        cy = (
            y1 + y2
        ) / 2.0

        return (
            float(cx),
            float(cy),
        )

    # ========================================================
    # Segmentation-mask centroid
    # ========================================================

    @staticmethod
    def _mask_centroid(
        mask: Optional[np.ndarray],
    ) -> Optional[tuple[float, float]]:
        """
        Calculate the centroid of a binary segmentation mask.

        Parameters
        ----------
        mask : numpy.ndarray or None
            Binary mask.

        Returns
        -------
        tuple[float, float] or None
            (cx, cy) in image pixel coordinates.
        """

        if mask is None:
            return None

        moments = cv2.moments(
            mask,
            binaryImage=True,
        )

        if moments["m00"] <= 0:

            return None

        cx = (
            moments["m10"]
            / moments["m00"]
        )

        cy = (
            moments["m01"]
            / moments["m00"]
        )

        return (
            float(cx),
            float(cy),
        )

    # ========================================================
    # Structured detection extraction
    # ========================================================

    def get_detections(
        self,
        results: Any,
        frame_shape: tuple[int, ...] | None = None,
    ) -> list[VisionDetection]:
        """
        Convert Ultralytics YOLO results into project-level
        VisionDetection objects.

        This method is the architectural boundary between YOLO
        and the rest of the project.

        Parameters
        ----------
        results :
            Ultralytics Results object.

        frame_shape : tuple or None
            Shape of the original OpenCV camera frame.

            Required when segmentation masks need to be resized
            to the original frame resolution.

            If omitted, the method attempts to use the shape
            stored by Ultralytics in results.orig_shape.

        Returns
        -------
        list[VisionDetection]
            Project-level detections.

        Notes
        -----
        The returned objects contain no dependency on the
        Ultralytics Results API.
        """

        if results is None:

            return []

        # ----------------------------------------------------
        # Determine frame shape
        # ----------------------------------------------------

        if frame_shape is None:

            original_shape = getattr(
                results,
                "orig_shape",
                None,
            )

            if original_shape is not None:

                frame_shape = (
                    int(original_shape[0]),
                    int(original_shape[1]),
                )

            else:

                frame_shape = None

        # ----------------------------------------------------
        # No detections
        # ----------------------------------------------------

        if results.boxes is None:

            return []

        if len(results.boxes) == 0:

            return []

        detections: list[
            VisionDetection
        ] = []

        # ====================================================
        # Convert every YOLO detection
        # ====================================================

        for i, box in enumerate(
            results.boxes
        ):

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
                box
                .xyxy[0]
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

            bbox_centroid = (
                self._bbox_centroid(
                    bbox
                )
            )

            # ------------------------------------------------
            # Segmentation mask
            # ------------------------------------------------

            mask = None

            mask_centroid = None

            if frame_shape is not None:

                mask = (
                    self._extract_mask(
                        results,
                        i,
                        frame_shape,
                    )
                )

                mask_centroid = (
                    self._mask_centroid(
                        mask
                    )
                )

            # ------------------------------------------------
            # Create project-level object
            # ------------------------------------------------

            detection = VisionDetection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                bbox_centroid=bbox_centroid,
                mask=mask,
                mask_centroid=mask_centroid,
            )

            detections.append(
                detection
            )

        return detections

    # ========================================================
    # High-level detection interface
    # ========================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[VisionDetection]:
        """
        Run YOLO segmentation and directly return
        project-level VisionDetection objects.

        This is the preferred high-level interface for
        downstream modules.

        Parameters
        ----------
        frame : numpy.ndarray
            OpenCV BGR camera frame.

        Returns
        -------
        list[VisionDetection]
            Structured project-level detections.

        Example
        -------
        detections = segmenter.detect(frame)

        for detection in detections:
            print(
                detection.class_name,
                detection.confidence,
                detection.bbox_centroid,
            )
        """

        self._validate_frame(frame)

        results = self.segment(
            frame
        )

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

        This method is intended for debugging and visualization
        only.

        Parameters
        ----------
        frame : numpy.ndarray
            Original OpenCV BGR frame.

        results :
            Ultralytics Results object.

        Returns
        -------
        numpy.ndarray
            Annotated BGR frame.

        Notes
        -----
        The original frame is never modified.
        """

        self._validate_frame(
            frame
        )

        if results is None:

            return frame.copy()

        annotated = results.plot(
            img=frame.copy()
        )

        return annotated

    # ========================================================
    # Model information
    # ========================================================

    def class_names(
        self,
    ) -> dict[int, str]:
        """
        Return the class-name mapping used by the loaded model.
        """

        return dict(
            self.model.names
        )

    # ========================================================
    # Representation
    # ========================================================

    def __repr__(self) -> str:

        return (
            "YOLOSegmenter("
            f"model='{self.model_path}', "
            f"device='{self.device}', "
            f"confidence={self.confidence}, "
            f"iou={self.iou}"
            ")"
        )