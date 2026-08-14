"""
vision.segmentation
===================

YOLO11 instance segmentation interface for the Multimodal Acoustic Camera project.

This module wraps the Ultralytics YOLO model behind a reusable class so that
the segmentation engine can be used by the vision pipeline, fusion engine,
and future deployment targets.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

from typing import Any

import cv2
import torch
from ultralytics import YOLO

from config.settings import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    YOLO_DEVICE,
    YOLO_VERBOSE,
)


class YOLOSegmenter:
    """
    Reusable YOLO11 segmentation engine.

    Example
    -------
    segmenter = YOLOSegmenter()

    results = segmenter.segment(frame)
    annotated = segmenter.annotate(frame, results)
    detections = segmenter.get_detections(results)
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
            Detection confidence threshold.
        iou : float
            Intersection-over-Union threshold.
        device : str
            Inference device ("cpu", "cuda", or "auto").
        """

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou

        # Automatic device selection
        if device.lower() == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[YOLO] Using device: {self.device}")

    def segment(self, frame):
        """
        Run YOLO11 instance segmentation on a frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Input BGR image.

        Returns
        -------
        ultralytics.engine.results.Results
            Segmentation result object.
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=YOLO_VERBOSE,
        )

        return results[0]

    def annotate(self, frame, results):
        """
        Draw segmentation masks, bounding boxes, and labels.

        Parameters
        ----------
        frame : numpy.ndarray
            Original frame.
        results : Results
            YOLO segmentation result.

        Returns
        -------
        numpy.ndarray
            Annotated visualization frame.
        """

        return results.plot()

    def get_detections(self, results) -> list[dict[str, Any]]:
        """
        Convert YOLO results into a structured list of detections.

        Returns
        -------
        list[dict]
            Each detection contains:

            - class_id
            - class_name
            - confidence
            - bbox (x1, y1, x2, y2)
            - mask (if available)
        """

        detections: list[dict[str, Any]] = []

        if results.boxes is None:
            return detections

        for i, box in enumerate(results.boxes):
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            mask = None
            if results.masks is not None:
                mask = results.masks.data[i].cpu().numpy()

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": results.names[class_id],
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "mask": mask,
                }
            )

        return detections