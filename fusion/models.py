"""
fusion.models
=============

Data models for multimodal fusion.

This module defines the structured output produced by the
Multimodal Acoustic Camera fusion engine.

The fusion layer should depend on these project-level models
rather than directly exposing implementation-specific objects
from Ultralytics, OpenCV, MUSIC, or Acoular.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from acoustic.models import AcousticFrame
from vision.models import VisionFrame


@dataclass
class MultimodalResult:
    """
    Combined result from the vision and acoustic pipelines.

    Attributes
    ----------
    timestamp : float
        Timestamp representing the synchronized multimodal result.

    vision : VisionFrame
        Synchronized visual information.

    acoustic : AcousticFrame
        Synchronized acoustic information.

    timestamp_difference_ms : float
        Absolute time difference between the selected vision
        and acoustic frames.

    synchronized : bool
        Whether the two modalities satisfy the synchronization
        requirement.

    fusion_confidence : float
        Initial fusion confidence.

        This is deliberately a simple system-level confidence
        measure for Milestone 17. It is NOT intended to represent
        scientifically validated source-association confidence.

    status : str
        Human-readable fusion status.

    metadata : dict
        Optional additional metadata for future fusion stages.
    """

    timestamp: float

    vision: VisionFrame

    acoustic: AcousticFrame

    timestamp_difference_ms: float

    synchronized: bool

    fusion_confidence: float = 0.0

    status: str = "unknown"

    metadata: dict = field(default_factory=dict)

    # ---------------------------------------------------------
    # Convenience properties
    # ---------------------------------------------------------

    @property
    def vision_detection_count(self) -> int:
        """
        Number of visual detections in this fused result.
        """

        return len(self.vision.detections)

    @property
    def acoustic_available(self) -> bool:
        """
        Whether acoustic localization information is available.
        """

        return self.acoustic.localization is not None

    @property
    def is_valid(self) -> bool:
        """
        Whether this represents a valid synchronized result.
        """

        return (
            self.synchronized
            and self.vision is not None
            and self.acoustic is not None
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self) -> dict:
        """
        Return a lightweight serializable summary.

        This is useful for logging, debugging, and future
        communication between modules.
        """

        return {
            "timestamp": self.timestamp,
            "timestamp_difference_ms": self.timestamp_difference_ms,
            "synchronized": self.synchronized,
            "fusion_confidence": self.fusion_confidence,
            "status": self.status,
            "vision_frame_id": self.vision.frame_id,
            "vision_detection_count": self.vision_detection_count,
            "acoustic_frame_id": self.acoustic.frame_id,
            "acoustic_available": self.acoustic_available,
        }