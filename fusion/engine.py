"""
fusion.engine
=============

Multimodal fusion engine for the
Real-Time Multimodal Acoustic Camera.

Milestone 21
------------

This module performs temporal/data-level fusion and prepares the
acoustic spatial information required by the visual heatmap stage.

Pipeline:

    VisionFrame
          +
    AcousticFrame
          |
          v
    TimestampSynchronizer
          |
          v
    FusionEngine
          |
          v
    MultimodalResult
          |
          v
    Acoustic spatial visualization

The engine still does NOT perform camera-acoustic spatial calibration.
It only preserves acoustic localization/spatial information so that
the visualization layer can use it.

It does NOT perform:

- camera calibration
- coordinate transformation
- object-to-sound association
- acoustic/visual triangulation
- source classification
- scientifically validated confidence estimation

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

from typing import Optional

from acoustic.models import AcousticFrame
from vision.models import VisionFrame

from fusion.models import MultimodalResult
from fusion.synchronization import (
    SYNC_TOLERANCE_MS,
    TimestampSynchronizer,
)


class FusionEngine:
    """
    Multimodal fusion engine.

    The engine synchronizes a VisionFrame and an AcousticFrame
    and packages the result into a MultimodalResult.

    Acoustic spatial information already produced by the MUSIC
    stage is preserved through the result metadata.

    Parameters
    ----------
    synchronizer : TimestampSynchronizer, optional
        Synchronization component used to determine whether
        vision and acoustic frames belong to the same time
        window.
    """

    def __init__(
        self,
        synchronizer: Optional[TimestampSynchronizer] = None,
    ) -> None:

        if synchronizer is None:
            synchronizer = TimestampSynchronizer(
                tolerance_ms=SYNC_TOLERANCE_MS,
            )

        self.synchronizer = synchronizer

        self.processed_count = 0
        self.successful_fusions = 0
        self.rejected_count = 0

    # ========================================================
    # Fusion
    # ========================================================

    def fuse(
        self,
        vision: VisionFrame,
        acoustic: AcousticFrame,
    ) -> Optional[MultimodalResult]:
        """
        Attempt to fuse one VisionFrame and one AcousticFrame.

        The timestamps must fall within the configured
        synchronization tolerance.

        Returns
        -------
        MultimodalResult or None
            Synchronized multimodal result, or None when the
            frames cannot be synchronized.
        """

        self.processed_count += 1

        synchronized_pair = (
            self.synchronizer.synchronize_pair(
                vision,
                acoustic,
            )
        )

        # ----------------------------------------------------
        # No temporal match
        # ----------------------------------------------------

        if synchronized_pair is None:

            self.rejected_count += 1

            return None

        # ----------------------------------------------------
        # Calculate fusion confidence
        # ----------------------------------------------------

        fusion_confidence = (
            self._calculate_fusion_confidence(
                synchronized_pair.timestamp_difference_ms,
                acoustic,
            )
        )

        # ----------------------------------------------------
        # Prepare fusion metadata
        # ----------------------------------------------------

        metadata = self._build_fusion_metadata(
            acoustic=acoustic,
            vision=vision,
            timestamp_difference_ms=(
                synchronized_pair.timestamp_difference_ms
            ),
        )

        # ----------------------------------------------------
        # Create multimodal result
        # ----------------------------------------------------

        result = MultimodalResult(
            timestamp=synchronized_pair.timestamp,
            vision=synchronized_pair.vision,
            acoustic=synchronized_pair.acoustic,
            timestamp_difference_ms=(
                synchronized_pair.timestamp_difference_ms
            ),
            synchronized=True,
            fusion_confidence=fusion_confidence,
            status="synchronized",
            metadata=metadata,
        )

        self.successful_fusions += 1

        return result

    # ========================================================
    # Fusion metadata
    # ========================================================

    @staticmethod
    def _build_fusion_metadata(
        acoustic: AcousticFrame,
        vision: VisionFrame,
        timestamp_difference_ms: float,
    ) -> dict:
        """
        Build metadata required by downstream visualization.

        This method deliberately does not transform acoustic
        coordinates into camera coordinates.

        The acoustic localization remains in the acoustic array's
        coordinate system until calibration is implemented.
        """

        metadata = {
            "fusion_stage": "temporal",
            "spatial_projection_available": False,
            "camera_acoustic_calibrated": False,
            "timestamp_difference_ms": (
                float(timestamp_difference_ms)
            ),
            "vision_resolution": (
                int(vision.width),
                int(vision.height),
            ),
        }

        # ----------------------------------------------------
        # Acoustic localization
        # ----------------------------------------------------

        if acoustic.localization is not None:

            localization = acoustic.localization

            metadata["acoustic_localization"] = {
                "x": float(localization.x),
                "y": float(localization.y),
                "z": float(localization.z),
                "response": float(localization.response),
                "confidence": float(localization.confidence),
            }

        else:

            metadata["acoustic_localization"] = None

        # ----------------------------------------------------
        # Preserve MUSIC metadata if present
        # ----------------------------------------------------

        if acoustic.metadata:

            metadata["music"] = dict(acoustic.metadata)

        else:

            metadata["music"] = {}

        return metadata

    # ========================================================
    # Confidence
    # ========================================================

    def _calculate_fusion_confidence(
        self,
        timestamp_difference_ms: float,
        acoustic: AcousticFrame,
    ) -> float:
        """
        Calculate an initial system-level fusion confidence.

        Temporal alignment and acoustic confidence are combined
        multiplicatively.

        This is NOT a scientifically validated confidence metric.
        """

        tolerance = self.synchronizer.get_tolerance_ms()

        # ----------------------------------------------------
        # Temporal confidence
        # ----------------------------------------------------

        if tolerance <= 0.0:

            temporal_confidence = 1.0

        else:

            temporal_confidence = max(
                0.0,
                1.0
                - (
                    timestamp_difference_ms
                    / tolerance
                ),
            )

        # ----------------------------------------------------
        # Acoustic confidence
        # ----------------------------------------------------

        acoustic_confidence = 0.0

        if acoustic.localization is not None:

            acoustic_confidence = float(
                acoustic.localization.confidence
            )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        confidence = (
            temporal_confidence
            * acoustic_confidence
        )

        return max(
            0.0,
            min(1.0, confidence),
        )

    # ========================================================
    # Statistics
    # ========================================================

    def get_processed_count(self) -> int:
        """
        Return total number of fusion attempts.
        """

        return self.processed_count

    def get_successful_fusion_count(self) -> int:
        """
        Return number of successful synchronized fusions.
        """

        return self.successful_fusions

    def get_rejected_count(self) -> int:
        """
        Return number of rejected fusion attempts.
        """

        return self.rejected_count

    def get_statistics(self) -> dict[str, int]:
        """
        Return fusion processing statistics.
        """

        return {
            "processed": self.processed_count,
            "successful": self.successful_fusions,
            "rejected": self.rejected_count,
        }