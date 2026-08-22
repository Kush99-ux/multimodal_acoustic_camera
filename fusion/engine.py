"""
fusion.engine
=============

First multimodal fusion engine for the
Real-Time Multimodal Acoustic Camera.

Milestone 17
------------

This module establishes the first complete software-level
fusion pipeline:

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

This first implementation intentionally performs only
temporal and data-level fusion.

It does NOT yet perform:

- object-to-sound association
- spatial projection
- acoustic coordinate correction
- camera calibration
- acoustic/visual triangulation
- source classification
- advanced confidence estimation

The acoustic localization is treated as an input measurement.
This allows the overall multimodal software architecture to
develop independently of the current acoustic localization
accuracy.

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
    First-generation multimodal fusion engine.

    Parameters
    ----------
    synchronizer : TimestampSynchronizer, optional
        Synchronization component used to determine whether
        vision and acoustic frames belong to the same time
        window.

    Notes
    -----
    The engine does not modify either input frame.
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

        Parameters
        ----------
        vision : VisionFrame
            Vision data produced by the vision pipeline.

        acoustic : AcousticFrame
            Acoustic data produced by the acoustic pipeline.

        Returns
        -------
        MultimodalResult or None
            A synchronized multimodal result if the timestamps
            are within the configured tolerance.

            Returns None when the frames cannot be synchronized.
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
        # Calculate initial fusion confidence
        # ----------------------------------------------------

        fusion_confidence = (
            self._calculate_fusion_confidence(
                synchronized_pair.timestamp_difference_ms,
                acoustic,
            )
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
        )

        self.successful_fusions += 1

        return result

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

        This is intentionally conservative.

        Temporal alignment contributes to the confidence,
        while acoustic confidence contributes as the second
        component.

        This value is NOT a scientifically validated
        source-association confidence.

        Returns
        -------
        float
            Value between 0.0 and 1.0.
        """

        # ----------------------------------------------------
        # Temporal confidence
        #
        # 0 ms difference  -> 1.0
        # tolerance        -> 0.0
        # ----------------------------------------------------

        tolerance = self.synchronizer.get_tolerance_ms()

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
        # Combine conservatively
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