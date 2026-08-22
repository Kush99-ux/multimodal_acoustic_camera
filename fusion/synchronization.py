"""
fusion.synchronization
======================

Timestamp synchronization layer for the
Multimodal Acoustic Camera project.

Milestone 16
------------

This module matches timestamped VisionFrame and AcousticFrame
objects using the local monotonic clock.

The synchronizer intentionally does NOT perform:

- spatial fusion
- acoustic localization
- YOLO inference
- MUSIC processing
- coordinate transformation
- confidence fusion

Its only responsibility is temporal alignment.

Architecture
------------

VisionFrame
      |
      |
      v
  Synchronizer
      ^
      |
      |
AcousticFrame

Output:

SynchronizedFramePair

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from acoustic.models import AcousticFrame
from vision.models import VisionFrame


# ============================================================
# Configuration
# ============================================================

SYNC_TOLERANCE_MS = 100.0

SYNC_TOLERANCE_SECONDS = SYNC_TOLERANCE_MS / 1000.0

# Small numerical tolerance for floating-point timestamp comparisons.
# This prevents an exact boundary value such as 100 ms from being
# rejected because of floating-point representation.
TIMESTAMP_EPSILON_SECONDS = 1e-9

# ============================================================
# Synchronized Frame Pair
# ============================================================


@dataclass(frozen=True)
class SynchronizedFramePair:
    """
    A temporally matched vision/acoustic frame pair.

    Parameters
    ----------
    vision : VisionFrame
        Camera/vision frame.

    acoustic : AcousticFrame
        Acoustic localization frame.

    timestamp_difference : float
        Absolute timestamp difference in seconds.

    """

    vision: VisionFrame
    acoustic: AcousticFrame
    timestamp_difference: float

    @property
    def timestamp_difference_ms(self) -> float:
        """
        Return timestamp difference in milliseconds.
        """

        return self.timestamp_difference * 1000.0

    @property
    def timestamp(self) -> float:
        """
        Return the midpoint timestamp of the synchronized pair.

        This is useful later when the fusion engine needs one
        representative timestamp for the combined result.
        """

        return (
            self.vision.timestamp
            + self.acoustic.timestamp
        ) / 2.0


# ============================================================
# Synchronizer
# ============================================================


class TimestampSynchronizer:
    """
    Match VisionFrame and AcousticFrame objects by timestamp.

    The synchronizer searches for the acoustic frame whose
    timestamp is closest to the vision frame timestamp.

    A match is accepted only when:

        abs(vision.timestamp - acoustic.timestamp)
            <= tolerance

    Parameters
    ----------
    tolerance_ms : float
        Maximum permitted timestamp difference in milliseconds.

    """

    def __init__(
        self,
        tolerance_ms: float = SYNC_TOLERANCE_MS,
    ) -> None:

        if tolerance_ms < 0:

            raise ValueError(
                "Synchronization tolerance cannot be negative."
            )

        self.tolerance_ms = float(tolerance_ms)

        self.tolerance_seconds = (
            self.tolerance_ms / 1000.0
        )

    # --------------------------------------------------------
    # Timestamp difference
    # --------------------------------------------------------

    @staticmethod
    def timestamp_difference(
        vision: VisionFrame,
        acoustic: AcousticFrame,
    ) -> float:
        """
        Return the absolute timestamp difference in seconds.
        """

        return abs(
            vision.timestamp
            - acoustic.timestamp
        )

    # --------------------------------------------------------
    # Pair matching
    # --------------------------------------------------------

    def match(
        self,
        vision: VisionFrame,
        acoustic_frames: Sequence[AcousticFrame],
    ) -> Optional[SynchronizedFramePair]:
        """
        Find the closest acoustic frame to a vision frame.

        Parameters
        ----------
        vision : VisionFrame
            Vision frame for which an acoustic frame is required.

        acoustic_frames : sequence of AcousticFrame
            Available acoustic frames.

        Returns
        -------
        SynchronizedFramePair or None
            Closest valid pair, or None if no acoustic frame is
            within the configured synchronization tolerance.
        """

        if not acoustic_frames:

            return None

        closest_frame: Optional[AcousticFrame] = None
        closest_difference = float("inf")

        for acoustic in acoustic_frames:

            difference = self.timestamp_difference(
                vision,
                acoustic,
            )

            if difference < closest_difference:

                closest_difference = difference
                closest_frame = acoustic

        if closest_frame is None:

            return None

        if (
            closest_difference 
            > self.tolerance_seconds + TIMESTAMP_EPSILON_SECONDS
        ):

            return None

        return SynchronizedFramePair(
            vision=vision,
            acoustic=closest_frame,
            timestamp_difference=closest_difference,
        )

    # --------------------------------------------------------
    # Boolean matching
    # --------------------------------------------------------

    def is_synchronized(
        self,
        vision: VisionFrame,
        acoustic: AcousticFrame,
    ) -> bool:
        """
        Return True if two frames are within the tolerance.
        """

        difference = self.timestamp_difference(
            vision,
            acoustic,
        )

        return (
            difference <= self.tolerance_seconds + TIMESTAMP_EPSILON_SECONDS
        )

    # --------------------------------------------------------
    # Direct pair synchronization
    # --------------------------------------------------------

    def synchronize_pair(
        self,
        vision: VisionFrame,
        acoustic: AcousticFrame,
    ) -> Optional[SynchronizedFramePair]:
        """
        Synchronize exactly one vision/acoustic pair.

        This is useful when a caller has already selected the
        candidate acoustic frame.
        """

        difference = self.timestamp_difference(
            vision,
            acoustic,
        )

        if (
            difference > self.tolerance_seconds + TIMESTAMP_EPSILON_SECONDS
        ):

            return None

        return SynchronizedFramePair(
            vision=vision,
            acoustic=acoustic,
            timestamp_difference=difference,
        )

    # --------------------------------------------------------
    # Configuration information
    # --------------------------------------------------------

    def get_tolerance_ms(self) -> float:
        """
        Return synchronization tolerance in milliseconds.
        """

        return self.tolerance_ms