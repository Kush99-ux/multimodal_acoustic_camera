"""
acoustic.models
===============

Project-level data models for the acoustic subsystem.

Milestone 15
------------

This module provides a clean interface between the acoustic
localization implementation and the future multimodal fusion
engine.

The fusion layer must not depend directly on:
    - MUSIC internals
    - Acoular objects
    - steering matrices
    - covariance matrices
    - eigenvectors
    - raw audio buffers

Instead, the acoustic subsystem exposes:

    AcousticLocalization
            ↓
       AcousticFrame

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import math


# ============================================================
# Acoustic Localization Result
# ============================================================


@dataclass(frozen=True)
class AcousticLocalization:
    """
    Structured result produced by the acoustic localization layer.

    This represents one localization estimate from MUSIC.

    Parameters
    ----------
    x : float
        Estimated source X coordinate in metres.

    y : float
        Estimated source Y coordinate in metres.

    z : float
        Estimated source Z coordinate in metres.

    response : float
        MUSIC peak response / normalized localization response.

        This is an algorithmic response value and should NOT
        automatically be interpreted as a probability.

    frequency : float, optional
        Dominant or analyzed acoustic frequency in Hz.

    confidence : float, optional
        Optional application-level confidence value.

        This is deliberately separate from MUSIC response because
        MUSIC response itself is not necessarily a calibrated
        probability.

    valid : bool
        Whether this localization estimate should be considered
        usable by downstream systems.

    source_count : int
        Number of acoustic sources represented by this estimate.

    """

    x: float
    y: float
    z: float

    response: float

    frequency: Optional[float] = None

    confidence: Optional[float] = None

    valid: bool = True

    source_count: int = 1

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def __post_init__(self) -> None:

        values = (
            self.x,
            self.y,
            self.z,
            self.response,
        )

        for value in values:

            if not math.isfinite(float(value)):

                raise ValueError(
                    "Acoustic localization coordinates and "
                    "response must be finite."
                )

        if self.frequency is not None:

            if (
                not math.isfinite(float(self.frequency))
                or self.frequency < 0
            ):

                raise ValueError(
                    "frequency must be a finite non-negative value."
                )

        if self.confidence is not None:

            if not math.isfinite(float(self.confidence)):

                raise ValueError(
                    "confidence must be finite when provided."
                )

        if self.source_count < 0:

            raise ValueError(
                "source_count cannot be negative."
            )

    # --------------------------------------------------------
    # Convenience properties
    # --------------------------------------------------------

    @property
    def position(self) -> tuple[float, float, float]:
        """
        Return the estimated 3-D position.
        """

        return (
            self.x,
            self.y,
            self.z,
        )

    @property
    def xy_position(self) -> tuple[float, float]:
        """
        Return the estimated X/Y position.
        """

        return (
            self.x,
            self.y,
        )

    @property
    def has_position(self) -> bool:
        """
        Return whether a usable position exists.
        """

        return self.valid

    def distance_from_origin(self) -> float:
        """
        Return Euclidean distance from the acoustic-array origin.

        This is a geometric quantity derived from the localization
        coordinates. It is NOT a separate acoustic range estimate.
        """

        return math.sqrt(
            self.x ** 2
            + self.y ** 2
            + self.z ** 2
        )


# ============================================================
# Acoustic Frame
# ============================================================


@dataclass(frozen=True)
class AcousticFrame:
    """
    Timestamped acoustic localization frame.

    This is the primary object exposed to the synchronization
    and fusion layers.

    Parameters
    ----------
    frame_id : int
        Sequential acoustic frame identifier.

    timestamp : float
        Acquisition/processing timestamp using the same local
        monotonic clock domain used by the camera subsystem.

    localization : AcousticLocalization, optional
        Localization result associated with this frame.

    sample_rate : float, optional
        Audio sampling rate in Hz.

    frame_size : int, optional
        Number of audio samples represented by the processing frame.

    processing_time : float, optional
        Localization processing time in seconds.

    """

    frame_id: int

    timestamp: float

    localization: Optional[AcousticLocalization] = None

    sample_rate: Optional[float] = None

    frame_size: Optional[int] = None

    processing_time: Optional[float] = None

    # --------------------------------------------------------
    # Optional metadata
    # --------------------------------------------------------

    metadata: dict = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def __post_init__(self) -> None:

        if self.frame_id < 0:

            raise ValueError(
                "frame_id cannot be negative."
            )

        if not math.isfinite(float(self.timestamp)):

            raise ValueError(
                "timestamp must be finite."
            )

        if self.sample_rate is not None:

            if (
                not math.isfinite(float(self.sample_rate))
                or self.sample_rate <= 0
            ):

                raise ValueError(
                    "sample_rate must be positive."
                )

        if self.frame_size is not None:

            if self.frame_size <= 0:

                raise ValueError(
                    "frame_size must be positive."
                )

        if self.processing_time is not None:

            if (
                not math.isfinite(float(self.processing_time))
                or self.processing_time < 0
            ):

                raise ValueError(
                    "processing_time must be non-negative."
                )

    # --------------------------------------------------------
    # Convenience properties
    # --------------------------------------------------------

    @property
    def has_localization(self) -> bool:
        """
        Return True when this frame contains a valid localization.
        """

        return (
            self.localization is not None
            and self.localization.valid
        )

    @property
    def position(
        self,
    ) -> Optional[tuple[float, float, float]]:
        """
        Return the acoustic source position if available.
        """

        if not self.has_localization:

            return None

        return self.localization.position

    @property
    def confidence(self) -> Optional[float]:
        """
        Return localization confidence if available.
        """

        if self.localization is None:

            return None

        return self.localization.confidence

    @property
    def response(self) -> Optional[float]:
        """
        Return MUSIC response if available.
        """

        if self.localization is None:

            return None

        return self.localization.response

    @property
    def frequency(self) -> Optional[float]:
        """
        Return analyzed acoustic frequency if available.
        """

        if self.localization is None:

            return None

        return self.localization.frequency


# ============================================================
# Factory helpers
# ============================================================


def create_acoustic_localization(
    position: Sequence[float],
    response: float,
    *,
    frequency: Optional[float] = None,
    confidence: Optional[float] = None,
    valid: bool = True,
    source_count: int = 1,
) -> AcousticLocalization:
    """
    Create an AcousticLocalization from a position sequence.

    Parameters
    ----------
    position : sequence of float
        Three-element (x, y, z) position in metres.

    response : float
        MUSIC localization response.

    Returns
    -------
    AcousticLocalization
    """

    if len(position) != 3:

        raise ValueError(
            "position must contain exactly three values: "
            "(x, y, z)."
        )

    return AcousticLocalization(
        x=float(position[0]),
        y=float(position[1]),
        z=float(position[2]),
        response=float(response),
        frequency=frequency,
        confidence=confidence,
        valid=valid,
        source_count=source_count,
    )


def create_acoustic_frame(
    frame_id: int,
    timestamp: float,
    localization: Optional[AcousticLocalization] = None,
    *,
    sample_rate: Optional[float] = None,
    frame_size: Optional[int] = None,
    processing_time: Optional[float] = None,
    metadata: Optional[dict] = None,
) -> AcousticFrame:
    """
    Create a timestamped AcousticFrame.

    This helper keeps object construction explicit and makes it
    easier for the MUSIC subsystem to create project-level
    acoustic frames later.
    """

    return AcousticFrame(
        frame_id=frame_id,
        timestamp=float(timestamp),
        localization=localization,
        sample_rate=sample_rate,
        frame_size=frame_size,
        processing_time=processing_time,
        metadata={} if metadata is None else dict(metadata),
    )