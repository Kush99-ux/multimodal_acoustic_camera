"""
acoustic.orientation
====================

Microphone-array orientation utilities.

This module applies rotation and mirroring transforms to the UMA-16 geometry
before steering vectors are generated.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import numpy as np


class ArrayOrientation:
    """Apply rotation and mirror transforms to microphone coordinates."""

    @staticmethod
    def rotate(coords: np.ndarray, angle_deg: int) -> np.ndarray:
        """
        Rotate microphone coordinates around the array center.

        Parameters
        ----------
        coords : ndarray of shape (16, 3)
            Original microphone coordinates.

        angle_deg : int
            Rotation angle: 0, 90, 180, or 270 degrees.

        Returns
        -------
        ndarray
            Rotated coordinates.
        """

        angle = np.deg2rad(angle_deg)

        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle),  np.cos(angle), 0.0],
                [0.0,            0.0,           1.0],
            ]
        )

        return coords @ R.T

    @staticmethod
    def flip_horizontal(coords: np.ndarray) -> np.ndarray:
        """Mirror across the Y-axis."""

        out = coords.copy()
        out[:, 0] *= -1
        return out

    @staticmethod
    def flip_vertical(coords: np.ndarray) -> np.ndarray:
        """Mirror across the X-axis."""

        out = coords.copy()
        out[:, 1] *= -1
        return out

    @staticmethod
    def transform(
        coords: np.ndarray,
        rotation: int = 0,
        flip_h: bool = False,
        flip_v: bool = False,
    ) -> np.ndarray:
        """
        Apply rotation followed by optional mirror operations.
        """

        out = ArrayOrientation.rotate(coords, rotation)

        if flip_h:
            out = ArrayOrientation.flip_horizontal(out)

        if flip_v:
            out = ArrayOrientation.flip_vertical(out)

        return out