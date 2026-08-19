"""
acoustic.geometry
=================

Microphone array geometry definitions for the Multimodal Acoustic Camera project.

This module defines the physical coordinates of the miniDSP UMA-16 v2
microphone array as a 4x4 Uniform Rectangular Array (URA). All coordinates
are expressed in meters and centered at the origin of the array.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import numpy as np

# ============================================================
# UMA-16 v2 Physical Geometry
# ============================================================

# Microphone spacing (center-to-center) in meters.
# miniDSP documentation specifies approximately 23.5 mm spacing.
MIC_SPACING = 0.0235

# Number of microphones
NUM_MICROPHONES = 16

# Array dimensions
ROWS = 4
COLS = 4


def create_uma16_geometry(spacing: float = MIC_SPACING) -> np.ndarray:
    """
    Generate centered microphone coordinates for the UMA-16 array.

    The array is modeled as a 4x4 Uniform Rectangular Array (URA)
    centered at (0, 0, 0).

    Parameters
    ----------
    spacing : float
        Distance between adjacent microphones in meters.

    Returns
    -------
    numpy.ndarray
        Array of shape (16, 3) containing (x, y, z) coordinates.
    """

    coords = []

    x_positions = np.array([-1.5, -0.5, 0.5, 1.5]) * spacing
    y_positions = np.array([1.5, 0.5, -0.5, -1.5]) * spacing

    for y in y_positions:
        for x in x_positions:
            coords.append([x, y, 0.0])

    return np.asarray(coords, dtype=np.float64)


# Default geometry used throughout the project
UMA16_GEOMETRY = create_uma16_geometry()


def get_geometry() -> np.ndarray:
    """
    Return a copy of the default UMA-16 geometry.

    Returns
    -------
    numpy.ndarray
        Microphone coordinates with shape (16, 3).
    """

    return UMA16_GEOMETRY.copy()


def get_extent() -> tuple[float, float]:
    """
    Return the physical width and height of the array.

    Returns
    -------
    tuple
        (width, height) in meters.
    """

    width = (COLS - 1) * MIC_SPACING
    height = (ROWS - 1) * MIC_SPACING

    return width, height


if __name__ == "__main__":
    geometry = get_geometry()

    print("UMA-16 Geometry (meters)")
    print(geometry)

    width, height = get_extent()
    print(f"Array size: {width:.4f} m × {height:.4f} m")