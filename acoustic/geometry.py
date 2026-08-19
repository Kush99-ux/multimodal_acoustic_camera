"""
acoustic.geometry
=================

Physical microphone geometry for the miniDSP UMA-16 v2.

The physical microphone labels on the UMA-16 are NOT numbered
sequentially from left-to-right/top-to-bottom.

Verified physical layout from the actual UMA-16 board:

        TOP OF ARRAY

        MIC8   MIC7   MIC10  MIC9
        MIC6   MIC5   MIC12  MIC11
        MIC4   MIC3   MIC14  MIC13
        MIC2   MIC1   MIC16  MIC15

Coordinates are expressed in meters and centered at the
geometric center of the array.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import numpy as np


# ============================================================
# UMA-16 Physical Parameters
# ============================================================

MIC_SPACING = 0.0235

NUM_MICROPHONES = 16

ROWS = 4
COLS = 4


# ============================================================
# Physical microphone numbering
# ============================================================

# IMPORTANT:
#
# This represents the ACTUAL physical labels on the board.
#
# Each row is ordered from LEFT -> RIGHT when looking directly
# at the front of the microphone array.
#
#              LEFT ----------------> RIGHT
#
# Top:         MIC8   MIC7   MIC10   MIC9
# Row 2:       MIC6   MIC5   MIC12   MIC11
# Row 3:       MIC4   MIC3   MIC14   MIC13
# Bottom:      MIC2   MIC1   MIC16   MIC15

PHYSICAL_MIC_LAYOUT = np.array(
    [
        [8, 7, 10, 9],
        [6, 5, 12, 11],
        [4, 3, 14, 13],
        [2, 1, 16, 15],
    ],
    dtype=np.int32,
)


# ============================================================
# Geometry creation
# ============================================================

def create_uma16_geometry(
    spacing: float = MIC_SPACING,
) -> np.ndarray:
    """
    Generate UMA-16 coordinates in AUDIO CHANNEL ORDER.

    IMPORTANT:

    Row/column positions are determined from the physical
    microphone labels on the actual UMA-16 board.

    The returned array is indexed by microphone/channel number:

        geometry[0]  -> MIC1 / CH1
        geometry[1]  -> MIC2 / CH2
        ...
        geometry[15] -> MIC16 / CH16

    Coordinate convention:

        +X = right
        +Y = upward
        +Z = out of the microphone plane

    The array is centered at (0, 0, 0).

    Returns
    -------
    numpy.ndarray
        Shape (16, 3), where each row is:

            [x, y, z]

        for MIC1 ... MIC16.
    """

    # Physical coordinate positions.
    #
    # Looking directly at the front of the array:
    #
    # x:
    #   left  -> negative
    #   right -> positive
    #
    # y:
    #   bottom -> negative
    #   top    -> positive

    x_positions = np.array(
        [-1.5, -0.5, 0.5, 1.5],
        dtype=np.float64,
    ) * spacing

    y_positions = np.array(
        [1.5, 0.5, -0.5, -1.5],
        dtype=np.float64,
    ) * spacing

    # Allocate geometry in MIC/CHANNEL order.
    geometry = np.zeros(
        (NUM_MICROPHONES, 3),
        dtype=np.float64,
    )

    # Walk through the physical board layout and assign the
    # corresponding coordinate to the correct microphone number.
    for row in range(ROWS):
        for col in range(COLS):

            mic_number = int(
                PHYSICAL_MIC_LAYOUT[row, col]
            )

            mic_index = mic_number - 1

            geometry[mic_index] = [
                x_positions[col],
                y_positions[row],
                0.0,
            ]

    return geometry


# ============================================================
# Default geometry
# ============================================================

UMA16_GEOMETRY = create_uma16_geometry()


# ============================================================
# Accessors
# ============================================================

def get_geometry() -> np.ndarray:
    """
    Return a copy of the corrected UMA-16 geometry.

    The returned geometry is indexed by:

        geometry[0]  -> MIC1 / CH1
        geometry[1]  -> MIC2 / CH2
        ...
        geometry[15] -> MIC16 / CH16
    """

    return UMA16_GEOMETRY.copy()


def get_extent() -> tuple[float, float]:
    """
    Return the physical width and height of the microphone grid.

    Returns
    -------
    tuple[float, float]
        (width, height) in meters.
    """

    width = (COLS - 1) * MIC_SPACING
    height = (ROWS - 1) * MIC_SPACING

    return width, height


# ============================================================
# Debug / verification
# ============================================================

def print_geometry() -> None:
    """
    Print the corrected physical geometry in a human-readable
    format.

    This is useful for verifying that the software geometry
    matches the actual board.
    """

    geometry = get_geometry()

    print()
    print("=" * 70)
    print("UMA-16 CORRECTED PHYSICAL GEOMETRY")
    print("=" * 70)

    print()
    print("Physical board layout:")
    print()

    for row in PHYSICAL_MIC_LAYOUT:
        print(
            "   ".join(
                f"MIC{mic:02d}"
                for mic in row
            )
        )

    print()
    print("Channel / microphone coordinates:")
    print("-" * 70)

    for index, position in enumerate(geometry):
        mic_number = index + 1

        print(
            f"MIC{mic_number:02d} / CH{mic_number:02d} : "
            f"X = {position[0]:+.5f} m   "
            f"Y = {position[1]:+.5f} m   "
            f"Z = {position[2]:+.5f} m"
        )

    print("-" * 70)

    width, height = get_extent()

    print(
        f"Array size: "
        f"{width:.4f} m × {height:.4f} m"
    )

    print("=" * 70)
    print()


# ============================================================
# Standalone verification
# ============================================================

if __name__ == "__main__":
    print_geometry()