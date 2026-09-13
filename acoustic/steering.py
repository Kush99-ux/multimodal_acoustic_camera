"""
acoustic.steering
=================

Steering-vector factory for the miniDSP UMA-16 v2 microphone array.

Stage B
-------

This module restores the microphone geometry and coordinate convention
used by the original acoustic-camera implementation.

IMPORTANT
---------

The original implementation did NOT use a generic 23.5 mm microphone
spacing model.

It used the explicit 16 microphone coordinates and then remapped the
raw coordinates into Acoular's (x, y, z) convention:

    raw column 0 -> Acoular X
    raw column 2 -> Acoular Y
    raw column 1 -> Acoular Z

Therefore the physical microphone plane is:

    X-Z

and the array broadside / forward direction is:

    +Y

The microphone ordering is intentionally preserved exactly as supplied
by the original implementation.

This module does NOT assume that the physical channel ordering has
been experimentally verified. It currently uses:

    audio channel 0 -> microphone geometry 0
    audio channel 1 -> microphone geometry 1
    ...
    audio channel 15 -> microphone geometry 15

That identity mapping must later be experimentally verified.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

from typing import Dict

import acoular
import numpy as np

from config.settings import (
    FREQ_MIN,
    FREQ_MAX,
    FREQ_STEP,
    FREQ_CENTERS,
    GRID_X_MIN,
    GRID_X_MAX,
    GRID_Y_MIN,
    GRID_Y_MAX,
    GRID_INCREMENT,
    SPEED_OF_SOUND,
)


# ================================================================
# Original UMA-16 microphone geometry
# ================================================================

# These coordinates are taken directly from the original
# acoustic-camera implementation.
#
# Raw convention:
#
#     column 0 -> x
#     column 1 -> z
#     column 2 -> y
#
# The original implementation subsequently remapped these into
# Acoular's (x, y, z) convention.

ORIGINAL_MIC_RAW = np.array(
    [
        [ 0.021, -0.063, 0.0],
        [ 0.063, -0.063, 0.0],

        [ 0.021, -0.021, 0.0],
        [ 0.063, -0.021, 0.0],

        [ 0.021,  0.021, 0.0],
        [ 0.063,  0.021, 0.0],

        [ 0.021,  0.063, 0.0],
        [ 0.063,  0.063, 0.0],

        [-0.063,  0.063, 0.0],
        [-0.021,  0.063, 0.0],

        [-0.063,  0.021, 0.0],
        [-0.021,  0.021, 0.0],

        [-0.063, -0.021, 0.0],
        [-0.021, -0.021, 0.0],

        [-0.063, -0.063, 0.0],
        [-0.021, -0.063, 0.0],
    ],
    dtype=np.float64,
)


# ================================================================
# Original axis remapping
# ================================================================

def build_microphone_positions() -> np.ndarray:
    """
    Build the exact Acoular microphone-position matrix.

    Returns
    -------
    numpy.ndarray
        Array of shape (3, 16).

        Row 0 -> Acoular X
        Row 1 -> Acoular Y
        Row 2 -> Acoular Z

    Notes
    -----
    This reproduces the original implementation exactly:

        mic_pos[0, :] = mic_raw[0, :]
        mic_pos[1, :] = mic_raw[2, :]
        mic_pos[2, :] = mic_raw[1, :]
    """

    mic_raw = ORIGINAL_MIC_RAW.T.copy()

    mic_pos = np.zeros_like(mic_raw)

    # Original implementation.
    mic_pos[0, :] = mic_raw[0, :]
    mic_pos[1, :] = mic_raw[2, :]
    mic_pos[2, :] = mic_raw[1, :]

    return mic_pos


# ================================================================
# Steering Factory
# ================================================================

class SteeringFactory:
    """
    Factory for precomputed Acoular steering matrices.

    Coordinate convention
    ---------------------

    The restored original convention is:

        X -> horizontal array axis
        Y -> forward / broadside direction
        Z -> vertical array axis

    Therefore:

        microphone plane = X-Z
        forward direction = +Y

    The search grid is constructed in angular coordinates around
    the +Y broadside direction.
    """

    def __init__(self) -> None:

        # --------------------------------------------------------
        # Frequencies
        # --------------------------------------------------------

        if FREQ_STEP <= 0:
            raise ValueError(
                "FREQ_STEP must be positive."
            )

        if FREQ_MAX < FREQ_MIN:
            raise ValueError(
                "FREQ_MAX must be >= FREQ_MIN."
            )

        self.frequencies = np.array(
            FREQ_CENTERS,
            dtype=np.float64,
        )

        # --------------------------------------------------------
        # Microphone geometry
        # --------------------------------------------------------

        self.mic_positions = build_microphone_positions()

        if self.mic_positions.shape != (3, 16):
            raise RuntimeError(
                "UMA-16 geometry must have shape (3, 16)."
            )

        self.microphone_geometry = acoular.MicGeom()

        self.microphone_geometry.pos_total = (
            self.mic_positions
        )

        # --------------------------------------------------------
        # Acoustic environment
        # --------------------------------------------------------

        self.environment = acoular.Environment(
            c=float(SPEED_OF_SOUND)
        )

        # --------------------------------------------------------
        # Search grid
        # --------------------------------------------------------
        #
        # IMPORTANT:
        #
        # The original implementation searched angular space rather
        # than the project's old generic X/Y/Z plane.
        #
        # GRID_X_* and GRID_Y_* are therefore interpreted here as
        # angular/search parameters only if the existing configuration
        # is still used.
        #
        # To preserve the original implementation faithfully, we use
        # the original camera FOV values below.

        camera_fov_h = 50.0
        camera_fov_v = camera_fov_h / (4.0 / 3.0)

        az_center = 90.0
        el_center = 0.0

        az_min = (
            az_center
            - camera_fov_h / 2.0
        )

        az_max = (
            az_center
            + camera_fov_h / 2.0
        )

        el_min = (
            el_center
            - camera_fov_v / 2.0
        )

        el_max = (
            el_center
            + camera_fov_v / 2.0
        )

        step_deg = 0.5

        self.az_values = np.arange(
            az_min,
            az_max + step_deg,
            step_deg,
            dtype=np.float64,
        )

        self.el_values = np.arange(
            el_min,
            el_max + step_deg,
            step_deg,
            dtype=np.float64,
        )

        self.az_grid, self.el_grid = np.meshgrid(
            self.az_values,
            self.el_values,
        )

        # --------------------------------------------------------
        # Angular grid -> Cartesian unit vectors
        # --------------------------------------------------------

        az_rad = np.deg2rad(
            self.az_grid.flatten()
        )

        el_rad = np.deg2rad(
            self.el_grid.flatten()
        )

        x_sphere = (
            np.cos(el_rad)
            * np.cos(az_rad)
        )

        y_sphere = (
            np.cos(el_rad)
            * np.sin(az_rad)
        )

        z_sphere = np.sin(el_rad)

        grid_positions = np.array(
            [
                x_sphere,
                y_sphere,
                z_sphere,
            ],
            dtype=np.float64,
        )

        self.grid = acoular.ImportGrid(
            pos=grid_positions
        )

        # --------------------------------------------------------
        # SteeringVector
        # --------------------------------------------------------

        self.steering_vector = acoular.SteeringVector(
            grid=self.grid,
            mics=self.microphone_geometry,
            env=self.environment,
            steer_type="true level",
        )

        # --------------------------------------------------------
        # Matrix cache
        # --------------------------------------------------------

        self.steering_matrices: Dict[
            float,
            np.ndarray,
        ] = {}

    # ============================================================
    # Precomputation
    # ============================================================

    def precompute_matrices(self) -> None:
        """
        Precompute steering matrices for all configured frequencies.

        Acoular returns:

            (grid_points, microphones)

        The MUSIC implementation expects:

            (microphones, grid_points)

        therefore the matrix is transposed here, exactly as in the
        original implementation.
        """

        self.steering_matrices.clear()

        print(
            "[Steering] Building ORIGINAL UMA-16 geometry..."
        )

        print(
            f"[Steering] Microphones: "
            f"{self.microphone_geometry.num_mics}"
        )

        print(
            "[Steering] Coordinate convention: "
            "X-Z array plane, +Y broadside"
        )

        print(
            f"[Steering] Speed of sound: "
            f"{self.environment.c:.2f} m/s"
        )

        print(
            "[Steering] Precomputing steering matrices..."
        )

        for frequency in self.frequencies:

            frequency = float(frequency)

            transfer = (
                self.steering_vector.transfer(
                    frequency
                )
            )

            # Acoular:
            #
            #     grid_points × microphones
            #
            # MUSIC:
            #
            #     microphones × grid_points

            matrix = np.asarray(
                transfer.T,
                dtype=np.complex128,
            )

            self.steering_matrices[
                frequency
            ] = matrix

        print(
            "[Steering] Cached "
            f"{len(self.steering_matrices)} "
            "frequency matrices"
        )

    # ============================================================
    # Accessors
    # ============================================================

    def get_matrix(
        self,
        frequency: float,
    ) -> np.ndarray:
        """
        Return the precomputed steering matrix.
        """

        frequency = float(frequency)

        if frequency not in self.steering_matrices:
            raise KeyError(
                f"No steering matrix cached for "
                f"{frequency} Hz."
            )

        return self.steering_matrices[
            frequency
        ]

    def get_band_steering(
        self,
        low_hz: float,
        high_hz: float,
    ) -> np.ndarray:
        """
        Return the average of all precomputed steering matrices
        within the specified frequency band [low_hz, high_hz).
        """
        midpoint = (low_hz + high_hz) / 2.0

        nearest_freq = min(
            self.frequencies,
            key=lambda f: abs(float(f) - midpoint),
        )

        return self.get_matrix(float(nearest_freq))

    def get_frequencies(self) -> np.ndarray:
        """
        Return configured MUSIC frequencies.
        """

        return self.frequencies.copy()

    def get_microphone_geometry(
        self,
    ) -> acoular.MicGeom:
        """
        Return the Acoular microphone geometry.
        """

        return self.microphone_geometry

    def get_environment(
        self,
    ) -> acoular.Environment:
        """
        Return the acoustic propagation environment.
        """

        return self.environment

    def grid_shape(self) -> tuple[int, int]:
        """
        Return the 2D angular grid shape.
        """

        return self.az_grid.shape

    def get_grid_positions(self) -> np.ndarray:
        """
        Return Cartesian positions of all search directions.

        Shape:

            (3, N)

        Coordinates:

            X, Y, Z
        """

        return self.grid.pos.copy()

    def get_azimuth_values(self) -> np.ndarray:
        """
        Return the 1D array of azimuth values (degrees) corresponding to grid columns.
        """
        return self.az_values.copy()

    def get_elevation_values(self) -> np.ndarray:
        """
        Return the 1D array of elevation values (degrees) corresponding to grid rows.
        """
        return self.el_values.copy()

    def get_azimuth_grid(self) -> np.ndarray:
        """
        Return the 2D azimuth grid in degrees.
        """

        return self.az_grid.copy()

    def get_elevation_grid(self) -> np.ndarray:
        """
        Return the 2D elevation grid in degrees.
        """

        return self.el_grid.copy()

    # ============================================================
    # Geometry validation
    # ============================================================

    def validate_geometry(self) -> dict:
        """
        Perform internal consistency checks on the restored geometry.

        This validates the software representation.

        It does NOT prove that the physical UMA-16 channel ordering
        matches the assumed ordering.
        """

        positions = self.mic_positions

        checks = {
            "channel_count": (
                positions.shape[1] == 16
            ),

            "array_is_xz_plane": (
                np.allclose(
                    positions[1, :],
                    0.0,
                )
            ),

            "x_range": (
                np.isclose(
                    positions[0, :].min(),
                    -0.063,
                )
                and
                np.isclose(
                    positions[0, :].max(),
                    0.063,
                )
            ),

            "z_range": (
                np.isclose(
                    positions[2, :].min(),
                    -0.063,
                )
                and
                np.isclose(
                    positions[2, :].max(),
                    0.063,
                )
            ),

            "broadside_is_positive_y": (
                np.allclose(
                    self.grid.pos[
                        0,
                        np.argmin(
                            np.abs(
                                self.az_grid.flatten()
                                - 90.0
                            )
                        )
                    ],
                    0.0,
                )
            ),
        }

        checks["all_passed"] = all(
            checks.values()
        )

        return checks