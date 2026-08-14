"""
acoustic.steering
=================

Steering-vector generation and precomputation for the miniDSP UMA-16 v2 array.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import numpy as np
from acoular import MicGeom, RectGrid, SteeringVector

from acoustic.geometry import get_geometry
from config.settings import (
    SPEED_OF_SOUND,
    GRID_Z,
    GRID_X_MIN,
    GRID_X_MAX,
    GRID_Y_MIN,
    GRID_Y_MAX,
    GRID_INCREMENT,
    FREQ_MIN,
    FREQ_MAX,
    FREQ_STEP,
)


class SteeringFactory:
    """
    Factory for creating and caching steering vectors for MUSIC localization.
    """

    def __init__(self, c: float = SPEED_OF_SOUND) -> None:
        self.c = c

        # -----------------------------------------------------
        # Microphone geometry
        # -----------------------------------------------------
        geometry = get_geometry()
        self.geometry = geometry

        # Acoular expects coordinates as (3, N)
        self.mic_geom = MicGeom(pos_total=geometry.T)

        # -----------------------------------------------------
        # Localization grid
        # -----------------------------------------------------
        self.grid = RectGrid(
            x_min=GRID_X_MIN,
            x_max=GRID_X_MAX,
            y_min=GRID_Y_MIN,
            y_max=GRID_Y_MAX,
            z=GRID_Z,
            increment=GRID_INCREMENT,
        )

        # -----------------------------------------------------
        # Steering vector object
        # -----------------------------------------------------
        self.steering = SteeringVector(
            grid=self.grid,
            mics=self.mic_geom,
            env=None,
        )

        # Frequency bins
        self.frequencies = np.arange(
            FREQ_MIN,
            FREQ_MAX + FREQ_STEP,
            FREQ_STEP,
            dtype=np.float64,
        )

        # Cache for precomputed steering matrices
        self._steering_cache: dict[float, np.ndarray] = {}

    # ---------------------------------------------------------
    # Basic accessors
    # ---------------------------------------------------------

    def get_grid(self):
        return self.grid

    def get_microphone_geometry(self):
        return self.mic_geom

    def get_frequencies(self) -> np.ndarray:
        return self.frequencies.copy()

    def get_steering_vector(self):
        return self.steering

    def grid_shape(self) -> tuple[int, int]:
        nx = int(round((GRID_X_MAX - GRID_X_MIN) / GRID_INCREMENT)) + 1
        ny = int(round((GRID_Y_MAX - GRID_Y_MIN) / GRID_INCREMENT)) + 1
        return ny, nx

    # ---------------------------------------------------------
    # Steering matrix generation
    # ---------------------------------------------------------

    def steering_matrix(self, frequency: float) -> np.ndarray:
        """
        Compute the steering matrix for a single frequency.

        Returns
        -------
        numpy.ndarray
            Complex steering matrix with shape (n_points, n_mics).
        """

        if frequency in self._steering_cache:
            return self._steering_cache[frequency]

        # Grid coordinates (3, n_points)
        points = self.grid.pos

        # Microphone coordinates (3, n_mics)
        mics = self.geometry.T

        # Distance from every grid point to every microphone
        diff = points[:, :, None] - mics[:, None, :]
        distances = np.linalg.norm(diff, axis=0)

        # Time delays
        delays = distances / self.c

        # Steering vectors
        A = np.exp(-2j * np.pi * frequency * delays)

        # Normalize
        A /= np.sqrt(A.shape[1])

        self._steering_cache[frequency] = A

        return A

    def precompute_matrices(self) -> None:
        """
        Precompute steering matrices for all configured frequencies.
        """

        print("[Steering] Precomputing steering matrices...")

        for f in self.frequencies:
            self.steering_matrix(float(f))

        print(
            f"[Steering] Cached {len(self._steering_cache)} frequency matrices"
        )

    def get_matrix(self, frequency: float) -> np.ndarray:
        """
        Retrieve a steering matrix from the cache.
        """

        return self.steering_matrix(frequency)

    def cache_size(self) -> int:
        """
        Number of cached steering matrices.
        """

        return len(self._steering_cache)