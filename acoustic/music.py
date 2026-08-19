"""
acoustic.music
==============

MUSIC (Multiple Signal Classification) localization engine for the
miniDSP UMA-16 v2 microphone array.

This implementation is extracted and modularized from the validated
`uma_live_camera.py` prototype.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import numpy as np

from acoustic.steering import SteeringFactory
from config.settings import SAMPLE_RATE, FRAME_SIZE


class MusicLocalizer:
    """
    Real-time MUSIC localization engine.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
        smoothing: float = 0.3,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.smoothing = smoothing

        self.factory = SteeringFactory()
        self.factory.precompute_matrices()

        self.frequencies = self.factory.get_frequencies()
        self.grid_shape = self.factory.grid_shape()

        self.previous_csm = None

        self.fft_frequencies = np.fft.rfftfreq(
            self.frame_size,
            1.0 / self.sample_rate,
        )

    def _nearest_fft_bin(self, frequency: float) -> int:
        return int(np.argmin(np.abs(self.fft_frequencies - frequency)))

    def _compute_csm(self, fft_frame: np.ndarray, bin_index: int) -> np.ndarray:
        x = fft_frame[bin_index]
        csm = np.outer(x, np.conjugate(x))

        if self.previous_csm is None:
            self.previous_csm = csm
        else:
            self.previous_csm = (
                self.smoothing * csm
                + (1.0 - self.smoothing) * self.previous_csm
            )

        return self.previous_csm

    def _music_spectrum(
        self,
        csm: np.ndarray,
        steering_matrix: np.ndarray,
        n_sources: int = 1,
    ) -> np.ndarray:

        eigenvalues, eigenvectors = np.linalg.eigh(csm)

        order = np.argsort(eigenvalues)[::-1]

        eigenvectors = eigenvectors[:, order]

        noise_subspace = eigenvectors[:, n_sources:]

        projection = noise_subspace @ noise_subspace.conj().T

        denominator = np.real(
            np.einsum(
                "ij,jk,ik->i",
                steering_matrix.conj(),
                projection,
                steering_matrix,
            )
        )

        denominator = np.maximum(denominator, 1e-12)

        return 1.0 / denominator

    def localize(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute a normalized MUSIC localization heatmap.

        Parameters
        ----------
        frame : numpy.ndarray
            Audio frame with shape (FRAME_SIZE, NUM_CHANNELS)

        Returns
        -------
        numpy.ndarray
            Normalized heatmap with shape (grid_y, grid_x)
        """

        if frame.ndim != 2:
            raise ValueError("Frame must be 2-dimensional")

        fft_frame = np.fft.rfft(frame, axis=0)

        spectrum = np.zeros(
            self.grid_shape[0] * self.grid_shape[1],
            dtype=np.float64,
        )

        count = 0

        for f in self.frequencies:
            bin_index = self._nearest_fft_bin(f)

            csm = self._compute_csm(fft_frame, bin_index)

            A = self.factory.get_matrix(float(f))

            spectrum += self._music_spectrum(csm, A)

            count += 1

        if count > 0:
            spectrum /= count

        heatmap = spectrum.reshape(self.grid_shape)

        heatmap -= heatmap.min()

        maximum = heatmap.max()

        if maximum > 0:
            heatmap /= maximum

        return heatmap