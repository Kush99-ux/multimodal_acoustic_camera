"""
acoustic.music
==============

MUSIC (Multiple Signal Classification) localization engine for the
miniDSP UMA-16 v2 microphone array.

This module provides:

1. Real-time MUSIC localization.
2. FFT-based frequency analysis.
3. Cross-Spectral Matrix (CSM) estimation.
4. Noise-subspace MUSIC spectrum calculation.
5. Normalized acoustic localization heatmaps.
6. Numerical extraction of the strongest estimated acoustic source.

The localization result is expressed in the same physical coordinate
system used by the SteeringFactory.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import numpy as np

from acoustic.steering import SteeringFactory
from config.settings import (
    SAMPLE_RATE,
    FRAME_SIZE,
    MUSIC_SMOOTHING,
    NUM_SOURCES,
    GRID_Z,
    GRID_X_MIN,
    GRID_X_MAX,
    GRID_Y_MIN,
    GRID_Y_MAX,
    GRID_INCREMENT,
)


class MusicLocalizer:
    """
    Real-time MUSIC localization engine.

    The localizer receives a multichannel audio frame from the UMA-16,
    computes the frequency-domain representation, estimates the
    cross-spectral matrices, evaluates the MUSIC pseudospectrum over
    the configured spatial grid, and returns a normalized localization
    heatmap.

    The strongest point in the heatmap can then be converted into a
    physical (x, y, z) coordinate using :meth:`get_peak_position`.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
        smoothing: float = MUSIC_SMOOTHING,
        n_sources: int = NUM_SOURCES,
    ) -> None:

        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.smoothing = smoothing
        self.n_sources = n_sources

        # -----------------------------------------------------
        # Steering / geometry subsystem
        # -----------------------------------------------------

        self.factory = SteeringFactory()

        # Precompute all configured steering matrices once during
        # initialization rather than repeatedly during localization.
        self.factory.precompute_matrices()

        self.frequencies = self.factory.get_frequencies()
        self.grid_shape = self.factory.grid_shape()

        # -----------------------------------------------------
        # Temporal CSM state
        # -----------------------------------------------------

        self.previous_csm: np.ndarray | None = None

        # FFT frequency bins corresponding to the configured
        # analysis frame.
        self.fft_frequencies = np.fft.rfftfreq(
            self.frame_size,
            1.0 / self.sample_rate,
        )

    # =========================================================
    # Frequency utilities
    # =========================================================

    def _nearest_fft_bin(self, frequency: float) -> int:
        """
        Find the FFT bin closest to a requested frequency.

        Parameters
        ----------
        frequency : float
            Requested frequency in Hz.

        Returns
        -------
        int
            Index of the closest FFT bin.
        """

        return int(
            np.argmin(
                np.abs(self.fft_frequencies - frequency)
            )
        )

    # =========================================================
    # Cross-Spectral Matrix
    # =========================================================

    def _compute_csm(
        self,
        fft_frame: np.ndarray,
        bin_index: int,
    ) -> np.ndarray:
        """
        Compute and temporally smooth the Cross-Spectral Matrix.

        Parameters
        ----------
        fft_frame : numpy.ndarray
            Frequency-domain multichannel audio frame.

        bin_index : int
            FFT bin to use.

        Returns
        -------
        numpy.ndarray
            Complex CSM with shape:

                (NUM_CHANNELS, NUM_CHANNELS)
        """

        # Extract the complex spectrum of all microphones at the
        # selected frequency bin.
        x = fft_frame[bin_index]

        # Spatial covariance / Cross-Spectral Matrix.
        csm = np.outer(
            x,
            np.conjugate(x),
        )

        # First frame initializes the temporal state.
        if self.previous_csm is None:
            self.previous_csm = csm

        # Subsequent frames use exponential smoothing.
        else:
            self.previous_csm = (
                self.smoothing * csm
                + (1.0 - self.smoothing) * self.previous_csm
            )

        return self.previous_csm

    # =========================================================
    # MUSIC spectrum
    # =========================================================

    def _music_spectrum(
        self,
        csm: np.ndarray,
        steering_matrix: np.ndarray,
        n_sources: int | None = None,
    ) -> np.ndarray:
        """
        Calculate the MUSIC pseudospectrum.

        Parameters
        ----------
        csm : numpy.ndarray
            Cross-Spectral Matrix.

        steering_matrix : numpy.ndarray
            Steering matrix with shape:

                (grid_points, microphones)

        n_sources : int, optional
            Number of dominant acoustic sources.

            If omitted, the configured NUM_SOURCES value is used.

        Returns
        -------
        numpy.ndarray
            MUSIC spectrum for every spatial grid point.
        """

        if n_sources is None:
            n_sources = self.n_sources

        # -----------------------------------------------------
        # Eigenvalue decomposition
        # -----------------------------------------------------

        eigenvalues, eigenvectors = np.linalg.eigh(csm)

        # Sort eigenvectors according to descending eigenvalue.
        order = np.argsort(eigenvalues)[::-1]

        eigenvectors = eigenvectors[:, order]

        # -----------------------------------------------------
        # Noise subspace
        # -----------------------------------------------------

        # The strongest n_sources eigenvectors represent the signal
        # subspace. The remaining eigenvectors form the noise
        # subspace.
        noise_subspace = eigenvectors[:, n_sources:]

        # Projection matrix onto the noise subspace.
        projection = (
            noise_subspace
            @ noise_subspace.conj().T
        )

        # -----------------------------------------------------
        # MUSIC denominator
        # -----------------------------------------------------

        denominator = np.real(
            np.einsum(
                "ij,jk,ik->i",
                steering_matrix.conj(),
                projection,
                steering_matrix,
            )
        )

        # Prevent numerical division by zero.
        denominator = np.maximum(
            denominator,
            1e-12,
        )

        # MUSIC pseudospectrum.
        return 1.0 / denominator

    # =========================================================
    # Localization
    # =========================================================

    def localize(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a normalized MUSIC localization heatmap.

        Parameters
        ----------
        frame : numpy.ndarray
            Audio frame with shape:

                (FRAME_SIZE, NUM_CHANNELS)

        Returns
        -------
        numpy.ndarray
            Normalized MUSIC heatmap with shape:

                (grid_y, grid_x)

            Values are normalized to the range:

                0.0 → 1.0
        """

        # -----------------------------------------------------
        # Validate input
        # -----------------------------------------------------

        if frame.ndim != 2:
            raise ValueError(
                "Audio frame must be 2-dimensional."
            )

        if frame.shape[0] != self.frame_size:
            raise ValueError(
                f"Expected {self.frame_size} samples per frame, "
                f"received {frame.shape[0]}."
            )

        expected_channels = (
            self.factory.get_microphone_geometry().pos_total.shape[1]
        )

        if frame.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} audio channels, "
                f"received {frame.shape[1]}."
            )

        # -----------------------------------------------------
        # FFT
        # -----------------------------------------------------

        fft_frame = np.fft.rfft(
            frame,
            axis=0,
        )

        # -----------------------------------------------------
        # MUSIC spectrum accumulator
        # -----------------------------------------------------

        number_of_grid_points = (
            self.grid_shape[0]
            * self.grid_shape[1]
        )

        spectrum = np.zeros(
            number_of_grid_points,
            dtype=np.float64,
        )

        count = 0

        # -----------------------------------------------------
        # Evaluate configured frequencies
        # -----------------------------------------------------

        for frequency in self.frequencies:

            frequency = float(frequency)

            # Find closest FFT bin.
            bin_index = self._nearest_fft_bin(
                frequency
            )

            # Estimate CSM.
            csm = self._compute_csm(
                fft_frame,
                bin_index,
            )

            # Retrieve precomputed steering matrix.
            steering_matrix = self.factory.get_matrix(
                frequency
            )

            # Calculate MUSIC pseudospectrum.
            spectrum += self._music_spectrum(
                csm,
                steering_matrix,
            )

            count += 1

        # -----------------------------------------------------
        # Frequency averaging
        # -----------------------------------------------------

        if count > 0:
            spectrum /= count

        # -----------------------------------------------------
        # Convert to 2D grid
        # -----------------------------------------------------

        heatmap = spectrum.reshape(
            self.grid_shape
        )

        # -----------------------------------------------------
        # Normalize
        # -----------------------------------------------------

        heatmap -= heatmap.min()

        maximum = heatmap.max()

        if maximum > 0:
            heatmap /= maximum

        return heatmap

    # =========================================================
    # Peak extraction
    # =========================================================

    def get_peak_position(
        self,
        heatmap: np.ndarray,
    ) -> dict[str, float | int]:
        """
        Extract the strongest MUSIC localization point.

        This method converts the maximum value in the MUSIC heatmap
        into physical x, y, and z coordinates using the same grid
        configuration used by SteeringFactory.

        Parameters
        ----------
        heatmap : numpy.ndarray
            MUSIC heatmap returned by :meth:`localize`.

        Returns
        -------
        dict
            Dictionary containing:

            x
                Estimated source x-coordinate in metres.

            y
                Estimated source y-coordinate in metres.

            z
                Localization-plane z-coordinate in metres.

            value
                Normalized MUSIC response at the peak.

            row
                Heatmap row containing the peak.

            column
                Heatmap column containing the peak.
        """

        # -----------------------------------------------------
        # Validate heatmap
        # -----------------------------------------------------

        if heatmap.ndim != 2:
            raise ValueError(
                "Heatmap must be a 2-dimensional array."
            )

        if heatmap.shape != self.grid_shape:
            raise ValueError(
                f"Expected heatmap shape {self.grid_shape}, "
                f"received {heatmap.shape}."
            )

        # -----------------------------------------------------
        # Find maximum response
        # -----------------------------------------------------

        peak_index = int(
            np.argmax(heatmap)
        )

        peak_row, peak_column = np.unravel_index(
            peak_index,
            heatmap.shape,
        )

        peak_value = float(
            heatmap[peak_row, peak_column]
        )

        # -----------------------------------------------------
        # Convert grid index → physical coordinates
        # -----------------------------------------------------

        x = (
            GRID_X_MIN
            + peak_column * GRID_INCREMENT
        )

        y = (
            GRID_Y_MAX
            - peak_row * GRID_INCREMENT
        )

        z = GRID_Z

        return {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "value": peak_value,
            "row": int(peak_row),
            "column": int(peak_column),
        }

    # =========================================================
    # Convenience method
    # =========================================================

    def localize_with_peak(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        """
        Run MUSIC localization and immediately extract the strongest
        estimated source position.

        Parameters
        ----------
        frame : numpy.ndarray
            Audio frame with shape:

                (FRAME_SIZE, NUM_CHANNELS)

        Returns
        -------
        tuple
            (
                heatmap,
                peak_position
            )

        Example
        -------
        heatmap, peak = localizer.localize_with_peak(frame)

        print(
            peak["x"],
            peak["y"],
            peak["z"],
        )
        """

        heatmap = self.localize(frame)

        peak = self.get_peak_position(
            heatmap
        )

        return heatmap, peak