"""
acoustic.music
==============
Real-time MUSIC localization for the miniDSP UMA-16.

This module restores exact behavioral equivalence with the golden reference
implementation (reference_original/uma_live_camera.py).

Signal Processing Pipeline:
  1. STFT: Hann window and rfft
  2. Sub-band selection: 6 center frequencies (500, 1000, 1500, 2000, 2500, 3000 Hz)
     with +-250 Hz bandwidth around each center
  3. CSM: Cross-spectral matrix computed per band and smoothed via EMA (alpha = 0.30)
  4. Regularization and Eigendecomposition: eigh on smoothed CSM
  5. Noise subspace: Minor eigenvectors spanning noise subspace
  6. MUSIC pseudo-spectrum: Steering projection onto noise subspace
  7. Per-band min-max normalization: Ensures all bands contribute equally
  8. Broadband accumulation: Sum of normalized sub-band spectra
  9. Contrast enhancement: Sigmoid shaping (gain = 3.0, center = 0.5)
 10. Peak extraction: Azimuth and elevation corresponding to maximum response

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import time
from typing import Optional
import numpy as np

from acoustic.audio import AudioFrame
from acoustic.models import AcousticFrame, AcousticLocalization
from acoustic.steering import SteeringFactory

from config.settings import (
    SAMPLE_RATE,
    FRAME_SIZE,
    MUSIC_SMOOTHING,
    CSM_ALPHA,
    NUM_SOURCES,
    NOISE_THRESHOLD_DB,
    SIGMOID_GAIN,
    SIGMOID_MIDPOINT,
    FREQ_CENTERS,
    BAND_WIDTH,
    BAND_WEIGHTING_MODE,
)


class MusicLocalizer:
    """Frequency-domain MUSIC localizer for the miniDSP UMA-16 v2."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
        smoothing: float = CSM_ALPHA,
        n_sources: int = NUM_SOURCES,
        noise_threshold_db: float = NOISE_THRESHOLD_DB,
        sigmoid_gain: float = SIGMOID_GAIN,
        sigmoid_midpoint: float = SIGMOID_MIDPOINT,
        band_weighting_mode: str = BAND_WEIGHTING_MODE,
    ) -> None:

        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)
        self.smoothing = float(smoothing)
        self.n_sources = int(n_sources)
        self.noise_threshold_db = float(noise_threshold_db)
        self.sigmoid_gain = float(sigmoid_gain)
        self.sigmoid_midpoint = float(sigmoid_midpoint)
        self.band_weighting_mode = str(band_weighting_mode).lower().strip()

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if not 0.0 <= self.smoothing <= 1.0:
            raise ValueError("smoothing must be between 0 and 1.")
        if self.n_sources < 0:
            raise ValueError("n_sources cannot be negative.")
        if self.band_weighting_mode not in ("baseline", "energy"):
            raise ValueError(
                f"Unsupported band_weighting_mode: '{band_weighting_mode}'. "
                f"Supported modes: 'baseline', 'energy'."
            )

        # SteeringFactory owns the validated 42 mm UMA-16 geometry.
        self.steering_factory = SteeringFactory()
        self.steering_factory.precompute_matrices()

        self.grid_shape = tuple(self.steering_factory.grid_shape())
        self.az_values = self.steering_factory.get_azimuth_values()
        self.el_values = self.steering_factory.get_elevation_values()

        geometry = self.steering_factory.get_microphone_geometry()
        self.num_mics = int(geometry.pos_total.shape[1])

        if self.num_mics <= 0:
            raise ValueError("Microphone geometry contains no microphones.")

        # Exact 6 center frequencies from golden reference: 500, 1000, 1500, 2000, 2500, 3000 Hz
        self.freq_centers = np.array(FREQ_CENTERS, dtype=np.float64)
        self.band_width = float(BAND_WIDTH)
        self.ema_alpha = self.smoothing

        # Cross-spectral matrix temporal history per center frequency
        self._smoothed_csms: dict[float, Optional[np.ndarray]] = {
            float(f): None for f in self.freq_centers
        }

        # Precompute FFT bin boundaries for each sub-band (center +- BAND_WIDTH / 2)
        freqs = np.fft.rfftfreq(self.frame_size, 1.0 / self.sample_rate)
        self._band_bins: list[tuple[int, int]] = []
        for center in self.freq_centers:
            f_min = center - (self.band_width / 2.0)
            f_max = center + (self.band_width / 2.0)
            idx_low = int(np.argmin(np.abs(freqs - f_min)))
            idx_high = int(np.argmin(np.abs(freqs - f_max)))
            if idx_low == idx_high:
                idx_high += 1
            self._band_bins.append((idx_low, idx_high))

        print(f"[MUSIC] Initialized with {self.num_mics} microphones")
        print(f"[MUSIC] Frame size: {self.frame_size} samples @ {self.sample_rate} Hz")
        print(f"[MUSIC] Sub-band centers: {list(self.freq_centers)} Hz (width: {self.band_width} Hz)")
        print(f"[MUSIC] Noise threshold: {self.noise_threshold_db} dB, EMA alpha: {self.ema_alpha}")

    # ------------------------------------------------------------------
    # Signal Level (dB)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_db_level(signal: np.ndarray) -> float:
        """Calculate RMS sound pressure level in dB offset to positive range."""
        rms = float(np.sqrt(np.mean(signal ** 2)))
        return float(20.0 * np.log10(rms + 1e-9) + 90.0)

    # ------------------------------------------------------------------
    # Frequency / CSM
    # ------------------------------------------------------------------

    def _compute_band_csm(
        self,
        spectrum: np.ndarray,
        idx_low: int,
        idx_high: int,
    ) -> np.ndarray:
        """
        Compute spatial CSM by averaging FFT-bin outer products inside one band.
        spectrum shape: (channels, freq_bins)
        """
        X_band = spectrum[:, idx_low:idx_high]
        bin_count = X_band.shape[1]
        if bin_count == 0:
            return np.zeros((self.num_mics, self.num_mics), dtype=np.complex128)

        # X_band @ X_band^H / N_bins
        csm = (X_band @ X_band.conj().T) / float(bin_count)
        # Enforce exact Hermiticity
        csm = 0.5 * (csm + csm.conj().T)
        return csm

    def _smooth_csm(
        self,
        center_freq: float,
        csm_instant: np.ndarray,
    ) -> np.ndarray:
        """Apply exponential moving average to the cross-spectral matrix."""
        previous = self._smoothed_csms.get(center_freq)
        if previous is None:
            smoothed = csm_instant.copy()
        else:
            smoothed = (
                (1.0 - self.ema_alpha) * previous
                + self.ema_alpha * csm_instant
            )
        self._smoothed_csms[center_freq] = smoothed
        return smoothed

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------

    def reset_csm_state(self) -> None:
        """Clear all per-frequency temporal CSM history."""
        for f in self.freq_centers:
            self._smoothed_csms[float(f)] = None

    # ------------------------------------------------------------------
    # MUSIC Core
    # ------------------------------------------------------------------

    def _music_band_spectrum(
        self,
        csm: np.ndarray,
        steering: np.ndarray,
    ) -> np.ndarray:
        """
        Compute MUSIC pseudo-spectrum for one frequency band.
        csm: (16, 16) smoothed covariance matrix
        steering: (16, N_grid) steering matrix for this center frequency
        Returns: (N_grid,) unnormalized pseudo-spectrum
        """
        # Regularize to avoid singular matrix
        R = csm + 1e-10 * np.eye(self.num_mics)

        # eigh returns eigenvalues in ascending order
        eigvals, eigvecs = np.linalg.eigh(R)

        # Noise subspace: all columns except the last (dominant signal)
        noise_subspace = eigvecs[:, :-1]  # shape: (16, 15)

        # Project steering vectors onto noise subspace: (15, 16) @ (16, N_grid) -> (15, N_grid)
        projection = noise_subspace.conj().T @ steering

        # Denominator: sum of squared projections
        denominator = np.sum(np.abs(projection) ** 2, axis=0) + 1e-12

        return 1.0 / denominator

    # ------------------------------------------------------------------
    # Localization
    # ------------------------------------------------------------------

    def localize(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute the normalized 2-D MUSIC heatmap across the angular grid.

        Input:
            frame: shape (FRAME_SIZE, 16)
        Output:
            heatmap: shape (grid_el, grid_az) normalized to [0, 1]
        """
        heatmap, _ = self.localize_with_peak(frame)
        return heatmap

    def localize_with_peak(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        """
        Execute the complete MUSIC localization pipeline with noise-gating,
        per-band min-max normalization, sigmoid contrast enhancement, and peak extraction.
        """
        data = np.asarray(frame, dtype=np.float64)
        if data.ndim == 1:
            data = data[:, None]

        if data.shape[0] != self.frame_size:
            raise ValueError(
                f"Expected {self.frame_size} samples, received {data.shape[0]}."
            )
        if data.shape[1] != self.num_mics:
            raise ValueError(
                f"Expected {self.num_mics} microphones, received {data.shape[1]}."
            )

        current_db = self.calculate_db_level(data)

        # Noise floor gating: below threshold, skip MUSIC and reset CSM history
        if current_db < self.noise_threshold_db:
            self.reset_csm_state()
            empty_heatmap = np.zeros(self.grid_shape, dtype=np.float64)
            center_col = len(self.az_values) // 2
            center_row = len(self.el_values) // 2
            peak = {
                "azimuth": float(self.az_values[center_col]),
                "elevation": float(self.el_values[center_row]),
                "value": 0.0,
                "row": center_row,
                "column": center_col,
                "db": current_db,
                "valid": False,
            }
            return empty_heatmap, peak

        # Step 1: STFT with Hann window
        window = np.hanning(self.frame_size)[:, None]
        windowed = data * window
        fft_out = np.fft.rfft(windowed, axis=0)  # (bins, channels)
        spectrum = fft_out.T  # (channels, bins)

        n_grid_points = self.grid_shape[0] * self.grid_shape[1]
        total_spectrum = np.zeros(n_grid_points, dtype=np.float64)

        # Step 2: Accumulate across 6 sub-bands
        if self.band_weighting_mode == "baseline":
            # Golden Reference Baseline: unweighted sum of normalized sub-band spectra
            for i, center in enumerate(self.freq_centers):
                idx_low, idx_high = self._band_bins[i]
                if idx_high <= idx_low:
                    continue

                csm_inst = self._compute_band_csm(spectrum, idx_low, idx_high)
                csm_smooth = self._smooth_csm(float(center), csm_inst)

                # Steering matrix precomputed for this exact center frequency
                steering = self.steering_factory.get_matrix(float(center))

                # MUSIC pseudo-spectrum for this band
                spectrum_band = self._music_band_spectrum(csm_smooth, steering)

                # EXACT PER-BAND MIN-MAX NORMALIZATION (Golden Reference)
                s_min = float(np.min(spectrum_band))
                s_max = float(np.max(spectrum_band))
                if s_max > s_min:
                    spectrum_band = (spectrum_band - s_min) / (s_max - s_min)

                total_spectrum += spectrum_band

        elif self.band_weighting_mode == "energy":
            # Candidate Variant: physical spectral energy-weighted accumulation
            band_energies = []
            band_spectra = []
            for i, center in enumerate(self.freq_centers):
                idx_low, idx_high = self._band_bins[i]
                if idx_high <= idx_low:
                    continue

                x_b = spectrum[:, idx_low:idx_high]
                energy = float(np.sum(np.abs(x_b) ** 2))
                band_energies.append(energy)

                csm_inst = self._compute_band_csm(spectrum, idx_low, idx_high)
                csm_smooth = self._smooth_csm(float(center), csm_inst)

                steering = self.steering_factory.get_matrix(float(center))
                spectrum_band = self._music_band_spectrum(csm_smooth, steering)

                s_min = float(np.min(spectrum_band))
                s_max = float(np.max(spectrum_band))
                if s_max > s_min:
                    norm_spec = (spectrum_band - s_min) / (s_max - s_min + 1e-12)
                else:
                    norm_spec = spectrum_band
                band_spectra.append(norm_spec)

            tot_energy = sum(band_energies) + 1e-12
            for i in range(len(band_spectra)):
                weight = band_energies[i] / tot_energy
                total_spectrum += weight * band_spectra[i]

        # Step 3: Global min-max normalization to [0, 1]
        t_min = float(np.min(total_spectrum))
        t_max = float(np.max(total_spectrum))
        if t_max > t_min:
            total_spectrum = (total_spectrum - t_min) / (t_max - t_min + 1e-9)

        # Step 4: Sigmoid contrast enhancement (gain=3.0, center=0.5)
        total_spectrum = 1.0 / (
            1.0 + np.exp(-self.sigmoid_gain * (total_spectrum - self.sigmoid_midpoint))
        )

        # Re-normalize to [0, 1]
        f_min = float(np.min(total_spectrum))
        f_max = float(np.max(total_spectrum))
        if f_max > f_min:
            total_spectrum = (total_spectrum - f_min) / (f_max - f_min + 1e-9)

        # Reshape to 2D grid: (76 rows elevation, 101 cols azimuth)
        heatmap = total_spectrum.reshape(self.grid_shape)
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)

        # Peak extraction
        peak_idx = int(np.argmax(total_spectrum))
        row, column = np.unravel_index(peak_idx, self.grid_shape)
        value = float(heatmap[row, column])
        azimuth = float(self.az_values[column])
        elevation = float(self.el_values[row])

        peak = {
            "azimuth": azimuth,
            "elevation": elevation,
            "value": value,
            "row": int(row),
            "column": int(column),
            "db": current_db,
            "valid": True,
        }

        return heatmap, peak

    def get_peak_position(
        self,
        heatmap: np.ndarray,
    ) -> dict[str, float | int | bool]:
        """Return the strongest MUSIC position from an existing heatmap."""
        heatmap = np.asarray(heatmap)
        if heatmap.ndim != 2:
            raise ValueError("Heatmap must be 2-dimensional.")
        if heatmap.shape != self.grid_shape:
            raise ValueError(
                f"Expected heatmap shape {self.grid_shape}, received {heatmap.shape}."
            )

        peak_idx = int(np.argmax(heatmap))
        row, column = np.unravel_index(peak_idx, heatmap.shape)
        value = float(heatmap[row, column])

        return {
            "azimuth": float(self.az_values[column]),
            "elevation": float(self.el_values[row]),
            "value": value,
            "row": int(row),
            "column": int(column),
            "valid": value > 0.0,
        }

    # ------------------------------------------------------------------
    # AudioFrame Integration
    # ------------------------------------------------------------------

    def localize_audio_frame(
        self,
        audio_frame: AudioFrame,
    ) -> AcousticFrame:
        """Convert one AudioFrame into a project AcousticFrame."""
        if not isinstance(audio_frame, AudioFrame):
            raise TypeError("localize_audio_frame expects an AudioFrame.")

        data = np.asarray(audio_frame.data)
        if data.ndim != 2:
            raise ValueError("AudioFrame data must be 2-dimensional.")
        if data.shape != (self.frame_size, self.num_mics):
            raise ValueError(
                f"Expected audio shape ({self.frame_size}, {self.num_mics}), received {data.shape}."
            )

        start = time.perf_counter()
        heatmap, peak = self.localize_with_peak(data)
        processing_time = time.perf_counter() - start

        confidence = float(np.clip(peak["value"], 0.0, 1.0))
        is_valid = bool(peak["valid"])

        localization = AcousticLocalization(
            x=0.0,
            y=0.0,
            z=0.0,
            response=float(peak["value"]),
            frequency=None,
            confidence=confidence,
            valid=is_valid,
            source_count=self.n_sources,
        )

        return AcousticFrame(
            frame_id=audio_frame.frame_id,
            timestamp=audio_frame.timestamp,
            localization=localization,
            sample_rate=float(self.sample_rate),
            frame_size=int(self.frame_size),
            processing_time=processing_time,
            heatmap=heatmap,
            metadata={
                "source": "MusicLocalizer",
                "peak_row": int(peak["row"]),
                "peak_column": int(peak["column"]),
                "azimuth_deg": float(peak["azimuth"]),
                "elevation_deg": float(peak["elevation"]),
                "heatmap_shape": tuple(int(v) for v in heatmap.shape),
                "heatmap": heatmap,
                "num_microphones": self.num_mics,
                "num_frequencies": len(self.freq_centers),
                "db_level": float(peak.get("db", 0.0)),
                "valid": is_valid,
            },
        )
