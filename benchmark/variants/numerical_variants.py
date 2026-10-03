"""
benchmark.variants.numerical_variants
=====================================
Float32 vs Float64 precision comparison variant.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

from benchmark.variants.base_variant import AcousticAlgorithmVariant
from acoustic.music import MusicLocalizer


class PrecisionVariant(AcousticAlgorithmVariant):
    """
    Executes MUSIC with explicit Float32 precision to assess numerical stability.
    """

    def __init__(self, use_float32: bool = True) -> None:
        dtype_str = "float32" if use_float32 else "float64"
        super().__init__(
            name=f"precision_{dtype_str}",
            description=f"MUSIC localizer running with {dtype_str} precision",
            parameters={"dtype": dtype_str},
        )
        self.use_float32 = use_float32
        self.localizer = MusicLocalizer()
        self.c_dtype = np.complex64 if use_float32 else np.complex128
        self.r_dtype = np.float32 if use_float32 else np.float64

    def localize(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        data = np.asarray(frame, dtype=self.r_dtype)
        current_db = float(self.localizer.calculate_db_level(data))
        if current_db < self.localizer.noise_threshold_db:
            self.reset_state()
            return self.localizer.localize_with_peak(frame)

        window = np.hanning(self.localizer.frame_size).astype(self.r_dtype)[:, None]
        fft_out = np.fft.rfft(data * window, axis=0)
        spectrum = fft_out.T.astype(self.c_dtype)

        n_grid = self.localizer.grid_shape[0] * self.localizer.grid_shape[1]
        total_spectrum = np.zeros(n_grid, dtype=self.r_dtype)

        for i, center in enumerate(self.localizer.freq_centers):
            idx_low, idx_high = self.localizer._band_bins[i]
            x_b = spectrum[:, idx_low:idx_high]
            b_cnt = x_b.shape[1]
            csm_inst = (x_b @ x_b.conj().T) / float(b_cnt)
            csm_smooth = self.localizer._smooth_csm(float(center), csm_inst)
            csm_smooth = csm_smooth.astype(self.c_dtype)

            # Eigendecomposition
            R = csm_smooth + (1e-7 if self.use_float32 else 1e-10) * np.eye(self.localizer.num_mics, dtype=self.c_dtype)
            eigvals, eigvecs = np.linalg.eigh(R)
            noise_subspace = eigvecs[:, :-1]

            steering = self.localizer.steering_factory.get_matrix(float(center)).astype(self.c_dtype)
            projection = noise_subspace.conj().T @ steering
            denom = np.sum(np.abs(projection) ** 2, axis=0) + (1e-8 if self.use_float32 else 1e-12)
            spec_band = (1.0 / denom).astype(self.r_dtype)

            s_min, s_max = np.min(spec_band), np.max(spec_band)
            if s_max > s_min:
                spec_band = (spec_band - s_min) / (s_max - s_min)
            total_spectrum += spec_band

        t_min, t_max = np.min(total_spectrum), np.max(total_spectrum)
        if t_max > t_min:
            total_spectrum = (total_spectrum - t_min) / (t_max - t_min + 1e-6)
        total_spectrum = 1.0 / (1.0 + np.exp(-3.0 * (total_spectrum - 0.5)))
        f_min, f_max = np.min(total_spectrum), np.max(total_spectrum)
        if f_max > f_min:
            total_spectrum = (total_spectrum - f_min) / (f_max - f_min + 1e-6)

        heatmap = total_spectrum.reshape(self.localizer.grid_shape).astype(np.float64)
        peak_idx = int(np.argmax(total_spectrum))
        row, col = np.unravel_index(peak_idx, self.localizer.grid_shape)

        peak = {
            "azimuth": float(self.localizer.az_values[col]),
            "elevation": float(self.localizer.el_values[row]),
            "value": float(heatmap[row, col]),
            "row": int(row),
            "column": int(col),
            "db": current_db,
            "valid": True,
        }
        return heatmap, peak

    def reset_state(self) -> None:
        self.localizer.reset_csm_state()
