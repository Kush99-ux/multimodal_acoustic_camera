"""
benchmark.variants.csm_variants
===============================
CSM temporal smoothing algorithm variants.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List
from collections import deque
import numpy as np

from benchmark.variants.base_variant import AcousticAlgorithmVariant
from acoustic.music import MusicLocalizer


class CSMAlphaVariant(AcousticAlgorithmVariant):
    """
    Custom EMA alpha smoothing variant.
    """

    def __init__(self, alpha: float, name_suffix: Optional[str] = None) -> None:
        suffix = name_suffix if name_suffix is not None else f"{int(round(alpha * 100)):02d}"
        super().__init__(
            name=f"csm_ema_{suffix}",
            description=f"MUSIC localizer with modified CSM EMA alpha={alpha:.2f}",
            parameters={"csm_alpha": alpha},
        )
        self.localizer = MusicLocalizer(smoothing=alpha)

    def localize(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.localizer.localize_with_peak(frame)

    def reset_state(self) -> None:
        self.localizer.reset_csm_state()


class CSMSlidingWindowVariant(AcousticAlgorithmVariant):
    """
    Sliding window average of K consecutive instantaneous CSMs instead of exponential decay.
    """

    def __init__(self, window_len: int = 3) -> None:
        super().__init__(
            name=f"csm_sliding_window_k{window_len}",
            description=f"MUSIC localizer with finite sliding-window CSM average (K={window_len} frames)",
            parameters={"window_length": window_len},
        )
        self.window_len = window_len
        self.localizer = MusicLocalizer(smoothing=1.0)  # instantaneous base
        self._history: Dict[float, deque[np.ndarray]] = {
            float(f): deque(maxlen=window_len) for f in self.localizer.freq_centers
        }

    def localize(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        data = np.asarray(frame, dtype=np.float64)
        current_db = self.localizer.calculate_db_level(data)
        if current_db < self.localizer.noise_threshold_db:
            self.reset_state()
            return self.localizer.localize_with_peak(frame)

        # Compute STFT
        window = np.hanning(self.localizer.frame_size)[:, None]
        fft_out = np.fft.rfft(data * window, axis=0)
        spectrum = fft_out.T

        n_grid = self.localizer.grid_shape[0] * self.localizer.grid_shape[1]
        total_spectrum = np.zeros(n_grid, dtype=np.float64)

        for i, center in enumerate(self.localizer.freq_centers):
            idx_low, idx_high = self.localizer._band_bins[i]
            csm_inst = self.localizer._compute_band_csm(spectrum, idx_low, idx_high)
            self._history[float(center)].append(csm_inst)

            # Average over sliding window
            csm_avg = np.mean(self._history[float(center)], axis=0)
            csm_avg = 0.5 * (csm_avg + csm_avg.conj().T)

            steering = self.localizer.steering_factory.get_matrix(float(center))
            spectrum_band = self.localizer._music_band_spectrum(csm_avg, steering)

            # Per-band min-max
            s_min, s_max = np.min(spectrum_band), np.max(spectrum_band)
            if s_max > s_min:
                spectrum_band = (spectrum_band - s_min) / (s_max - s_min)
            total_spectrum += spectrum_band

        # Sigmoid contrast shaping
        t_min, t_max = np.min(total_spectrum), np.max(total_spectrum)
        if t_max > t_min:
            total_spectrum = (total_spectrum - t_min) / (t_max - t_min + 1e-9)
        total_spectrum = 1.0 / (1.0 + np.exp(-3.0 * (total_spectrum - 0.5)))
        f_min, f_max = np.min(total_spectrum), np.max(total_spectrum)
        if f_max > f_min:
            total_spectrum = (total_spectrum - f_min) / (f_max - f_min + 1e-9)

        heatmap = total_spectrum.reshape(self.localizer.grid_shape)
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
        for f in self.localizer.freq_centers:
            self._history[float(f)].clear()
