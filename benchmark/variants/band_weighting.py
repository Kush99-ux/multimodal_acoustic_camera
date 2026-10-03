"""
benchmark.variants.band_weighting
=================================
Alternative sub-band accumulation and weighting strategies.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

from benchmark.variants.base_variant import AcousticAlgorithmVariant
from acoustic.music import MusicLocalizer


class BandWeightingVariant(AcousticAlgorithmVariant):
    """
    Sub-band accumulation strategy variant.
    Strategies:
      - 'raw': No per-band normalization (accumulate raw pseudo-spectra)
      - 'energy': Weight each band by its total band power
      - 'dynamic_range': Weight each band by (max - min) / median
      - 'active_gated': Drop bands whose power is below 10% of max band power
    """

    def __init__(self, strategy: str = "energy") -> None:
        super().__init__(
            name=f"band_weighting_{strategy}",
            description=f"MUSIC localizer with '{strategy}' sub-band accumulation",
            parameters={"strategy": strategy},
        )
        self.strategy = strategy
        self.localizer = MusicLocalizer()

    def localize(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        data = np.asarray(frame, dtype=np.float64)
        current_db = self.localizer.calculate_db_level(data)
        if current_db < self.localizer.noise_threshold_db:
            self.reset_state()
            return self.localizer.localize_with_peak(frame)

        # STFT
        window = np.hanning(self.localizer.frame_size)[:, None]
        fft_out = np.fft.rfft(data * window, axis=0)
        spectrum = fft_out.T

        n_grid = self.localizer.grid_shape[0] * self.localizer.grid_shape[1]
        total_spectrum = np.zeros(n_grid, dtype=np.float64)

        # Precompute band energies
        band_energies = []
        band_spectra = []
        for i, center in enumerate(self.localizer.freq_centers):
            idx_low, idx_high = self.localizer._band_bins[i]
            x_b = spectrum[:, idx_low:idx_high]
            energy = float(np.sum(np.abs(x_b) ** 2))
            band_energies.append(energy)

            csm_inst = self.localizer._compute_band_csm(spectrum, idx_low, idx_high)
            csm_smooth = self.localizer._smooth_csm(float(center), csm_inst)
            steering = self.localizer.steering_factory.get_matrix(float(center))
            spec = self.localizer._music_band_spectrum(csm_smooth, steering)
            band_spectra.append(spec)

        tot_energy = sum(band_energies) + 1e-12
        max_energy = max(band_energies) + 1e-12

        for i in range(len(self.localizer.freq_centers)):
            spec = band_spectra[i]
            s_min, s_max = np.min(spec), np.max(spec)
            norm_spec = (spec - s_min) / (s_max - s_min + 1e-12) if s_max > s_min else spec

            if self.strategy == "raw":
                # Unnormalized accumulation
                total_spectrum += spec

            elif self.strategy == "energy":
                # Weight by band energy fraction
                weight = band_energies[i] / tot_energy
                total_spectrum += weight * norm_spec

            elif self.strategy == "dynamic_range":
                # Weight by peak-to-median contrast
                med = float(np.median(spec)) + 1e-12
                dr = (s_max - s_min) / med
                total_spectrum += dr * norm_spec

            elif self.strategy == "active_gated":
                # Only include bands with >= 10% of maximum band energy
                if band_energies[i] >= 0.10 * max_energy:
                    total_spectrum += norm_spec

            else:
                total_spectrum += norm_spec

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
        self.localizer.reset_csm_state()
