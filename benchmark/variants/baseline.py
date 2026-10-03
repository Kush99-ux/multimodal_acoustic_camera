"""
benchmark.variants.baseline
===========================
Unmodified production MusicLocalizer baseline wrapper.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

from benchmark.variants.base_variant import AcousticAlgorithmVariant
from acoustic.music import MusicLocalizer


class BaselineProductionVariant(AcousticAlgorithmVariant):
    """
    Wraps the current production MusicLocalizer exactly as it exists.
    """

    def __init__(self) -> None:
        super().__init__(
            name="baseline_production",
            description="Current production 6-band MUSIC localizer with EMA alpha=0.30 and per-band min-max normalization",
            parameters={
                "frequencies": [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0],
                "band_width": 500.0,
                "csm_alpha": 0.30,
                "normalization": "per_band_min_max",
                "sigmoid_gain": 3.0,
                "noise_threshold_db": 15.0,
            },
        )
        self.localizer = MusicLocalizer()

    def localize(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.localizer.localize_with_peak(frame)

    def reset_state(self) -> None:
        self.localizer.reset_csm_state()
