"""
benchmark.models.signal
=======================
Multi-channel synthesized audio frame representation.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MultiChannelSignal:
    """
    16-channel audio frame representation.
    
    Attributes:
        data: NumPy array of shape (samples, channels), typically (4096, 16)
        sample_rate: Audio sampling frequency (Hz), default 48000
        snr_db: Injected Signal-to-Noise Ratio (dB)
        clean_data: Optional noise-free signal for reference
        noise_data: Optional isolated noise component
        delays_samples: True fractional delays applied per channel
    """
    data: np.ndarray
    sample_rate: int = 48000
    snr_db: Optional[float] = None
    clean_data: Optional[np.ndarray] = None
    noise_data: Optional[np.ndarray] = None
    delays_samples: Optional[np.ndarray] = None

    @property
    def num_samples(self) -> int:
        return self.data.shape[0]

    @property
    def num_channels(self) -> int:
        return self.data.shape[1]

    @property
    def duration_seconds(self) -> float:
        return float(self.num_samples) / float(self.sample_rate)

    @property
    def rms_per_channel(self) -> np.ndarray:
        return np.sqrt(np.mean(self.data ** 2, axis=0))

    @property
    def overall_rms(self) -> float:
        return float(np.sqrt(np.mean(self.data ** 2)))

    @property
    def db_spl(self) -> float:
        rms = self.overall_rms
        return float(20.0 * np.log10(rms + 1e-9) + 90.0)
