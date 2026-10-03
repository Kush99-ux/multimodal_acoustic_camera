"""
demo.core.diagnostics
=====================
Diagnostics calculation helpers for the 16-channel microphone array and 6-band spectrum.
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Tuple, Optional
import numpy as np

from config.settings import (
    SAMPLE_RATE,
    FRAME_SIZE,
    FREQ_CENTERS,
    BAND_WIDTH,
    SYNC_TOLERANCE_MS,
)


# Physical layout matrix of UMA-16 microphones:
# MIC8   MIC7   MIC10  MIC9
# MIC6   MIC5   MIC12  MIC11
# MIC4   MIC3   MIC14  MIC13
# MIC2   MIC1   MIC16  MIC15
MIC_LAYOUT_GRID = [
    [8, 7, 10, 9],
    [6, 5, 12, 11],
    [4, 3, 14, 13],
    [2, 1, 16, 15],
]


class StreamDiagnostics:
    """Computes genuine real-time audio and synchronization diagnostics."""

    def __init__(self, history_size: int = 60) -> None:
        self.history_size = int(history_size)
        self.sync_deltas: deque[float] = deque(maxlen=self.history_size)
        self.freq_centers = [float(f) for f in FREQ_CENTERS]
        self.band_width = float(BAND_WIDTH)
        self.sample_rate = float(SAMPLE_RATE)
        self.frame_size = int(FRAME_SIZE)

        # Precompute band bin indices
        freqs = np.fft.rfftfreq(self.frame_size, 1.0 / self.sample_rate)
        self._band_masks: list[np.ndarray] = []
        for fc in self.freq_centers:
            mask = (freqs >= (fc - self.band_width / 2.0)) & (freqs <= (fc + self.band_width / 2.0))
            self._band_masks.append(mask)

    def record_sync_delta(self, delta_ms: float) -> None:
        """Record one audio-video timestamp discrepancy."""
        self.sync_deltas.append(float(delta_ms))

    def get_sync_stats(self) -> Dict[str, float]:
        """Return rolling synchronization performance statistics."""
        if not self.sync_deltas:
            return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0, "std": 0.0, "count": 0}
        arr = np.array(self.sync_deltas)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "std": float(np.std(arr)),
            "count": len(arr),
            "pass_rate": float(np.mean(arr <= SYNC_TOLERANCE_MS) * 100.0),
        }

    def compute_mic_rms(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Compute root-mean-square energy per microphone channel.
        Returns an array of 16 float values normalized to [0, 1].
        """
        if audio_data is None or audio_data.ndim != 2:
            return np.zeros(16, dtype=np.float32)

        # audio_data shape: (samples, channels)
        rms = np.sqrt(np.mean(audio_data ** 2, axis=0))
        if len(rms) < 16:
            padded = np.zeros(16, dtype=np.float32)
            padded[:len(rms)] = rms
            rms = padded
        else:
            rms = rms[:16]

        # Convert to dB SPL approximation [-60 dB, 0 dB] -> [0.0, 1.0]
        db = 20.0 * np.log10(np.clip(rms, 1e-6, 1.0))
        norm = np.clip((db + 60.0) / 50.0, 0.0, 1.0)
        return norm.astype(np.float32)

    def compute_band_energies(self, audio_data: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Compute real sub-band energy across the 6 discrete MUSIC processing bands.
        Returns list of (center_hz, energy_fraction_0_to_1, db_val).
        """
        if audio_data is None or audio_data.ndim != 2:
            return [(fc, 0.0, -60.0) for fc in self.freq_centers]

        window = np.hanning(len(audio_data))[:, None]
        fft = np.fft.rfft(audio_data * window, axis=0)
        power = np.mean(np.abs(fft) ** 2, axis=1)  # Average power across 16 channels

        results = []
        for fc, mask in zip(self.freq_centers, self._band_masks):
            if np.any(mask):
                val = float(np.mean(power[mask]))
            else:
                val = 0.0
            db = float(10.0 * np.log10(np.clip(val, 1e-9, 1e6)))
            # Map [-40 dB, +10 dB] into [0, 1]
            frac = float(np.clip((db + 40.0) / 45.0, 0.0, 1.0))
            results.append((fc, frac, db))

        return results
