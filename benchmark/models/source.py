"""
benchmark.models.source
=======================
Data structures describing acoustic sound sources.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class WaveformType(str, Enum):
    TONE = "tone"
    MULTI_TONE = "multi_tone"
    CHIRP = "chirp"
    WHITE_NOISE = "white_noise"
    PINK_NOISE = "pink_noise"
    TRANSIENT_CLAP = "transient_clap"
    SPEECH_SYNTHETIC = "speech_synthetic"
    IMPULSE = "impulse"


@dataclass
class SoundSource:
    """
    Representation of a sound source in 3D angular space.
    
    Attributes:
        azimuth_deg: Source azimuth angle (-180..+180, where 90 = broadside)
        elevation_deg: Source elevation angle (-90..+90, where 0 = horizontal)
        distance_m: Distance from array center in meters (default 1.5m)
        waveform: Type of synthesized acoustic waveform
        frequency_hz: Nominal frequency for tones (or center frequency)
        frequency_range_hz: (f_min, f_max) for chirps/multi-tone
        amplitude: Peak amplitude of normalized signal (0..1)
        start_time_s: Start time of transient/burst within frame
        duration_s: Duration of active signal within frame
    """
    azimuth_deg: float = 90.0
    elevation_deg: float = 0.0
    distance_m: float = 1.5
    waveform: WaveformType = WaveformType.TONE
    frequency_hz: float = 1000.0
    frequency_range_hz: Tuple[float, float] = (500.0, 3000.0)
    amplitude: float = 0.8
    start_time_s: float = 0.0
    duration_s: Optional[float] = None
    name: str = "primary_source"

    @property
    def unit_vector(self) -> np.ndarray:
        """
        Cartesian unit direction vector [x, y, z] matching Acoular steering frame:
        x = cos(el) * cos(az)
        y = cos(el) * sin(az)  (forward broadside)
        z = sin(el)           (vertical)
        """
        az_rad = np.deg2rad(self.azimuth_deg)
        el_rad = np.deg2rad(self.elevation_deg)
        x = np.cos(el_rad) * np.cos(az_rad)
        y = np.cos(el_rad) * np.sin(az_rad)
        z = np.sin(el_rad)
        return np.array([x, y, z], dtype=np.float64)

    @property
    def position_m(self) -> np.ndarray:
        """3D Cartesian position vector of the source."""
        return self.unit_vector * self.distance_m
