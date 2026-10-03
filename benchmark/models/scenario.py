"""
benchmark.models.scenario
=========================
Specification for individual benchmark test scenarios.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from benchmark.models.source import SoundSource


@dataclass
class ReflectionPath:
    """Multipath reflection component."""
    azimuth_deg: float
    elevation_deg: float
    relative_amplitude: float  # e.g. 0.5 (-6 dB)
    delay_ms: float            # e.g. 5.0 ms


@dataclass
class MicrophoneImpairment:
    """Simulated hardware mic array imperfections."""
    gain_mismatches: Optional[List[float]] = None      # Multiplier per channel (e.g. 1.0 +- 0.05)
    phase_jitter_rad: Optional[List[float]] = None     # Phase error per channel
    timing_jitter_ms: Optional[List[float]] = None     # Timing delay per channel
    dead_channels: List[int] = field(default_factory=list)  # Channels that output 0 or pure noise


@dataclass
class BenchmarkScenario:
    """
    Complete configuration for one synthetic benchmark trial.
    """
    scenario_id: str
    category: str
    description: str
    sources: List[SoundSource] = field(default_factory=list)
    snr_db: float = 30.0
    noise_type: str = "white"   # "white", "pink", "low_frequency", "narrowband"
    reflections: List[ReflectionPath] = field(default_factory=list)
    impairment: Optional[MicrophoneImpairment] = None
    frame_size: int = 4096
    sample_rate: int = 48000
    speed_of_sound: float = 343.0
    repeat_trials: int = 1
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)
