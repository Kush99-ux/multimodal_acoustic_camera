"""
benchmark.config.scenarios
==========================
Catalog of standard benchmark scenarios across quick, standard, and full suites.
"""

from __future__ import annotations
import numpy as np
from typing import List
from benchmark.models.source import SoundSource, WaveformType
from benchmark.models.scenario import BenchmarkScenario, ReflectionPath, MicrophoneImpairment
from benchmark.config.benchmark_config import (
    GOLDEN_DIRECTIONS,
    AZIMUTH_MIN_DEG,
    AZIMUTH_MAX_DEG,
    ELEVATION_MIN_DEG,
    ELEVATION_MAX_DEG,
)


def get_quick_scenarios() -> List[BenchmarkScenario]:
    """
    Generate scenarios for rapid regression verification (~15-30s).
    Includes golden directions, core frequencies, SNR checks, and transient test.
    """
    scenarios = []

    # 1. Golden directions regression
    for idx, (az, el) in enumerate(GOLDEN_DIRECTIONS):
        scenarios.append(
            BenchmarkScenario(
                scenario_id=f"quick_dir_{idx+1:02d}",
                category="direction_accuracy",
                description=f"Golden regression direction Az={az:.1f}, El={el:.1f}",
                sources=[SoundSource(azimuth_deg=az, elevation_deg=el, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
                snr_db=30.0,
            )
        )

    # 2. Key frequencies: 500, 1000, 2000, 3000 Hz
    for freq in [500.0, 1000.0, 2000.0, 3000.0]:
        scenarios.append(
            BenchmarkScenario(
                scenario_id=f"quick_freq_{int(freq)}",
                category="frequency_behavior",
                description=f"Frequency test at {int(freq)} Hz",
                sources=[SoundSource(azimuth_deg=90.0, elevation_deg=0.0, waveform=WaveformType.TONE, frequency_hz=freq)],
                snr_db=30.0,
            )
        )

    # 3. SNR checks: 30 dB, 15 dB, 0 dB
    for snr in [30.0, 15.0, 0.0]:
        scenarios.append(
            BenchmarkScenario(
                scenario_id=f"quick_snr_{int(snr)}db",
                category="snr_robustness",
                description=f"SNR robustness test at {int(snr)} dB",
                sources=[SoundSource(azimuth_deg=103.0, elevation_deg=-5.0, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
                snr_db=snr,
                noise_type="white",
            )
        )

    # 4. Transient clap
    scenarios.append(
        BenchmarkScenario(
            scenario_id="quick_transient_clap",
            category="transient_behavior",
            description="Short transient clap burst at Az=103 deg",
            sources=[SoundSource(azimuth_deg=103.0, elevation_deg=0.0, waveform=WaveformType.TRANSIENT_CLAP, start_time_s=0.02, duration_s=0.01)],
            snr_db=25.0,
        )
    )

    return scenarios


def get_standard_scenarios() -> List[BenchmarkScenario]:
    """
    Generate standard comprehensive research benchmark scenarios.
    """
    scenarios = []

    # 1. Direction grid: Az 65..115 (step 5 deg), El -15..+15 (step 5 deg) -> 11 x 7 = 77 points
    az_grid = np.arange(65.0, 116.0, 5.0)
    el_grid = np.arange(-15.0, 16.0, 5.0)
    for az in az_grid:
        for el in el_grid:
            scenarios.append(
                BenchmarkScenario(
                    scenario_id=f"std_grid_az{int(az)}_el{int(el)}",
                    category="direction_accuracy",
                    description=f"Grid test Az={az:.1f}, El={el:.1f}",
                    sources=[SoundSource(azimuth_deg=az, elevation_deg=el, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
                    snr_db=30.0,
                )
            )

    # 2. Frequency sweep: 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 4000 Hz
    freqs = [500.0, 750.0, 1000.0, 1250.0, 1500.0, 1750.0, 2000.0, 2250.0, 2500.0, 2750.0, 3000.0, 3250.0, 3500.0, 4000.0]
    for freq in freqs:
        scenarios.append(
            BenchmarkScenario(
                scenario_id=f"std_freq_{int(freq)}",
                category="frequency_behavior",
                description=f"Frequency sweep at {int(freq)} Hz",
                sources=[SoundSource(azimuth_deg=103.0, elevation_deg=-5.0, waveform=WaveformType.TONE, frequency_hz=freq)],
                snr_db=30.0,
                metadata={"in_band": (500.0 <= freq <= 3000.0)},
            )
        )

    # 3. SNR and Noise sweep: +30, +20, +15, +10, +5, 0, -5 dB (white & pink)
    for n_type in ["white", "pink"]:
        for snr in [30.0, 20.0, 15.0, 10.0, 5.0, 0.0, -5.0]:
            scenarios.append(
                BenchmarkScenario(
                    scenario_id=f"std_snr_{n_type}_{int(snr)}db",
                    category="snr_robustness",
                    description=f"SNR test {n_type} noise at {int(snr)} dB",
                    sources=[SoundSource(azimuth_deg=103.0, elevation_deg=0.0, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
                    snr_db=snr,
                    noise_type=n_type,
                )
            )

    # 4. Signal types: tone, multi-tone, chirp, white noise, pink noise, transient clap, speech
    sig_types = [
        (WaveformType.TONE, "pure_tone"),
        (WaveformType.MULTI_TONE, "multi_tone"),
        (WaveformType.CHIRP, "chirp_500_3000"),
        (WaveformType.WHITE_NOISE, "white_noise_band"),
        (WaveformType.PINK_NOISE, "pink_noise_band"),
        (WaveformType.TRANSIENT_CLAP, "transient_clap"),
        (WaveformType.SPEECH_SYNTHETIC, "speech_synthetic"),
    ]
    for wtype, label in sig_types:
        scenarios.append(
            BenchmarkScenario(
                scenario_id=f"std_signal_{label}",
                category="signal_types",
                description=f"Signal type test: {label}",
                sources=[SoundSource(azimuth_deg=90.0, elevation_deg=0.0, waveform=wtype, frequency_hz=1000.0)],
                snr_db=25.0,
            )
        )

    # 5. Transient position within frame (start, 25%, 50%, 75%, end)
    positions = [0.005, 0.021, 0.042, 0.064, 0.078]
    pos_labels = ["start", "pos_25pct", "pos_50pct", "pos_75pct", "near_end"]
    for t_pos, plabel in zip(positions, pos_labels):
        scenarios.append(
            BenchmarkScenario(
                scenario_id=f"std_transient_{plabel}",
                category="transient_position",
                description=f"Transient burst at {plabel} (t={t_pos*1000:.1f}ms)",
                sources=[SoundSource(azimuth_deg=103.0, elevation_deg=0.0, waveform=WaveformType.TRANSIENT_CLAP, start_time_s=t_pos, duration_s=0.006)],
                snr_db=25.0,
            )
        )

    # 6. Multipath reflection test: direct source + reflection at different attenuation levels (-10, -6, -3, 0 dB)
    for ref_db, amp in [(-10, 0.316), (-6, 0.501), (-3, 0.707), (0, 1.0)]:
        scenarios.append(
            BenchmarkScenario(
                scenario_id=f"std_multipath_{ref_db}db",
                category="multipath",
                description=f"Multipath test with reflection at {ref_db} dB",
                sources=[SoundSource(azimuth_deg=90.0, elevation_deg=0.0, waveform=WaveformType.TONE, frequency_hz=1500.0)],
                reflections=[ReflectionPath(azimuth_deg=110.0, elevation_deg=5.0, relative_amplitude=amp, delay_ms=4.0)],
                snr_db=30.0,
            )
        )

    # 7. Multiple sources: 2 sources at varying angular separation (10, 20, 30 deg)
    for sep in [10.0, 20.0, 30.0]:
        az1 = 90.0 - sep / 2.0
        az2 = 90.0 + sep / 2.0
        scenarios.append(
            BenchmarkScenario(
                scenario_id=f"std_multisource_sep_{int(sep)}deg",
                category="multiple_sources",
                description=f"Dual simultaneous sources with {int(sep)} deg separation",
                sources=[
                    SoundSource(azimuth_deg=az1, elevation_deg=0.0, waveform=WaveformType.TONE, frequency_hz=1000.0, name="source_1"),
                    SoundSource(azimuth_deg=az2, elevation_deg=0.0, waveform=WaveformType.TONE, frequency_hz=2000.0, name="source_2"),
                ],
                snr_db=30.0,
            )
        )

    # 8. Microphone impairments (gain mismatch, phase mismatch, 1 dead mic, 2 dead mics)
    scenarios.append(
        BenchmarkScenario(
            scenario_id="std_mic_gain_mismatch",
            category="microphone_faults",
            description="Microphone array with +/- 10% random gain errors",
            sources=[SoundSource(azimuth_deg=103.0, elevation_deg=-5.0, waveform=WaveformType.TONE, frequency_hz=1500.0)],
            impairment=MicrophoneImpairment(gain_mismatches=[1.0 + 0.1 * np.sin(i) for i in range(16)]),
            snr_db=30.0,
        )
    )
    scenarios.append(
        BenchmarkScenario(
            scenario_id="std_mic_phase_jitter",
            category="microphone_faults",
            description="Microphone array with +/- 15 deg random phase jitter",
            sources=[SoundSource(azimuth_deg=103.0, elevation_deg=-5.0, waveform=WaveformType.TONE, frequency_hz=1500.0)],
            impairment=MicrophoneImpairment(phase_jitter_rad=[np.deg2rad(15.0 * np.cos(i)) for i in range(16)]),
            snr_db=30.0,
        )
    )
    scenarios.append(
        BenchmarkScenario(
            scenario_id="std_mic_1_dead",
            category="microphone_faults",
            description="Microphone array with 1 dead microphone (CH4)",
            sources=[SoundSource(azimuth_deg=103.0, elevation_deg=-5.0, waveform=WaveformType.TONE, frequency_hz=1500.0)],
            impairment=MicrophoneImpairment(dead_channels=[3]),
            snr_db=30.0,
        )
    )
    scenarios.append(
        BenchmarkScenario(
            scenario_id="std_mic_2_dead",
            category="microphone_faults",
            description="Microphone array with 2 dead microphones (CH4, CH12)",
            sources=[SoundSource(azimuth_deg=103.0, elevation_deg=-5.0, waveform=WaveformType.TONE, frequency_hz=1500.0)],
            impairment=MicrophoneImpairment(dead_channels=[3, 11]),
            snr_db=30.0,
        )
    )

    return scenarios
