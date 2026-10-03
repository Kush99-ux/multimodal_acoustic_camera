"""
benchmark.experiments.exp04_waveforms
"""
from __future__ import annotations
import os, sys, time
script_dir = os.path.dirname(os.path.abspath(__file__))
while script_dir in sys.path:
    sys.path.remove(script_dir)
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from typing import Dict, Any, List
from benchmark.variants.baseline import BaselineProductionVariant
from benchmark.core.generator import SyntheticAudioGenerator
from benchmark.models.source import SoundSource, WaveformType
from benchmark.models.scenario import BenchmarkScenario, ReflectionPath, MicrophoneImpairment
from benchmark.core.metrics import evaluate_trial, calculate_angular_error
from benchmark.config.benchmark_config import PLOTS_DIR
import benchmark.core.plotting as bp

def run_single_trial(localizer, generator, scenario, trial_index=0):
    sig = generator.generate(scenario)
    t0 = time.perf_counter()
    heatmap, peak = localizer.localize(sig.data)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    true_az = scenario.sources[0].azimuth_deg if scenario.sources else 90.0
    true_el = scenario.sources[0].elevation_deg if scenario.sources else 0.0
    return evaluate_trial(
        scenario_id=scenario.scenario_id,
        trial_index=trial_index,
        true_az=true_az,
        true_el=true_el,
        heatmap=heatmap,
        peak_dict=peak,
        processing_time_ms=elapsed_ms,
    )

def run_waveform_experiment(generator=None, localizer=None) -> Dict[str, Any]:
    if generator is None: generator = SyntheticAudioGenerator()
    if localizer is None: localizer = BaselineProductionVariant()

    sig_configs = [
        (WaveformType.TONE, "Pure Tone (1500 Hz)", {"frequency_hz": 1500.0}),
        (WaveformType.MULTI_TONE, "Multi-Tone (6 Freqs)", {"frequency_hz": 1500.0}),
        (WaveformType.CHIRP, "Linear Chirp (500-3000 Hz)", {"frequency_range_hz": (500.0, 3000.0)}),
        (WaveformType.WHITE_NOISE, "White Noise Band", {}),
        (WaveformType.PINK_NOISE, "Pink Noise Band", {}),
        (WaveformType.TRANSIENT_CLAP, "Transient Clap (10ms)", {"start_time_s": 0.02, "duration_s": 0.01}),
        (WaveformType.SPEECH_SYNTHETIC, "Synthetic Speech Vowel", {}),
    ]

    results = []
    for wtype, label, kwargs in sig_configs:
        sc = BenchmarkScenario(
            scenario_id=f"wf_{wtype.value}",
            category="waveforms",
            description=label,
            sources=[SoundSource(azimuth_deg=103.0, elevation_deg=-5.0, waveform=wtype, **kwargs)],
            snr_db=30.0,
        )
        localizer.reset_state()
        res = run_single_trial(localizer, generator, sc)
        results.append({
            "type": wtype.value,
            "label": label,
            "error_deg": res.angular_error_deg,
            "contrast": res.peak_contrast,
            "db_level": res.db_level,
            "passed": res.passed_tolerance,
        })

    return {"waveforms": results}

if __name__ == "__main__":
    print("[Exp 04] Running waveform comparison experiment...")
    res = run_waveform_experiment()
    for w in res["waveforms"]:
        print(f"  {w['label']:30s} -> Err: {w['error_deg']:4.2f}°, Passed: {w['passed']}")
