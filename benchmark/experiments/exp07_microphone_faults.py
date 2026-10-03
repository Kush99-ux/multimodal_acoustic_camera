"""
benchmark.experiments.exp07_microphone_faults
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

def run_microphone_faults_experiment(generator=None, localizer=None) -> Dict[str, Any]:
    if generator is None: generator = SyntheticAudioGenerator()
    if localizer is None: localizer = BaselineProductionVariant()

    dead_mic_counts = [0, 1, 2, 3, 4]
    dead_lists = [[], [0], [0, 8], [0, 4, 8], [0, 4, 8, 12]]
    errors = []
    contrasts = []

    for count, dead_mics in zip(dead_mic_counts, dead_lists):
        sc = BenchmarkScenario(
            scenario_id=f"fault_dead_{count}",
            category="fault_tolerance",
            description=f"{count} dead mics",
            sources=[SoundSource(azimuth_deg=90.0, elevation_deg=0.0, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
            snr_db=30.0,
            impairment=MicrophoneImpairment(dead_channels=dead_mics) if dead_mics else None,
        )
        localizer.reset_state()
        res = run_single_trial(localizer, generator, sc)
        errors.append(res.angular_error_deg)
        contrasts.append(res.peak_contrast)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig6_path = os.path.join(PLOTS_DIR, "fig06_microphone_fault_curves.png")
    bp.plot_microphone_fault_curves(dead_mic_counts, errors, contrasts, fig6_path)

    return {
        "dead_mic_counts": dead_mic_counts,
        "errors": errors,
        "contrasts": contrasts,
    }

if __name__ == "__main__":
    print("[Exp 07] Running microphone fault tolerance experiment...")
    res = run_microphone_faults_experiment()
    for c, e, ct in zip(res["dead_mic_counts"], res["errors"], res["contrasts"]):
        print(f"  {c} dead mics -> Err: {e:4.2f}°, Contrast: {ct:5.2f}")
