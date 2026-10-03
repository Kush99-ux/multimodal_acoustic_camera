"""
benchmark.experiments.exp03_snr_sweep
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

def run_snr_sweep_experiment(generator=None, localizer=None) -> Dict[str, Any]:
    if generator is None: generator = SyntheticAudioGenerator()
    if localizer is None: localizer = BaselineProductionVariant()

    snr_levels = [30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0, -5.0]
    white_errors = []
    pink_errors = []
    contrasts = []

    for snr in snr_levels:
        sc_w = BenchmarkScenario(
            scenario_id=f"snr_white_{int(snr)}",
            category="snr_white",
            description=f"White noise SNR {snr} dB",
            sources=[SoundSource(azimuth_deg=103.0, elevation_deg=-5.0, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
            snr_db=snr,
            noise_type="white",
        )
        localizer.reset_state()
        rw = run_single_trial(localizer, generator, sc_w)
        white_errors.append(rw.angular_error_deg)
        contrasts.append(rw.peak_contrast)

        sc_p = BenchmarkScenario(
            scenario_id=f"snr_pink_{int(snr)}",
            category="snr_pink",
            description=f"Pink noise SNR {snr} dB",
            sources=[SoundSource(azimuth_deg=103.0, elevation_deg=-5.0, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
            snr_db=snr,
            noise_type="pink",
        )
        localizer.reset_state()
        rp = run_single_trial(localizer, generator, sc_p)
        pink_errors.append(rp.angular_error_deg)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig2_path = os.path.join(PLOTS_DIR, "fig02_snr_curves.png")
    bp.plot_snr_curves(snr_levels, white_errors, pink_errors, fig2_path)

    fig12_path = os.path.join(PLOTS_DIR, "fig12_contrast_vs_snr.png")
    bp.plot_contrast_vs_snr(snr_levels, contrasts, fig12_path)

    return {
        "snr_levels": snr_levels,
        "white_errors": white_errors,
        "pink_errors": pink_errors,
        "contrasts": contrasts,
    }

if __name__ == "__main__":
    print("[Exp 03] Running SNR sweep experiment...")
    res = run_snr_sweep_experiment()
    print("Done. White errors:", [f"{e:.1f}°" for e in res["white_errors"]])
