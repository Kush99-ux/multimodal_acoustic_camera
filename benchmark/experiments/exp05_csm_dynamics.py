"""
benchmark.experiments.exp05_csm_dynamics
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

from benchmark.variants.csm_variants import CSMAlphaVariant

def run_csm_dynamics_experiment(generator=None) -> Dict[str, Any]:
    if generator is None: generator = SyntheticAudioGenerator()

    alphas = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0]
    static_noise_errors = []
    step_lags = []

    sig_init = generator.generate(BenchmarkScenario(
        scenario_id="csm_step_1", category="csm", description="Step initial",
        sources=[SoundSource(azimuth_deg=75.0, elevation_deg=0.0, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
        snr_db=25.0
    ))
    sig_jump = generator.generate(BenchmarkScenario(
        scenario_id="csm_step_2", category="csm", description="Step jump",
        sources=[SoundSource(azimuth_deg=105.0, elevation_deg=0.0, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
        snr_db=25.0
    ))
    sig_noisy = generator.generate(BenchmarkScenario(
        scenario_id="csm_noisy", category="csm", description="Noisy static",
        sources=[SoundSource(azimuth_deg=90.0, elevation_deg=0.0, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
        snr_db=5.0
    ))

    for a in alphas:
        var = CSMAlphaVariant(alpha=a)
        var.reset_state()
        for _ in range(5):
            h, peak = var.localize(sig_noisy.data)
        err_static = calculate_angular_error(90.0, 0.0, float(peak["azimuth"]), float(peak["elevation"]))
        static_noise_errors.append(err_static)

        var.reset_state()
        for _ in range(5):
            var.localize(sig_init.data)
        lag_frames = 0
        for step_i in range(1, 15):
            h, peak = var.localize(sig_jump.data)
            az_now = float(peak["azimuth"])
            if abs(az_now - 105.0) <= 2.0:
                lag_frames = step_i
                break
        if lag_frames == 0:
            lag_frames = 15
        step_lags.append(float(lag_frames))

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig4_path = os.path.join(PLOTS_DIR, "fig04_csm_alpha_tradeoff.png")
    bp.plot_csm_alpha_tradeoff(alphas, static_noise_errors, step_lags, fig4_path)

    return {
        "alphas": alphas,
        "static_noise_errors": static_noise_errors,
        "step_lags": step_lags,
    }

if __name__ == "__main__":
    print("[Exp 05] Running CSM dynamics & alpha tradeoff experiment...")
    res = run_csm_dynamics_experiment()
    for a, e, l in zip(res["alphas"], res["static_noise_errors"], res["step_lags"]):
        print(f"  Alpha {a:4.2f} -> Noisy Error: {e:4.2f}°, Step Lag: {l:2.0f} frames")
