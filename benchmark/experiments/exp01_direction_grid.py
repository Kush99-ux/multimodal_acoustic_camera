"""
benchmark.experiments.exp01_direction_grid
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

def run_direction_grid_experiment(generator=None, localizer=None) -> Dict[str, Any]:
    if generator is None: generator = SyntheticAudioGenerator()
    if localizer is None: localizer = BaselineProductionVariant()

    az_vals = np.arange(65.0, 116.0, 5.0)
    el_vals = np.arange(-15.0, 16.0, 5.0)
    error_matrix = np.zeros((len(el_vals), len(az_vals)), dtype=np.float64)
    az_biases = []
    el_biases = []
    trials = []

    for r_idx, el in enumerate(el_vals):
        for c_idx, az in enumerate(az_vals):
            sc = BenchmarkScenario(
                scenario_id=f"grid_az{int(az)}_el{int(el)}",
                category="spatial_grid",
                description=f"Grid Az={az}, El={el}",
                sources=[SoundSource(azimuth_deg=az, elevation_deg=el, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
                snr_db=30.0,
            )
            localizer.reset_state()
            res = run_single_trial(localizer, generator, sc, trial_index=r_idx*len(az_vals)+c_idx)
            trials.append(res)
            error_matrix[r_idx, c_idx] = res.angular_error_deg
            az_biases.append(res.azimuth_error_deg)
            el_biases.append(res.elevation_error_deg)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig1_path = os.path.join(PLOTS_DIR, "fig01_spatial_accuracy_heatmap.png")
    bp.plot_spatial_accuracy_heatmap(az_vals, el_vals, error_matrix, fig1_path)

    mean_el_biases = np.mean(error_matrix, axis=1).tolist()
    fig13_path = os.path.join(PLOTS_DIR, "fig13_elevation_bias_distribution.png")
    bp.plot_elevation_bias_distribution(el_vals.tolist(), mean_el_biases, fig13_path)

    return {
        "az_vals": az_vals.tolist(),
        "el_vals": el_vals.tolist(),
        "error_matrix": error_matrix.tolist(),
        "mean_error_deg": float(np.mean(error_matrix)),
        "median_error_deg": float(np.median(error_matrix)),
        "max_error_deg": float(np.max(error_matrix)),
        "rms_error_deg": float(np.sqrt(np.mean(error_matrix**2))),
        "az_bias_mean": float(np.mean(az_biases)),
        "el_bias_mean": float(np.mean(el_biases)),
    }

if __name__ == "__main__":
    print("[Exp 01] Running spatial direction grid experiment...")
    res = run_direction_grid_experiment()
    print(f"  Mean Error: {res['mean_error_deg']:.2f}°, Max: {res['max_error_deg']:.2f}°, RMS: {res['rms_error_deg']:.2f}°")
