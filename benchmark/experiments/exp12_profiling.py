"""
benchmark.experiments.exp12_profiling
"""
from __future__ import annotations
import os, sys
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

from benchmark.core.timing import AcousticProfiler
from benchmark.variants.csm_variants import CSMAlphaVariant
from benchmark.variants.band_weighting import BandWeightingVariant
from benchmark.variants.numerical_variants import PrecisionVariant

def run_profiling_experiment(generator=None) -> Dict[str, Any]:
    if generator is None: generator = SyntheticAudioGenerator()

    sc = BenchmarkScenario(
        scenario_id="profile_frame", category="profile", description="Profiling frame",
        sources=[SoundSource(azimuth_deg=90.0, elevation_deg=0.0, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
        snr_db=30.0
    )
    sig = generator.generate(sc)

    # Stage breakdown on Baseline
    profiler = AcousticProfiler()
    var = BaselineProductionVariant()
    var.reset_state()

    # Warmup
    for _ in range(5):
        var.localize(sig.data)

    for _ in range(30):
        with profiler.time_stage("STFT Analysis"):
            import scipy.signal
            _ = scipy.signal.stft(sig.data.T, fs=48000, nperseg=512)
        with profiler.time_stage("CSM Computation & EMA"):
            csm = np.cov(sig.data.T)
        with profiler.time_stage("Eigendecomposition"):
            _ = np.linalg.eigh(csm)
        with profiler.time_stage("MUSIC Steering Projection"):
            _, _ = var.localize(sig.data)

    stages = ["STFT Analysis", "CSM Computation & EMA", "Eigendecomposition", "MUSIC Steering Projection"]
    summary = profiler.get_summary()
    stage_durations = [summary[s]["median_ms"] for s in stages]

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig8_path = os.path.join(PLOTS_DIR, "fig08_profiling_breakdown.png")
    bp.plot_profiling_breakdown(stages, stage_durations, fig8_path)

    # Timing across variants
    variants = [
        ("Baseline (f64)", BaselineProductionVariant()),
        ("CSM Fast (a=0.6)", CSMAlphaVariant(alpha=0.6)),
        ("CSM Slow (a=0.1)", CSMAlphaVariant(alpha=0.1)),
        ("Band Energy", BandWeightingVariant(strategy="energy")),
        ("Precision f32", PrecisionVariant(use_float32=True)),
    ]
    var_names = []
    med_times = []
    p95_times = []

    for vname, vinst in variants:
        vinst.reset_state()
        times = []
        for _ in range(25):
            t0 = time.perf_counter()
            vinst.localize(sig.data)
            times.append((time.perf_counter() - t0) * 1000.0)
        var_names.append(vname)
        med_times.append(float(np.median(times)))
        p95_times.append(float(np.percentile(times, 95)))

    fig14_path = os.path.join(PLOTS_DIR, "fig14_variant_timing_comparison.png")
    bp.plot_variant_timing_comparison(var_names, med_times, p95_times, fig14_path)

    return {
        "stages": stages,
        "stage_durations_ms": stage_durations,
        "variant_names": var_names,
        "median_times_ms": med_times,
        "p95_times_ms": p95_times,
    }

import time
if __name__ == "__main__":
    print("[Exp 12] Running latency profiling experiment...")
    res = run_profiling_experiment()
    for s, d in zip(res["stages"], res["stage_durations_ms"]):
        print(f"  {s:30s} -> {d:.2f} ms")
