"""
benchmark.experiments.exp08_multipath
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

def run_multipath_experiment(generator=None, localizer=None) -> Dict[str, Any]:
    if generator is None: generator = SyntheticAudioGenerator()
    if localizer is None: localizer = BaselineProductionVariant()

    sirs_db = [20.0, 15.0, 10.0, 6.0, 3.0, 0.0]
    errors = []

    for sir in sirs_db:
        atten = 10.0 ** (-sir / 20.0)
        refl = ReflectionPath(
            azimuth_deg=75.0,
            elevation_deg=0.0,
            relative_amplitude=atten,
            delay_ms=1.5
        )
        sc = BenchmarkScenario(
            scenario_id=f"multipath_sir_{int(sir)}",
            category="multipath",
            description=f"SIR {sir} dB reflection",
            sources=[SoundSource(azimuth_deg=90.0, elevation_deg=0.0, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
            reflections=[refl],
            snr_db=30.0,
        )
        localizer.reset_state()
        res = run_single_trial(localizer, generator, sc)
        errors.append(res.angular_error_deg)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig7_path = os.path.join(PLOTS_DIR, "fig07_multipath_curves.png")
    bp.plot_multipath_curves(sirs_db, errors, fig7_path)

    return {
        "sirs_db": sirs_db,
        "errors": errors,
    }

if __name__ == "__main__":
    print("[Exp 08] Running multipath reflection experiment...")
    res = run_multipath_experiment()
    for s, e in zip(res["sirs_db"], res["errors"]):
        print(f"  SIR {s:4.1f} dB -> Err: {e:4.2f}°")
