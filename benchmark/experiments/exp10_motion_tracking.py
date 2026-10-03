"""
benchmark.experiments.exp10_motion_tracking
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

def run_motion_tracking_experiment(generator=None) -> Dict[str, Any]:
    if generator is None: generator = SyntheticAudioGenerator()

    speeds_dps = [10.0, 20.0, 40.0, 60.0]
    frame_dt = 4096.0 / 48000.0  # 0.08533 s
    lag_errors = []

    for spd in speeds_dps:
        var = BaselineProductionVariant()
        var.reset_state()
        cur_az = 70.0
        trial_errors = []
        for f_idx in range(12):
            cur_az += spd * frame_dt
            sc = BenchmarkScenario(
                scenario_id=f"motion_{int(spd)}_{f_idx}",
                category="motion",
                description=f"Speed {spd} dps",
                sources=[SoundSource(azimuth_deg=cur_az, elevation_deg=0.0, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
                snr_db=30.0,
            )
            sig = generator.generate(sc)
            h, peak = var.localize(sig.data)
            est_az = float(peak["azimuth"])
            trial_errors.append(abs(cur_az - est_az))
        lag_errors.append(float(np.mean(trial_errors[3:])))

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig10_path = os.path.join(PLOTS_DIR, "fig10_motion_tracking.png")
    bp.plot_motion_tracking(speeds_dps, lag_errors, fig10_path)

    return {
        "speeds_dps": speeds_dps,
        "lag_errors": lag_errors,
    }

if __name__ == "__main__":
    print("[Exp 10] Running motion tracking experiment...")
    res = run_motion_tracking_experiment()
    for s, e in zip(res["speeds_dps"], res["lag_errors"]):
        print(f"  Speed {s:4.1f}°/s -> Tracking Lag: {e:4.2f}°")
