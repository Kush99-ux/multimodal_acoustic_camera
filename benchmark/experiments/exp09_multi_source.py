"""
benchmark.experiments.exp09_multi_source
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

def run_multi_source_experiment(generator=None, localizer=None) -> Dict[str, Any]:
    if generator is None: generator = SyntheticAudioGenerator()
    if localizer is None: localizer = BaselineProductionVariant()

    separations = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0]
    resolved_rates = []

    for sep in separations:
        # Two sources placed symmetrically around 90 deg
        s1 = SoundSource(azimuth_deg=90.0 - sep/2, elevation_deg=0.0, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)
        s2 = SoundSource(azimuth_deg=90.0 + sep/2, elevation_deg=0.0, waveform=WaveformType.MULTI_TONE, frequency_hz=2500.0)
        sc = BenchmarkScenario(
            scenario_id=f"multi_src_sep_{int(sep)}",
            category="multi_source",
            description=f"Separation {sep} deg",
            sources=[s1, s2],
            snr_db=30.0,
        )
        sig = generator.generate(sc)
        localizer.reset_state()
        heatmap, peak = localizer.localize(sig.data)
        
        # Check if 2 local maxima can be resolved along azimuth line el=0
        mid_row = heatmap.shape[0] // 2
        line = heatmap[mid_row, :]
        from scipy.signal import find_peaks
        peaks_idx, props = find_peaks(line, height=0.5, distance=3)
        resolved = 1.0 if len(peaks_idx) >= 2 else 0.0
        resolved_rates.append(resolved)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig9_path = os.path.join(PLOTS_DIR, "fig09_multi_source_resolution.png")
    bp.plot_multi_source_resolution(separations, resolved_rates, fig9_path)

    return {
        "separations": separations,
        "resolved_rates": resolved_rates,
    }

if __name__ == "__main__":
    print("[Exp 09] Running multi-source resolution experiment...")
    res = run_multi_source_experiment()
    for s, r in zip(res["separations"], res["resolved_rates"]):
        print(f"  Separation {s:4.1f}° -> Resolved: {r > 0.5}")
