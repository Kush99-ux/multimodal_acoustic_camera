"""
benchmark.experiments.exp11_queue_behavior
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

def run_queue_behavior_experiment() -> Dict[str, Any]:
    # Simulates audio frame consumption when consumer is slower than producer
    # Producer: 48000 Hz, 4096 samples = 85.3 ms per frame
    # Slow Consumer: 150 ms per loop iteration (e.g. CPU YOLO)
    n_frames = 20
    audio_period_ms = 85.33
    consumer_period_ms = 150.0

    frame_indices = list(range(1, n_frames + 1))
    fifo_latencies_ms = []
    latest_latencies_ms = []

    # In FIFO: backlog accumulates by (consumer_period - audio_period) each step
    fifo_backlog_ms = 0.0
    for idx in frame_indices:
        fifo_backlog_ms += (consumer_period_ms - audio_period_ms)
        fifo_latencies_ms.append(fifo_backlog_ms)
        # Latest-frame: backlog is drained, age is just the current frame latency ~85.3 ms
        latest_latencies_ms.append(audio_period_ms)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig11_path = os.path.join(PLOTS_DIR, "fig11_queue_behavior.png")
    bp.plot_queue_behavior(frame_indices, fifo_latencies_ms, latest_latencies_ms, fig11_path)

    return {
        "frame_indices": frame_indices,
        "fifo_latencies_ms": fifo_latencies_ms,
        "latest_latencies_ms": latest_latencies_ms,
    }

if __name__ == "__main__":
    print("[Exp 11] Running queue behavior comparison experiment...")
    res = run_queue_behavior_experiment()
    print(f"  Final Frame FIFO Latency: {res['fifo_latencies_ms'][-1]:.1f} ms vs Latest-Frame: {res['latest_latencies_ms'][-1]:.1f} ms")
