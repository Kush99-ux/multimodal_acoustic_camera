"""
benchmark.core.timing
=====================
High-resolution execution timer and micro-profiler for acoustic localization stages.
"""

from __future__ import annotations
import time
from typing import Dict, Any, List
import numpy as np


class StageTimer:
    """Context manager for timing pipeline stages."""

    def __init__(self, stage_name: str, profiler: "AcousticProfiler") -> None:
        self.stage_name = stage_name
        self.profiler = profiler
        self.t_start = 0.0

    def __enter__(self) -> "StageTimer":
        self.t_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_ms = (time.perf_counter() - self.t_start) * 1000.0
        self.profiler.record_stage(self.stage_name, elapsed_ms)


class AcousticProfiler:
    """
    Records detailed execution timings across acoustic localization stages.
    """

    def __init__(self) -> None:
        self.stage_timings: Dict[str, List[float]] = {
            "stft_window": [],
            "csm_computation": [],
            "csm_smoothing": [],
            "eigendecomposition": [],
            "noise_subspace_projection": [],
            "spectrum_inversion": [],
            "normalization_accumulation": [],
            "sigmoid_contrast": [],
            "peak_extraction": [],
            "full_localization": [],
        }

    def record_stage(self, stage: str, duration_ms: float) -> None:
        if stage not in self.stage_timings:
            self.stage_timings[stage] = []
        self.stage_timings[stage].append(duration_ms)

    def time_stage(self, stage: str) -> StageTimer:
        return StageTimer(stage, self)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for stage, times in self.stage_timings.items():
            if not times:
                continue
            arr = np.asarray(times)
            summary[stage] = {
                "mean_ms": float(np.mean(arr)),
                "median_ms": float(np.median(arr)),
                "std_ms": float(np.std(arr)),
                "p95_ms": float(np.percentile(arr, 95)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
            }
        return summary
