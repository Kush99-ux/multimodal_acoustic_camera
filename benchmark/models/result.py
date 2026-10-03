"""
benchmark.models.result
=======================
Evaluation results and summary metric representations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class TrialResult:
    """Evaluation result for one individual synthetic trial."""
    scenario_id: str
    trial_index: int
    true_azimuth_deg: float
    true_elevation_deg: float
    est_azimuth_deg: float
    est_elevation_deg: float
    azimuth_error_deg: float
    elevation_error_deg: float
    angular_error_deg: float
    peak_value: float
    peak_contrast: float
    processing_time_ms: float
    db_level: float
    valid: bool
    passed_tolerance: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricDistribution:
    """Statistical summary of a metric."""
    mean: float
    median: float
    std: float
    p95: float
    min_val: float
    max_val: float

    @classmethod
    def from_values(cls, values: List[float] | np.ndarray) -> MetricDistribution:
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cls(
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr)),
            p95=float(np.percentile(arr, 95)),
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
        )


@dataclass
class ExperimentSummary:
    """Summary metrics for one complete experiment or benchmark suite."""
    experiment_name: str
    total_trials: int
    successful_trials: int
    failure_rate_percent: float
    angular_error: MetricDistribution
    azimuth_error: MetricDistribution
    elevation_error: MetricDistribution
    peak_contrast: MetricDistribution
    processing_time_ms: MetricDistribution
    trials: List[TrialResult] = field(default_factory=list)
