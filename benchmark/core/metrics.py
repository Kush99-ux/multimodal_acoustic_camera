"""
benchmark.core.metrics
======================
Comprehensive evaluation metrics for acoustic localization benchmarks.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from benchmark.models.result import TrialResult, MetricDistribution, ExperimentSummary
from benchmark.config.benchmark_config import (
    TOLERANCE_EXCELLENT_DEG,
    TOLERANCE_PASS_DEG,
    TOLERANCE_FAILURE_DEG,
)


def calculate_angular_error(
    true_az_deg: float,
    true_el_deg: float,
    est_az_deg: float,
    est_el_deg: float,
) -> float:
    """
    Calculate 3D spherical great-circle angular error between true and estimated directions.
    """
    # Unit vectors
    t_az, t_el = np.deg2rad(true_az_deg), np.deg2rad(true_el_deg)
    e_az, e_el = np.deg2rad(est_az_deg), np.deg2rad(est_el_deg)

    u_true = np.array([np.cos(t_el) * np.cos(t_az), np.cos(t_el) * np.sin(t_az), np.sin(t_el)])
    u_est = np.array([np.cos(e_el) * np.cos(e_az), np.cos(e_el) * np.sin(e_az), np.sin(e_el)])

    cos_angle = np.clip(np.dot(u_true, u_est), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def calculate_peak_contrast(heatmap: np.ndarray) -> float:
    """
    Calculate contrast ratio of the peak relative to median background:
    Contrast = (P_peak - P_median) / (P_peak + 1e-9)
    """
    p_peak = float(np.max(heatmap))
    p_median = float(np.median(heatmap))
    if p_peak < 1e-9:
        return 0.0
    return float(np.clip((p_peak - p_median) / p_peak, 0.0, 1.0))


def evaluate_trial(
    scenario_id: str,
    trial_index: int,
    true_az: float,
    true_el: float,
    heatmap: np.ndarray,
    peak_dict: Dict[str, Any],
    processing_time_ms: float,
    metadata: Dict[str, Any] = None,
) -> TrialResult:
    """
    Evaluate single trial results against ground truth.
    """
    est_az = float(peak_dict["azimuth"])
    est_el = float(peak_dict["elevation"])
    peak_val = float(peak_dict.get("value", 0.0))
    db_level = float(peak_dict.get("db", 0.0))
    valid = bool(peak_dict.get("valid", True))

    az_err = abs(est_az - true_az)
    el_err = abs(est_el - true_el)
    ang_err = calculate_angular_error(true_az, true_el, est_az, est_el)
    contrast = calculate_peak_contrast(heatmap)

    passed = (ang_err <= TOLERANCE_PASS_DEG) and valid

    return TrialResult(
        scenario_id=scenario_id,
        trial_index=trial_index,
        true_azimuth_deg=true_az,
        true_elevation_deg=true_el,
        est_azimuth_deg=est_az,
        est_elevation_deg=est_el,
        azimuth_error_deg=az_err,
        elevation_error_deg=el_err,
        angular_error_deg=ang_err,
        peak_value=peak_val,
        peak_contrast=contrast,
        processing_time_ms=processing_time_ms,
        db_level=db_level,
        valid=valid,
        passed_tolerance=passed,
        metadata=metadata or {},
    )


def summarize_trials(
    experiment_name: str,
    trials: List[TrialResult],
) -> ExperimentSummary:
    """
    Summarize a collection of trial results into aggregated statistical distributions.
    """
    total = len(trials)
    if total == 0:
        empty_dist = MetricDistribution(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return ExperimentSummary(
            experiment_name=experiment_name,
            total_trials=0,
            successful_trials=0,
            failure_rate_percent=0.0,
            angular_error=empty_dist,
            azimuth_error=empty_dist,
            elevation_error=empty_dist,
            peak_contrast=empty_dist,
            processing_time_ms=empty_dist,
            trials=[],
        )

    success_count = sum(1 for t in trials if t.passed_tolerance)
    fail_rate = 100.0 * (total - success_count) / float(total)

    ang_errors = [t.angular_error_deg for t in trials]
    az_errors = [t.azimuth_error_deg for t in trials]
    el_errors = [t.elevation_error_deg for t in trials]
    contrasts = [t.peak_contrast for t in trials]
    timings = [t.processing_time_ms for t in trials]

    return ExperimentSummary(
        experiment_name=experiment_name,
        total_trials=total,
        successful_trials=success_count,
        failure_rate_percent=fail_rate,
        angular_error=MetricDistribution.from_values(ang_errors),
        azimuth_error=MetricDistribution.from_values(az_errors),
        elevation_error=MetricDistribution.from_values(el_errors),
        peak_contrast=MetricDistribution.from_values(contrasts),
        processing_time_ms=MetricDistribution.from_values(timings),
        trials=trials,
    )
