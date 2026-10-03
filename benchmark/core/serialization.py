"""
benchmark.core.serialization
============================
Structured export and import of benchmark results to JSON and CSV.
"""

from __future__ import annotations
import os
import json
import csv
from typing import Dict, Any, List
import numpy as np

from benchmark.models.result import TrialResult, ExperimentSummary, MetricDistribution


def _jsonify(obj: Any) -> Any:
    """Convert dataclasses and NumPy types to JSON-serializable primitives."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return {k: _jsonify(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def save_run_json(output_path: str, data: Dict[str, Any]) -> None:
    """Save run results to formatted JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(data), f, indent=2)


def save_trials_csv(output_path: str, trials: List[TrialResult]) -> None:
    """Export list of trial results to a CSV table."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not trials:
        return

    fieldnames = [
        "scenario_id",
        "trial_index",
        "true_azimuth_deg",
        "true_elevation_deg",
        "est_azimuth_deg",
        "est_elevation_deg",
        "azimuth_error_deg",
        "elevation_error_deg",
        "angular_error_deg",
        "peak_value",
        "peak_contrast",
        "processing_time_ms",
        "db_level",
        "valid",
        "passed_tolerance",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            row = {
                "scenario_id": t.scenario_id,
                "trial_index": t.trial_index,
                "true_azimuth_deg": f"{t.true_azimuth_deg:.2f}",
                "true_elevation_deg": f"{t.true_elevation_deg:.2f}",
                "est_azimuth_deg": f"{t.est_azimuth_deg:.2f}",
                "est_elevation_deg": f"{t.est_elevation_deg:.2f}",
                "azimuth_error_deg": f"{t.azimuth_error_deg:.3f}",
                "elevation_error_deg": f"{t.elevation_error_deg:.3f}",
                "angular_error_deg": f"{t.angular_error_deg:.3f}",
                "peak_value": f"{t.peak_value:.4f}",
                "peak_contrast": f"{t.peak_contrast:.4f}",
                "processing_time_ms": f"{t.processing_time_ms:.2f}",
                "db_level": f"{t.db_level:.1f}",
                "valid": t.valid,
                "passed_tolerance": t.passed_tolerance,
            }
            writer.writerow(row)


def load_run_json(file_path: str) -> Dict[str, Any]:
    """Load and parse run JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
