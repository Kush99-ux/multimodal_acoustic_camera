"""
benchmark.core.registry
=======================
Persistent registry tracking golden baseline, benchmark runs, and variant catalog.
"""

from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from benchmark.config.benchmark_config import REGISTRY_FILE, BENCHMARK_VERSION


class BenchmarkRegistry:
    """Manages the benchmark registry file."""

    def __init__(self, registry_file: str = REGISTRY_FILE) -> None:
        self.registry_file = registry_file
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "version": BENCHMARK_VERSION,
            "created_at": datetime.now().isoformat(),
            "golden_baseline": None,
            "runs": [],
            "accepted_variants": [],
            "rejected_variants": [],
        }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def register_run(
        self,
        run_id: str,
        variant_name: str,
        suite_tier: str,
        metrics_summary: Dict[str, Any],
        is_baseline: bool = False,
    ) -> None:
        entry = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "variant": variant_name,
            "suite_tier": suite_tier,
            "is_baseline": is_baseline,
            "median_angular_error_deg": metrics_summary.get("angular_error", {}).get("median", 0.0),
            "p95_angular_error_deg": metrics_summary.get("angular_error", {}).get("p95", 0.0),
            "failure_rate_percent": metrics_summary.get("failure_rate_percent", 0.0),
            "median_processing_ms": metrics_summary.get("processing_time_ms", {}).get("median", 0.0),
        }
        self.data["runs"].append(entry)
        if is_baseline:
            self.data["golden_baseline"] = run_id
        self.save()

    def get_golden_baseline(self) -> Optional[str]:
        return self.data.get("golden_baseline")

    def list_runs(self) -> List[Dict[str, Any]]:
        return self.data.get("runs", [])
