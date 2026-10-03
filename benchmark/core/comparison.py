"""
benchmark.core.comparison
=========================
Differential analysis engine comparing a candidate run against the golden baseline.
"""

from __future__ import annotations
import os
from typing import Dict, Any, List, Optional
import numpy as np

from benchmark.core.serialization import load_run_json
from benchmark.config.benchmark_config import COMPARISONS_DIR, BASELINE_DIR


class BenchmarkComparator:
    """
    Compares two benchmark runs across accuracy, robustness, and performance metrics.
    """

    def __init__(self, baseline_data: Dict[str, Any], candidate_data: Dict[str, Any]) -> None:
        self.baseline = baseline_data
        self.candidate = candidate_data

    @classmethod
    def from_files(cls, baseline_path: str, candidate_path: str) -> "BenchmarkComparator":
        b_data = load_run_json(baseline_path)
        c_data = load_run_json(candidate_path)
        return cls(b_data, c_data)

    def compute_diff(self) -> Dict[str, Any]:
        """
        Compute metric differentials: candidate - baseline.
        Negative delta in error or latency represents improvement.
        """
        b_sum = self.baseline.get("summary", {})
        c_sum = self.candidate.get("summary", {})

        diff = {
            "baseline_run_id": self.baseline.get("manifest", {}).get("run_id"),
            "baseline_variant": self.baseline.get("manifest", {}).get("variant"),
            "candidate_run_id": self.candidate.get("manifest", {}).get("run_id"),
            "candidate_variant": self.candidate.get("manifest", {}).get("variant"),
            "suite_tier": self.candidate.get("manifest", {}).get("suite_tier"),
            "metrics": {},
            "regressions": [],
            "improvements": [],
        }

        # Angular error comparison
        b_ang = b_sum.get("angular_error", {})
        c_ang = c_sum.get("angular_error", {})
        delta_median_err = c_ang.get("median", 0.0) - b_ang.get("median", 0.0)
        delta_p95_err = c_ang.get("p95", 0.0) - b_ang.get("p95", 0.0)

        # Failure rate
        b_fail = b_sum.get("failure_rate_percent", 0.0)
        c_fail = c_sum.get("failure_rate_percent", 0.0)
        delta_fail = c_fail - b_fail

        # Processing time
        b_time = b_sum.get("processing_time_ms", {})
        c_time = c_sum.get("processing_time_ms", {})
        delta_median_time = c_time.get("median", 0.0) - b_time.get("median", 0.0)

        # Peak contrast
        b_cont = b_sum.get("peak_contrast", {})
        c_cont = c_sum.get("peak_contrast", {})
        delta_contrast = c_cont.get("median", 0.0) - b_cont.get("median", 0.0)

        diff["metrics"] = {
            "angular_error_median_deg": {
                "baseline": b_ang.get("median", 0.0),
                "candidate": c_ang.get("median", 0.0),
                "delta": delta_median_err,
                "improved": delta_median_err < -0.05,
            },
            "angular_error_p95_deg": {
                "baseline": b_ang.get("p95", 0.0),
                "candidate": c_ang.get("p95", 0.0),
                "delta": delta_p95_err,
                "improved": delta_p95_err < -0.05,
            },
            "failure_rate_percent": {
                "baseline": b_fail,
                "candidate": c_fail,
                "delta": delta_fail,
                "improved": delta_fail < -0.01,
            },
            "median_processing_ms": {
                "baseline": b_time.get("median", 0.0),
                "candidate": c_time.get("median", 0.0),
                "delta": delta_median_time,
                "improved": delta_median_time < -0.5,
            },
            "peak_contrast_median": {
                "baseline": b_cont.get("median", 0.0),
                "candidate": c_cont.get("median", 0.0),
                "delta": delta_contrast,
                "improved": delta_contrast > 0.02,
            },
        }

        # Identify regressions
        if delta_median_err > 0.1:
            diff["regressions"].append(f"Median angular error regressed by +{delta_median_err:.2f} deg")
        if delta_p95_err > 0.2:
            diff["regressions"].append(f"P95 angular error regressed by +{delta_p95_err:.2f} deg")
        if delta_fail > 1.0:
            diff["regressions"].append(f"Failure rate increased by +{delta_fail:.1f}%")
        if delta_median_time > 2.0:
            diff["regressions"].append(f"Processing latency increased by +{delta_median_time:.1f} ms")

        # Identify improvements
        if delta_median_err < -0.1:
            diff["improvements"].append(f"Median angular error improved by {delta_median_err:.2f} deg")
        if delta_p95_err < -0.2:
            diff["improvements"].append(f"P95 angular error improved by {delta_p95_err:.2f} deg")
        if delta_fail < -1.0:
            diff["improvements"].append(f"Failure rate decreased by {delta_fail:.1f}%")
        if delta_median_time < -2.0:
            diff["improvements"].append(f"Processing latency reduced by {delta_median_time:.1f} ms")

        return diff

    def generate_markdown_report(self) -> str:
        """Render a comparative markdown table and tradeoff analysis."""
        diff = self.compute_diff()
        m = diff["metrics"]

        md = [
            f"# Benchmark Comparison Report: `{diff['candidate_variant']}` vs `{diff['baseline_variant']}`",
            f"**Candidate Run ID**: `{diff['candidate_run_id']}`  ",
            f"**Baseline Run ID**: `{diff['baseline_run_id']}`  ",
            f"**Suite Tier**: `{diff['suite_tier']}`  ",
            "",
            "## 1. Multi-Dimensional Performance Matrix",
            "",
            "| Metric | Baseline | Candidate | Delta | Evaluation |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| **Median Angular Error** | {m['angular_error_median_deg']['baseline']:.2f} deg | {m['angular_error_median_deg']['candidate']:.2f} deg | {m['angular_error_median_deg']['delta']:+.2f} deg | {'[IMPROVED]' if m['angular_error_median_deg']['improved'] else ('[REGRESSED]' if m['angular_error_median_deg']['delta'] > 0.1 else '[EQUIVALENT]')} |",
            f"| **P95 Angular Error** | {m['angular_error_p95_deg']['baseline']:.2f} deg | {m['angular_error_p95_deg']['candidate']:.2f} deg | {m['angular_error_p95_deg']['delta']:+.2f} deg | {'[IMPROVED]' if m['angular_error_p95_deg']['improved'] else ('[REGRESSED]' if m['angular_error_p95_deg']['delta'] > 0.2 else '[EQUIVALENT]')} |",
            f"| **Failure Rate** | {m['failure_rate_percent']['baseline']:.1f}% | {m['failure_rate_percent']['candidate']:.1f}% | {m['failure_rate_percent']['delta']:+.1f}% | {'[IMPROVED]' if m['failure_rate_percent']['improved'] else ('[REGRESSED]' if m['failure_rate_percent']['delta'] > 1.0 else '[EQUIVALENT]')} |",
            f"| **Median Processing Time** | {m['median_processing_ms']['baseline']:.1f} ms | {m['median_processing_ms']['candidate']:.1f} ms | {m['median_processing_ms']['delta']:+.1f} ms | {'[FASTER]' if m['median_processing_ms']['improved'] else ('[SLOWER]' if m['median_processing_ms']['delta'] > 2.0 else '[EQUIVALENT]')} |",
            f"| **Median Peak Contrast** | {m['peak_contrast_median']['baseline']:.2f} | {m['peak_contrast_median']['candidate']:.2f} | {m['peak_contrast_median']['delta']:+.2f} | {'[SHARPER]' if m['peak_contrast_median']['improved'] else ('[BLURRED]' if m['peak_contrast_median']['delta'] < -0.02 else '[EQUIVALENT]')} |",
            "",
            "## 2. Tradeoff Analysis",
        ]

        if diff["improvements"]:
            md.append("### Key Improvements:")
            for imp in diff["improvements"]:
                md.append(f"- [+] {imp}")
            md.append("")

        if diff["regressions"]:
            md.append("### Regressions & Tradeoffs:")
            for reg in diff["regressions"]:
                md.append(f"- [-] {reg}")
            md.append("")

        if not diff["improvements"] and not diff["regressions"]:
            md.append("No statistically significant differences detected between candidate and baseline.")
            md.append("")

        return "\n".join(md)
