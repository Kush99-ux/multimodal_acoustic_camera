"""
benchmark.compare
=================
Dedicated CLI to compare any two benchmark runs or a candidate run against baseline.
Usage:
  python benchmark/compare.py --candidate <RUN_ID>
  python benchmark/compare.py --baseline <RUN_ID_1> --candidate <RUN_ID_2>
"""

from __future__ import annotations
import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
while script_dir in sys.path:
    sys.path.remove(script_dir)
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.core.comparison import BenchmarkComparator
from benchmark.config.benchmark_config import (
    BASELINE_DIR,
    RUNS_DIR,
    COMPARISONS_DIR,
)


def resolve_run_file(identifier: str, default_baseline: bool = False) -> str:
    if not identifier:
        if default_baseline:
            return os.path.join(BASELINE_DIR, "baseline_results.json")
        raise ValueError("Must provide run identifier.")

    if os.path.isfile(identifier):
        return identifier
    if os.path.isdir(identifier):
        p = os.path.join(identifier, "run_results.json")
        if os.path.isfile(p):
            return p

    p = os.path.join(RUNS_DIR, identifier, "run_results.json")
    if os.path.isfile(p):
        return p

    raise FileNotFoundError(f"Could not locate benchmark results for: {identifier}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Compare Benchmark Runs")
    parser.add_argument("--candidate", type=str, required=True, help="Candidate run ID or directory")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline run ID or directory (defaults to golden baseline)")
    args = parser.parse_args()

    cand_file = resolve_run_file(args.candidate)
    base_file = resolve_run_file(args.baseline, default_baseline=True)

    comparator = BenchmarkComparator.from_files(base_file, cand_file)
    report = comparator.generate_markdown_report()
    print("\n" + report)

    os.makedirs(COMPARISONS_DIR, exist_ok=True)
    out_name = f"comparison_{os.path.basename(os.path.dirname(cand_file))}.md"
    report_path = os.path.join(COMPARISONS_DIR, out_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved comparison report to: {report_path}")


if __name__ == "__main__":
    main()
