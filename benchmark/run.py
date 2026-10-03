"""
benchmark.run
=============
Central command-line interface for the Synthetic Acoustic Benchmark Subsystem.
Usage:
  python benchmark/run.py --quick
  python benchmark/run.py --standard
  python benchmark/run.py --baseline
  python benchmark/run.py --standard --variant band_energy
  python benchmark/run.py --compare <RUN_ID>
"""

from __future__ import annotations
import os
import sys
import argparse

# Clean sys.path to prevent benchmark/config shadowing top-level config
script_dir = os.path.dirname(os.path.abspath(__file__))
while script_dir in sys.path:
    sys.path.remove(script_dir)
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.variants.baseline import BaselineProductionVariant
from benchmark.variants.csm_variants import CSMAlphaVariant, CSMSlidingWindowVariant
from benchmark.variants.band_weighting import BandWeightingVariant
from benchmark.variants.numerical_variants import PrecisionVariant
from benchmark.core.runner import BenchmarkRunner
from benchmark.core.comparison import BenchmarkComparator
from benchmark.config.benchmark_config import (
    BASELINE_DIR,
    RUNS_DIR,
    COMPARISONS_DIR,
)
from benchmark.core.registry import BenchmarkRegistry


def get_variant_by_name(name: str):
    """Instantiate variant by string name."""
    name_clean = name.lower().strip()
    if name_clean in ["baseline", "baseline_production", "production"]:
        return BaselineProductionVariant()
    elif name_clean in ["csm_ema_fast", "ema_fast", "csm_fast"]:
        return CSMAlphaVariant(alpha=0.60, name_suffix="fast_0.60")
    elif name_clean in ["csm_ema_slow", "ema_slow", "csm_slow"]:
        return CSMAlphaVariant(alpha=0.10, name_suffix="slow_0.10")
    elif name_clean in ["csm_no_ema", "no_ema"]:
        return CSMAlphaVariant(alpha=1.0, name_suffix="none_1.0")
    elif name_clean in ["csm_sliding_window", "sliding_window"]:
        return CSMSlidingWindowVariant(window_len=3)
    elif name_clean in ["band_energy", "energy_weighted"]:
        return BandWeightingVariant(strategy="energy")
    elif name_clean in ["band_raw", "raw_accumulation"]:
        return BandWeightingVariant(strategy="raw")
    elif name_clean in ["band_dynamic_range", "dynamic_range"]:
        return BandWeightingVariant(strategy="dynamic_range")
    elif name_clean in ["band_active_gated", "active_gated"]:
        return BandWeightingVariant(strategy="active_gated")
    elif name_clean in ["float32", "precision_float32"]:
        return PrecisionVariant(use_float32=True)
    elif name_clean in ["float64", "precision_float64"]:
        return PrecisionVariant(use_float32=False)
    else:
        raise ValueError(f"Unknown algorithm variant: '{name}'. Available: baseline, csm_ema_fast, csm_ema_slow, csm_no_ema, csm_sliding_window, band_energy, band_raw, band_dynamic_range, band_active_gated, float32")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Acoustic Benchmark Subsystem")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Run rapid regression suite (~15-30s)")
    group.add_argument("--standard", action="store_true", help="Run standard comprehensive research suite")
    group.add_argument("--full", action="store_true", help="Run full exhaustive research suite")
    group.add_argument("--baseline", action="store_true", help="Run and establish Golden Production Baseline")
    group.add_argument("--compare", type=str, metavar="RUN_ID", help="Compare a candidate run ID against golden baseline")

    parser.add_argument("--variant", type=str, default="baseline", help="Algorithm variant (default: baseline)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    # Compare mode
    if args.compare:
        cand_id = args.compare
        cand_dir = os.path.join(RUNS_DIR, cand_id) if not os.path.isdir(cand_id) else cand_id
        cand_file = os.path.join(cand_dir, "run_results.json")
        base_file = os.path.join(BASELINE_DIR, "baseline_results.json")

        if not os.path.exists(cand_file):
            print(f"Error: Candidate results not found at: {cand_file}")
            sys.exit(1)
        if not os.path.exists(base_file):
            print(f"Error: Golden baseline not found at: {base_file}. Run with --baseline first.")
            sys.exit(1)

        comparator = BenchmarkComparator.from_files(base_file, cand_file)
        report = comparator.generate_markdown_report()
        print("\n" + report)

        comp_dir = COMPARISONS_DIR
        os.makedirs(comp_dir, exist_ok=True)
        report_file = os.path.join(comp_dir, f"comparison_{cand_id}.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[Comparison Report Saved] {report_file}")
        return

    # Suite tier selection
    is_baseline = args.baseline
    if is_baseline:
        suite_tier = "standard"
        variant = BaselineProductionVariant()
    elif args.quick:
        suite_tier = "quick"
        variant = get_variant_by_name(args.variant)
    elif args.full:
        suite_tier = "full"
        variant = get_variant_by_name(args.variant)
    else:  # default or --standard
        suite_tier = "standard"
        variant = get_variant_by_name(args.variant)

    runner = BenchmarkRunner(
        variant=variant,
        suite_tier=suite_tier,
        output_dir=args.output_dir,
    )

    runner.run(is_baseline=is_baseline)


if __name__ == "__main__":
    main()
