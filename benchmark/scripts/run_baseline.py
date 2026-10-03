"""Helper script to run and establish Golden Production Baseline."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from benchmark.variants.baseline import BaselineProductionVariant
from benchmark.core.runner import BenchmarkRunner

if __name__ == "__main__":
    runner = BenchmarkRunner(variant=BaselineProductionVariant(), suite_tier="standard")
    runner.run(is_baseline=True)
