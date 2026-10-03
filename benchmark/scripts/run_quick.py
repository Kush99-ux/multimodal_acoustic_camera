"""Helper script to run rapid development verification (~15-30s)."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from benchmark.variants.baseline import BaselineProductionVariant
from benchmark.core.runner import BenchmarkRunner

if __name__ == "__main__":
    runner = BenchmarkRunner(variant=BaselineProductionVariant(), suite_tier="quick")
    runner.run(is_baseline=False)
