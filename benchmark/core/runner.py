"""
benchmark.core.runner
=====================
Orchestration engine executing benchmark suites on algorithm variants.
"""

from __future__ import annotations
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

from benchmark.models.scenario import BenchmarkScenario
from benchmark.models.result import TrialResult, ExperimentSummary
from benchmark.variants.base_variant import AcousticAlgorithmVariant
from benchmark.core.generator import SyntheticAudioGenerator
from benchmark.core.metrics import evaluate_trial, summarize_trials
from benchmark.core.serialization import save_run_json, save_trials_csv
from benchmark.core.registry import BenchmarkRegistry
from benchmark.config.benchmark_config import (
    BENCHMARK_VERSION,
    RUNS_DIR,
    BASELINE_DIR,
)
from benchmark.config.scenarios import get_quick_scenarios, get_standard_scenarios


class BenchmarkRunner:
    """
    Executes benchmark suites against an acoustic localizer variant.
    """

    def __init__(
        self,
        variant: AcousticAlgorithmVariant,
        suite_tier: str = "quick",
        output_dir: Optional[str] = None,
    ) -> None:
        self.variant = variant
        self.suite_tier = suite_tier
        self.generator = SyntheticAudioGenerator()
        self.registry = BenchmarkRegistry()

        timestamp_str = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.run_id = f"{variant.name}_{suite_tier}_{timestamp_str}"
        self.output_dir = output_dir or os.path.join(RUNS_DIR, self.run_id)

    def load_scenarios(self) -> List[BenchmarkScenario]:
        if self.suite_tier == "quick":
            return get_quick_scenarios()
        elif self.suite_tier in ["standard", "full"]:
            return get_standard_scenarios()
        else:
            raise ValueError(f"Unknown suite tier: {self.suite_tier}")

    def run(self, is_baseline: bool = False) -> Dict[str, Any]:
        """
        Execute all scenarios in the suite and compile summary results.
        """
        print(f"\n========================================================")
        print(f"  ACOUSTIC BENCHMARK RUN: {self.run_id}")
        print(f"  Variant:    {self.variant.name}")
        print(f"  Suite Tier: {self.suite_tier}")
        print(f"  Baseline:   {is_baseline}")
        print(f"========================================================")

        scenarios = self.load_scenarios()
        print(f"[Benchmark] Loaded {len(scenarios)} test scenarios.")

        all_trials: List[TrialResult] = []
        category_trials: Dict[str, List[TrialResult]] = {}

        t_run_start = time.perf_counter()

        for idx, scenario in enumerate(scenarios, 1):
            # Generate synthetic 16-channel signal
            signal = self.generator.generate(scenario)

            # Reset algorithm temporal state before independent trial
            self.variant.reset_state()

            # Execute localization
            t_start = time.perf_counter()
            heatmap, peak_dict = self.variant.localize(signal.data)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            # Evaluate against primary source ground truth
            true_az = scenario.sources[0].azimuth_deg if scenario.sources else 90.0
            true_el = scenario.sources[0].elevation_deg if scenario.sources else 0.0

            trial = evaluate_trial(
                scenario_id=scenario.scenario_id,
                trial_index=idx,
                true_az=true_az,
                true_el=true_el,
                heatmap=heatmap,
                peak_dict=peak_dict,
                processing_time_ms=elapsed_ms,
                metadata={
                    "category": scenario.category,
                    "snr_db": scenario.snr_db,
                    "noise_type": scenario.noise_type,
                    "waveform": scenario.sources[0].waveform.value if scenario.sources else "unknown",
                    "frequency_hz": scenario.sources[0].frequency_hz if scenario.sources else 1000.0,
                },
            )

            all_trials.append(trial)
            if scenario.category not in category_trials:
                category_trials[scenario.category] = []
            category_trials[scenario.category].append(trial)

            status = "PASS" if trial.passed_tolerance else "FAIL"
            print(f"  [{idx:02d}/{len(scenarios):02d}] {scenario.scenario_id:<32} -> Err: {trial.angular_error_deg:5.2f} deg | Time: {trial.processing_time_ms:5.1f} ms | [{status}]")

        total_run_time_s = time.perf_counter() - t_run_start

        # Aggregate summary
        overall_summary = summarize_trials("overall", all_trials)

        category_summaries = {}
        for cat, c_trials in category_trials.items():
            category_summaries[cat] = summarize_trials(cat, c_trials)

        manifest = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "benchmark_version": BENCHMARK_VERSION,
            "variant": self.variant.name,
            "variant_description": self.variant.description,
            "parameters": self.variant.parameters,
            "suite_tier": self.suite_tier,
            "total_scenarios": len(scenarios),
            "total_run_time_seconds": total_run_time_s,
            "is_baseline": is_baseline,
        }

        run_output = {
            "manifest": manifest,
            "summary": overall_summary,
            "category_summaries": category_summaries,
            "trials": all_trials,
        }

        # Save results in run directory
        save_run_json(os.path.join(self.output_dir, "run_manifest.json"), manifest)
        save_run_json(os.path.join(self.output_dir, "run_results.json"), run_output)
        save_trials_csv(os.path.join(self.output_dir, "trials.csv"), all_trials)

        # If designated as golden baseline, also write to baseline directory
        if is_baseline:
            os.makedirs(BASELINE_DIR, exist_ok=True)
            save_run_json(os.path.join(BASELINE_DIR, "baseline_metadata.json"), manifest)
            save_run_json(os.path.join(BASELINE_DIR, "baseline_results.json"), run_output)
            save_run_json(os.path.join(BASELINE_DIR, "baseline_summary.json"), overall_summary)
            save_trials_csv(os.path.join(BASELINE_DIR, "baseline_metrics.csv"), all_trials)

        from dataclasses import asdict
        self.registry.register_run(
            run_id=self.run_id,
            variant_name=self.variant.name,
            suite_tier=self.suite_tier,
            metrics_summary=asdict(overall_summary),
            is_baseline=is_baseline,
        )

        print(f"\n[Benchmark Complete] Run saved to: {self.output_dir}")
        print(f"  Median Angular Error: {overall_summary.angular_error.median:.2f} deg")
        print(f"  P95 Angular Error:    {overall_summary.angular_error.p95:.2f} deg")
        print(f"  Max Angular Error:    {overall_summary.angular_error.max_val:.2f} deg")
        print(f"  Failure Rate:         {overall_summary.failure_rate_percent:.1f}%")
        print(f"  Median Processing:    {overall_summary.processing_time_ms.median:.1f} ms")

        return run_output
