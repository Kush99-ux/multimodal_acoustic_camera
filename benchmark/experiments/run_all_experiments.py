"""
benchmark.experiments.run_all_experiments
=========================================
Executes all 12 acoustic research experiments, aggregates metrics, and generates
all 14 publication-grade figures.
"""
from __future__ import annotations
import os, sys, json
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
while script_dir in sys.path:
    sys.path.remove(script_dir)
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.experiments.exp01_direction_grid import run_direction_grid_experiment
from benchmark.experiments.exp02_frequency_sweep import run_frequency_sweep_experiment
from benchmark.experiments.exp03_snr_sweep import run_snr_sweep_experiment
from benchmark.experiments.exp04_waveforms import run_waveform_experiment
from benchmark.experiments.exp05_csm_dynamics import run_csm_dynamics_experiment
from benchmark.experiments.exp06_band_weighting import run_band_weighting_experiment
from benchmark.experiments.exp07_microphone_faults import run_microphone_faults_experiment
from benchmark.experiments.exp08_multipath import run_multipath_experiment
from benchmark.experiments.exp09_multi_source import run_multi_source_experiment
from benchmark.experiments.exp10_motion_tracking import run_motion_tracking_experiment
from benchmark.experiments.exp11_queue_behavior import run_queue_behavior_experiment
from benchmark.experiments.exp12_profiling import run_profiling_experiment

def main():
    print("=" * 60)
    print("  RUNNING ALL 12 BENCHMARK EXPERIMENTS & GENERATING PLOTS")
    print("=" * 60)
    all_results = {"timestamp": datetime.now().isoformat()}

    print("\n[1/12] Running Direction Grid Experiment...")
    all_results["exp01"] = run_direction_grid_experiment()

    print("\n[2/12] Running Frequency Sweep Experiment...")
    all_results["exp02"] = run_frequency_sweep_experiment()

    print("\n[3/12] Running SNR Sweep Experiment...")
    all_results["exp03"] = run_snr_sweep_experiment()

    print("\n[4/12] Running Waveform Comparison Experiment...")
    all_results["exp04"] = run_waveform_experiment()

    print("\n[5/12] Running CSM Dynamics Experiment...")
    all_results["exp05"] = run_csm_dynamics_experiment()

    print("\n[6/12] Running Band Weighting Experiment...")
    all_results["exp06"] = run_band_weighting_experiment()

    print("\n[7/12] Running Microphone Faults Experiment...")
    all_results["exp07"] = run_microphone_faults_experiment()

    print("\n[8/12] Running Multipath Experiment...")
    all_results["exp08"] = run_multipath_experiment()

    print("\n[9/12] Running Multi-Source Resolution Experiment...")
    all_results["exp09"] = run_multi_source_experiment()

    print("\n[10/12] Running Motion Tracking Experiment...")
    all_results["exp10"] = run_motion_tracking_experiment()

    print("\n[11/12] Running Queue Behavior Experiment...")
    all_results["exp11"] = run_queue_behavior_experiment()

    print("\n[12/12] Running Profiling Experiment...")
    all_results["exp12"] = run_profiling_experiment()

    out_path = os.path.join(PROJECT_ROOT, "benchmark", "results", "experiments_data.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"  Data saved to: {out_path}")
    print("  Plots saved to: benchmark/results/plots/")
    print("=" * 60)

if __name__ == "__main__":
    main()
