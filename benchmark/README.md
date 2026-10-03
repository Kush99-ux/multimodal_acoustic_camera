# Synthetic Acoustic Benchmark Subsystem

A permanent, reusable, and self-contained baseline and regression framework for evaluating and optimizing acoustic localization algorithms in the Real-Time Multimodal Acoustic Camera project.

---

## 1. Subsystem Architecture

```
benchmark/
├── config/                  # Scenario definitions & benchmark configuration
│   ├── benchmark_config.py  # Golden directions, tolerances, directory paths
│   └── scenarios.py         # Standard catalog of 128 benchmark scenarios
├── core/                    # Core simulation and evaluation engines
│   ├── generator.py         # Physics-based 16-ch spherical wavefront audio generator
│   ├── metrics.py           # Angular error, peak contrast, statistical distribution
│   ├── timing.py            # High-resolution profiler for pipeline stages
│   ├── plotting.py          # Publication-grade plotting suite (14 figures)
│   ├── comparison.py        # Differential multi-dimensional comparison engine
│   ├── registry.py          # Persistent run registry (benchmark_registry.json)
│   ├── runner.py            # Suite orchestrator & serialization
│   └── serialization.py     # JSON and CSV export routines
├── models/                  # Type-safe dataclass models
│   ├── scenario.py          # BenchmarkScenario, ReflectionPath, Impairment
│   ├── signal.py            # MultiChannelSignal
│   ├── source.py            # SoundSource, WaveformType
│   └── result.py            # TrialResult, ExperimentSummary, MetricDistribution
├── variants/                # Clean algorithm implementations for evaluation
│   ├── base_variant.py      # Abstract AcousticAlgorithmVariant base class
│   ├── baseline.py          # BaselineProductionVariant (wraps production code)
│   ├── csm_variants.py      # CSMAlphaVariant (fast/slow/instant), SlidingWindow
│   ├── band_weighting.py    # BandWeightingVariant (energy, raw, dynamic range, gated)
│   └── numerical_variants.py# PrecisionVariant (float32 vs float64)
├── experiments/             # 12 Standalone scientific research experiments
│   ├── exp01_direction_grid.py     # Spatial accuracy & bias heatmap
│   ├── exp02_frequency_sweep.py    # Frequency response & out-of-band roll-off
│   ├── exp03_snr_sweep.py          # SNR robustness & breakdown curves
│   ├── exp04_waveforms.py          # Waveform types & transient clap analysis
│   ├── exp05_csm_dynamics.py       # CSM EMA alpha smoothing vs tracking lag
│   ├── exp06_band_weighting.py     # Band weighting strategy comparison
│   ├── exp07_microphone_faults.py  # Dead channels & hardware jitter tolerance
│   ├── exp08_multipath.py          # Multipath reflection sensitivity
│   ├── exp09_multi_source.py       # Two-source resolution limits
│   ├── exp10_motion_tracking.py    # Dynamic source tracking lag vs velocity
│   ├── exp11_queue_behavior.py     # Real-time audio queue latency analysis
│   ├── exp12_profiling.py          # Stage-by-stage pipeline latency breakdown
│   └── run_all_experiments.py      # Master experiment runner
├── registry/
│   └── benchmark_registry.json     # Immutable ledger of all benchmark runs
├── results/
│   ├── baseline/            # Golden Baseline artifacts (results, metrics, summary)
│   ├── runs/                # Timestamped run directories
│   ├── plots/               # 14 Publication-grade generated PNG figures
│   └── comparisons/         # Generated differential markdown comparison reports
├── reports/
│   ├── SYNTHETIC_ACOUSTIC_BENCHMARK_REPORT.md  # Comprehensive scientific audit
│   └── ACOUSTIC_IMPROVEMENT_ROADMAP.md         # 13-area improvement roadmap
├── reference_analysis/
│   ├── REFERENCE_COMPARISON.md     # Comparative audit: reference vs production
│   └── REFERENCE_CANDIDATES.md     # Candidate hypotheses for future research
├── scripts/                 # Convenience entrypoint wrappers
│   ├── run_baseline.py      # Establish new golden baseline
│   ├── run_quick.py         # Run quick smoke test
│   └── run_standard.py      # Run standard research suite
├── tests/                   # Automated unittests for the benchmark itself
│   ├── test_benchmark_generator.py # Synthesizer geometry and physics tests
│   └── test_benchmark_regression.py# Golden directions regression test
├── run.py                   # Central CLI tool
├── compare.py               # Differential comparison tool
└── README.md                # This manual
```

---

## 2. Quick Start Guide

### Run Quick Regression Smoke Test (~15 seconds)
```bash
python benchmark/run.py --quick
```

### Run Standard Research Suite (~25 seconds)
```bash
python benchmark/run.py --standard
```

### Establish a New Golden Baseline
```bash
python benchmark/run.py --baseline
```

### Benchmark an Algorithmic Variant
```bash
python benchmark/run.py --standard --variant band_energy
python benchmark/run.py --standard --variant csm_ema_fast
```

### Compare Candidate Run Against Golden Baseline
```bash
python benchmark/compare.py --candidate <RUN_ID>
```

### Execute All 12 Scientific Experiments & Generate 14 Plots
```bash
python benchmark/experiments/run_all_experiments.py
```

### Run Benchmark Self-Tests
```bash
python -m unittest discover -s benchmark/tests
```

---

## 3. Standard Optimization Workflow

Whenever developing an acoustic optimization:

1. **Verify Baseline State**: Ensure `benchmark/results/baseline/` is present.
2. **Implement Candidate Variant**: Inherit from `AcousticAlgorithmVariant` in `benchmark/variants/`.
3. **Execute Benchmark**: Run `python benchmark/run.py --standard --variant <variant_name>`.
4. **Compare Against Baseline**: Run `python benchmark/compare.py --candidate <RUN_ID>`.
5. **Analyze Multi-Dimensional Metrics**: Inspect the comparison table:
   - Median Angular Error (deg)
   - P95 Angular Error (deg)
   - Failure Rate (%)
   - Median Processing Time (ms)
   - Peak Contrast Ratio
6. **Decision**:
   - If performance improves across dimensions without unacceptable regressions: promote to production!
   - If regressions occur: inspect generated plots and refine.

---

## 4. Key Scientific Results

| Metric | Baseline Production | Candidate (`band_weighting_energy`) | Status |
| :--- | :---: | :---: | :---: |
| **Median Angular Error** | 0.25° | 0.25° | Bounded by grid step |
| **P95 Angular Error** | 28.79° | **7.76°** | **-21.03° Improved** |
| **Failure Rate** | 20.3% | **7.8%** | **-12.5% Absolute Improvement** |
| **Narrowband Tone Error** | 26.53° | **0.25°** | **Fixed normalization flaw** |
| **Transient Clap (10 ms)** | 0.25° | 0.25° | 100% Reliable detection |
| **Latency per Frame** | 10.0 ms | 15.0 ms | Well within 85.3 ms budget |
| **Dead Mic Tolerance** | Up to 3 dead | Up to 3 dead | Sub-0.5° error maintained |
| **Breakdown SNR** | -5 dB | -5 dB | Robust under Gaussian/pink noise |
