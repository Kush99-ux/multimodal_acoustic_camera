# Benchmark Comparison Report: `band_weighting_energy` vs `baseline_production`
**Candidate Run ID**: `band_weighting_energy_standard_2026_09_28_110541`  
**Baseline Run ID**: `baseline_production_standard_2026_09_28_110508`  
**Suite Tier**: `standard`  

## 1. Multi-Dimensional Performance Matrix

| Metric | Baseline | Candidate | Delta | Evaluation |
| :--- | :---: | :---: | :---: | :---: |
| **Median Angular Error** | 0.25 deg | 0.25 deg | +0.00 deg | [EQUIVALENT] |
| **P95 Angular Error** | 28.79 deg | 7.76 deg | -21.03 deg | [IMPROVED] |
| **Failure Rate** | 20.3% | 7.8% | -12.5% | [IMPROVED] |
| **Median Processing Time** | 10.0 ms | 15.0 ms | +5.0 ms | [SLOWER] |
| **Median Peak Contrast** | 1.00 | 1.00 | +0.00 | [EQUIVALENT] |

## 2. Tradeoff Analysis
### Key Improvements:
- [+] P95 angular error improved by -21.03 deg
- [+] Failure rate decreased by -12.5%

### Regressions & Tradeoffs:
- [-] Processing latency increased by +5.0 ms
