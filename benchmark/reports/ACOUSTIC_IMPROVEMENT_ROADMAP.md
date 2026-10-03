# Acoustic Improvement Roadmap: Prioritized Engineering & Research Initiatives

**System**: Real-Time Multimodal Acoustic Camera Subsystem  
**Baseline Version**: `baseline_production_standard_2026_09_28_110508`  
**Date**: September 28, 2026  
**Document Classification**: Engineering Specification & Research Roadmap  

---

## Strategic Overview

The **Synthetic Acoustic Benchmark Subsystem** has established rigorous empirical baselines for the acoustic localization pipeline. Through 12 controlled experiments across 128 standard scenarios, specific scientific bottlenecks and performance ceilings of the current implementation were identified.

This document formalizes the **13 Prioritized Acoustic Improvement Areas**, categorized by impact, technical risk, mathematical formulation, implementation complexity, and verification criteria using the benchmark suite.

---

## Prioritization Summary Matrix

| ID | Improvement Area | Priority | Expected Impact | Implementation Complexity | Benchmark Verification |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **01** | **Intelligent Frequency-Band Weighting** | **P0 (Critical)** | Failure rate: 20.3% → 7.8% | Low (~30 lines) | `--standard --variant band_energy` |
| **02** | **Transient Sound Fast-Path Trigger** | **P1 (High)** | Instantaneous snap detection | Low (~45 lines) | `exp04_waveforms.py` |
| **03** | **Adaptive CSM Temporal Smoothing** | **P1 (High)** | Lag reduced by 60% during motion | Medium (~60 lines) | `exp05_csm_dynamics.py`, `exp10` |
| **04** | **Array Calibration & Dead-Channel Masking** | **P2 (Medium)** | Robustness to hardware faults | Medium (~80 lines) | `exp07_microphone_faults.py` |
| **05** | **Elevation Bias Calibration Correction** | **P2 (Medium)** | Eliminates +0.25° vertical bias | Low (~15 lines) | `exp01_direction_grid.py` |
| **06** | **Multipath Spatial Deconvolution** | **P2 (Medium)** | Robustness in reflective rooms | High (~150 lines) | `exp08_multipath.py` |
| **07** | **Multi-Source Peak Disambiguation** | **P2 (Medium)** | Multi-speaker tracking | Medium (~90 lines) | `exp09_multi_source.py` |
| **08** | **Adaptive Noise Floor Estimation** | **P2 (Medium)** | Dynamic environment thresholding | Low (~40 lines) | `exp03_snr_sweep.py` |
| **09** | **Optimized Spherical Steering Projection** | **P3 (Efficiency)** | 9.2 ms → 4.5 ms execution | Medium (~70 lines) | `exp12_profiling.py` |
| **10** | **Lock-Free Ring Buffer Audio Capture** | **P3 (Efficiency)** | Zero memory copy jitter | Medium (~100 lines) | `exp11_queue_behavior.py` |
| **11** | **Multimodal Spatial Prior Feedback** | **P3 (System)** | 50% search grid reduction | High (~180 lines) | Full Multimodal App |
| **12** | **Continuous Synthetic Regression CI Pipeline** | **P3 (DevOps)** | Zero-regression assurance | Low (~50 lines) | `benchmark/tests/` |
| **13** | **Float32 Numerical Optimization** | **P3 (Efficiency)** | 25% CPU memory cache savings | Low (~25 lines) | `benchmark/variants/numerical` |

---

## Detailed Improvement Specifications

### Area 01: Intelligent Frequency-Band Weighting & Active Gating [PRIORITY: P0]
- **Current Limitation**: Production MUSIC applies independent min-max normalization `(spec - min) / (max - min)` to all 6 sub-bands. Inactive bands inflate noise floors to $[0, 1]$, causing up to 26° localization error on narrowband sounds.
- **Proposed Solution**: Weight sub-bands by physical spectral energy:
  $$w_b = \frac{\sum_{k} \|X_b(k)\|^2}{\sum_{b'} \sum_k \|X_{b'}(k)\|^2}$$
  Or apply hard noise-gating: discard bands where band SPL is $< 10$ dB above floor.
- **Benchmark Proof**: The benchmarked `band_weighting_energy` candidate variant demonstrated a **61.6% reduction in overall failure rate** and **sub-0.25° error** on pure tones.

### Area 02: Transient Sound Fast-Path Trigger [PRIORITY: P1]
- **Current Limitation**: When a finger snap or hand clap occurs, the 4096-sample window captures the burst, but the CSM EMA smoothing ($\alpha = 0.30$) requires several frames to reflect the full energy of the onset, muting transient response.
- **Proposed Solution**: Implement an acoustic onset flux detector:
  $$\text{Flux}_k = \sum_f \max(0, |X_k(f)| - |X_{k-1}(f)|)$$
  When $\text{Flux}_k$ exceeds threshold $\tau_{\text{transient}}$, temporarily bypass EMA ($\alpha = 1.0$) for that frame to yield an instantaneous localization fix.

### Area 03: Adaptive CSM Temporal Smoothing [PRIORITY: P1]
- **Current Limitation**: Fixed $\alpha = 0.30$ imposes a 600 ms lag during source motion, causing acoustic hotspots to lag behind moving speakers at velocities $> 30^\circ$/s.
- **Proposed Solution**: Modulate $\alpha$ dynamically as a function of spatial innovation:
  $$\alpha_k = \alpha_{\text{min}} + (\alpha_{\text{max}} - \alpha_{\text{min}}) \cdot \tanh(\gamma \cdot \|\mathbf{R}_{\text{inst}} - \mathbf{R}_{k-1}\|_F)$$
  Yielding $\alpha \approx 0.15$ during static holding and $\alpha \to 0.85$ during rapid angular shifts.

### Area 04: Microphone Array Calibration & Dead-Channel Masking [PRIORITY: P2]
- **Current Limitation**: Array assumes 16 perfectly matched microphones. A dead channel or electronic gain shift distorts the noise subspace projection.
- **Proposed Solution**: Implement an automated startup and online diagnostic routine: compute diagonal variance $\sigma_m^2 = \mathbb{E}[x_m(t)^2]$. If $\sigma_m^2 < 0.01 \cdot \bar{\sigma}^2$, automatically zero out the $m$-th row and column of the CSM and slice the $m$-th row of the steering matrix.

### Area 05: Elevation Bias Calibration Correction [PRIORITY: P2]
- **Current Limitation**: Benchmark revealed a persistent systematic $+0.25^\circ$ elevation offset across all test scenarios caused by grid quantization centering.
- **Proposed Solution**: Re-center the elevation grid generation in `acoustic/steering.py` around exact zero:
  `np.arange(-18.5, 18.51, 0.5)` instead of half-integer offset endpoints.

### Area 06: Multipath Spatial Deconvolution [PRIORITY: P2]
- **Current Limitation**: Acoustic reflections with $\text{SIR} < 10$ dB create peak pulling (3° to 7° error).
- **Proposed Solution**: Incorporate spatial smoothing or forward-backward averaging across subarray virtual apertures to decorrelate coherent multipath reflections prior to eigendecomposition.

### Area 07: Multi-Source Peak Disambiguation [PRIORITY: P2]
- **Current Limitation**: Production pipeline calls `np.argmax(spec)` which returns only 1 source.
- **Proposed Solution**: Implement 2D non-maximum suppression (NMS) over the pseudo-spectrum heatmap to identify secondary peaks exceeding a dynamic threshold (e.g. within 6 dB of global maximum).

### Area 08: Dynamic Adaptive Noise Floor Gating [PRIORITY: P2]
- **Current Limitation**: A hardcoded 15.0 dB SPL threshold (`calculate_db_level`) gates localization. In quiet rooms, genuine whispers fail to trigger; in loud rooms, background noise continuously triggers computation.
- **Proposed Solution**: Implement a recursive minimum statistics noise floor tracker:
  $$P_{\text{floor}}(t) = \min_{\tau \in [t-T, t]} P(\tau)$$
  Trigger localization when $P(t) > P_{\text{floor}}(t) + 6.0\text{ dB}$.

### Area 09: Fast Spherical Steering Projection [PRIORITY: P3]
- **Current Limitation**: Steering projection consumes 79.3% of execution time (9.22 ms).
- **Proposed Solution**: Pre-multiply and pack steering matrices using float32 contiguous C-order layouts, or evaluate OpenBLAS GEMM threading optimizations.

### Area 10: Lock-Free Audio Buffer Capture [PRIORITY: P3]
- **Current Limitation**: The PortAudio callback in `acoustic/audio.py` appends to a Python list protected by a mutex lock, introducing potential GIL contention.
- **Proposed Solution**: Replace list concatenation with a lock-free circular ring buffer implemented in NumPy/C.

### Area 11: Multimodal Spatial Prior Guided Search [PRIORITY: P3]
- **Current Limitation**: The full $101 \times 76$ grid (7,676 points) is evaluated every frame regardless of visual bounding box locations.
- **Proposed Solution**: When YOLO detects a speaker at $(\theta_v, \phi_v)$, restrict the MUSIC projection to a focused ROI window around the visual prior $(\theta_v \pm 10^\circ, \phi_v \pm 5^\circ)$, reducing steering projection time by $> 75\%$.

### Area 12: Continuous Synthetic Regression CI/CD Pipeline [PRIORITY: P3]
- **Proposed Solution**: Integrate `benchmark/run.py --quick` into automated pre-commit / pull-request hooks to prevent inadvertent regressions in acoustic accuracy or latency.

### Area 13: Float32 Precision Optimization [PRIORITY: P3]
- **Proposed Solution**: Benchmark `PrecisionVariant` verified that float32 precision introduces $< 0.001^\circ$ error relative to float64, while cutting memory footprint and cache miss rate by 50%.
