# Reference Candidates: Algorithmic & Architectural Hypotheses

## 1. Executive Summary

Based on the forensic audit of `reference_original/` and the project requirements, this document specifies the candidate algorithmic modifications to be evaluated by the synthetic benchmark subsystem.

None of these candidates are to be introduced directly into production code. Each candidate will be implemented as an offline benchmark variant under `benchmark/variants/`, benchmarked across the standard suite, and compared against the golden baseline.

---

## 2. Prioritized Candidate Matrix

### Candidate 1: Frequency-Band Weighting vs. Equal Min-Max Normalization
- **Current Production & Reference Baseline**:
  Every sub-band pseudo-spectrum is min-max normalized to $[0, 1]$ and then directly summed with equal weight:
  $$P_{total}(\theta, \phi) = \sum_{k=1}^6 \frac{P_k(\theta, \phi) - \min P_k}{\max P_k - \min P_k}$$
- **The Problem**:
  If a sound source only has energy in the 1000 Hz band (e.g., pure tone, whistle, or vocal formant), the other 5 bands contain pure ambient noise. Min-max normalizing those 5 noise bands inflates background noise peaks to full $[0, 1]$ amplitude, creating 5 spurious pseudo-spectra that corrupt the single true peak.
- **Proposed Benchmark Variants**:
  1. `raw_accumulation`: Accumulate pseudo-spectra directly without per-band min-max normalization.
  2. `energy_weighted`: Weight each band's normalized spectrum by its relative spectral energy $E_k = \sum_{f \in \text{band}_k} |X(f)|^2$.
  3. `dynamic_range_weighted`: Weight each band by its raw peak-to-floor contrast $(\max P_k - \min P_k) / \text{median}(P_k)$.
  4. `active_band_gated`: Only accumulate bands whose energy exceeds an SNR threshold relative to the noise floor.
- **Evaluation Criteria**: Directional accuracy under single-tone, multi-tone, and band-limited noise.

### Candidate 2: CSM Temporal Smoothing (EMA Factor $\alpha$)
- **Current Production & Reference Baseline**:
  $$\text{CSM}_t = (1 - \alpha) \text{CSM}_{t-1} + \alpha \text{CSM}_{inst}, \quad \alpha = 0.30$$
- **The Tradeoff**:
  - $\alpha = 0.30$ provides spatial smoothing by averaging across multiple frames, increasing the rank of the covariance matrix for stationary sources.
  - However, for transient sources (claps, snaps) or fast-moving sound sources, $\alpha = 0.30$ introduces temporal lag and bleeds energy across frames.
- **Proposed Benchmark Variants**:
  1. `csm_no_ema`: $\alpha = 1.0$ (instantaneous frame covariance, zero temporal memory).
  2. `csm_ema_fast`: $\alpha = 0.60$ (responsive to rapid direction changes and transients).
  3. `csm_ema_slow`: $\alpha = 0.10$ (high stability for continuous stationary noise).
  4. `csm_sliding_window`: Sliding window of $K=3$ consecutive instantaneous CSMs with equal weight.
- **Evaluation Criteria**: Stationary accuracy vs. transient latency vs. tracking lag for moving sources.

### Candidate 3: Subspace Estimation & Multi-Source Rank
- **Current Production & Reference Baseline**:
  Always partitions subspace into 1 signal eigenvector (column 15) and 15 noise eigenvectors (columns 0..14).
- **The Problem**:
  When 2 or 3 simultaneous sound sources are active, the signal subspace has dimension 2 or 3. If only 1 eigenvector is removed from the noise subspace, the remaining signal eigenvectors leak into the noise subspace, suppressing the second source and causing severe localization errors or ghost peaks.
- **Proposed Benchmark Investigation**:
  Evaluate the eigenvalue spectrum under 1, 2, and 3 simultaneous sources and measure multi-source resolution as a function of assumed source count $N_{sources}$.

### Candidate 4: Distance Estimation Heuristic
- **Reference Implementation Technique**:
  `estimate_distance()` combined signal dB with spatial concentration (peak-to-mean sharpness):
  $$\text{sharpness} = \frac{P_{peak}}{\text{mean}(P) + 10^{-6}}$$
  $$\text{score} = 0.6 \cdot \text{clamp}\left(\frac{\text{dB} - 30}{40}\right) + 0.4 \cdot \text{clamp}\left(\frac{\text{sharpness}}{10}\right)$$
  $$\text{distance} = \text{clamp}\left(\frac{1.0}{\text{score} + 10^{-6}}, 0.5, 5.0\right)$$
- **Scientific Assessment**:
  Not a physical triangulation, but a useful confidence-correlated indicator. Peak sharpness directly measures spatial concentration. The benchmark will evaluate whether peak sharpness correlates strongly with angular estimation confidence.
