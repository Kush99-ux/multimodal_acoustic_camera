# Reference Comparison: Production vs. Golden Reference (`reference_original/`)

## 1. Overview & Forensic Context

This document provides a line-by-line comparative audit of the acoustic signal processing and localization implementation in `reference_original/uma_live_camera.py` versus the current production acoustic subsystem (`acoustic/music.py`, `acoustic/steering.py`, `acoustic/geometry.py`, `acoustic/audio.py`, `config/settings.py`).

The golden reference represents the historical script that demonstrated reliable transient detection (finger snaps, hand claps) at distance. The purpose of this comparison is to identify exact algorithmic, mathematical, and architectural equivalences and differences, evaluate their scientific and performance implications, and formulate controlled synthetic experiments.

---

## 2. Exhaustive Dimension-by-Dimension Comparison

| Architectural Dimension | Golden Reference (`reference_original/uma_live_camera.py`) | Current Production (`acoustic/` + `config/`) | Algorithmic / Mathematical Difference | Scientific & Practical Significance | Benchmark Candidate? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Microphone Geometry Array** | `mic_raw`: 16 coordinates in (X, Z, 0) format, remapped to Acoular (X, Y=0, Z). | `ORIGINAL_MIC_RAW` in `acoustic/steering.py`, identical remapping to Acoular. | **None**. Exact identical coordinates: $X \in [-0.063, +0.063]$, $Z \in [-0.063, +0.063]$, $Y=0$. | Ensures identical spatial aperture and spatial array manifold. | Baseline Verified |
| **Array Coordinate Convention** | Array plane: X-Z. Broadside forward: +Y. Azimuth 90° = forward. | Array plane: X-Z. Broadside forward: +Y. Azimuth 90° = forward. | **None**. Exact equivalence restored in Stage B. | Prevents coordinate inversion errors between left and right. | Baseline Verified |
| **Scanning Grid Geometry** | Azimuth 65° to 115° (0.5° step, 101 pts). Elevation -18.75° to +18.75° (0.5° step, 76 pts). Total = 7,676 directions. | Identical angular grid generated via `SteeringFactory`: 101 az × 76 el = 7,676 directions. | **None**. Exact matching dimensions and angular step. | Preserves identical spatial resolution and search manifold. | Baseline Verified |
| **Steering Formulation** | `acoular.SteeringVector(grid, mics, steer_type='true level')`. Precomputed at startup. | `acoular.SteeringVector(grid, mics, steer_type='true level')`. Precomputed in `SteeringFactory`. | **None**. Matrices are cached as `(channels, grid_points)`. | Guarantees identical acoustic transfer function and unit-norm normalization. | Baseline Verified |
| **Frequency Sub-Bands** | 6 sub-bands: 500, 1000, 1500, 2000, 2500, 3000 Hz. Bandwidth = 500 Hz (±250 Hz around center). | 6 sub-bands: 500, 1000, 1500, 2000, 2500, 3000 Hz. Bandwidth = 500 Hz (±250 Hz around center). | **None**. Bin selection logic via `np.argmin(np.abs(freqs - f))` is identical. | Targets primary human speech and transient energy band. | Baseline Verified |
| **STFT & Windowing** | Hann window `np.hanning(4096)[:, None]`, `np.fft.rfft(windowed, n=4096, axis=0)`. | Hann window `np.hanning(4096)[:, None]`, `np.fft.rfft(windowed, axis=0)`. | **None**. Identical spectral resolution (48 kHz / 4096 = 11.71875 Hz/bin). | Sidelobe suppression (-31.5 dB) for STFT bin isolation. | Baseline Verified |
| **CSM Calculation** | `csm_instant = np.einsum('fi,fj->ij', X_band, X_band.conj()) / X_band.shape[0]`. | `csm = (X_band @ X_band.conj().T) / float(bin_count)`. Added `csm = 0.5 * (csm + csm.conj().T)`. | Matrix multiplication vs `einsum` (mathematically identical); production enforces exact numerical Hermiticity. | Enforcing Hermiticity eliminates small imaginary drift on diagonal without affecting eigenvalues. | Yes (Numerical test) |
| **CSM Temporal Smoothing (EMA)** | Exponential moving average with $\alpha = 0.30$: $\text{CSM}_{t} = 0.7 \text{CSM}_{t-1} + 0.3 \text{CSM}_{inst}$. | Identical EMA smoothing with $\alpha = 0.30$. | **None**. Exact identical recursive formula. | Balances covariance rank enhancement against temporal lag. | Yes (CSM Study) |
| **CSM Regularization** | $R = \text{CSM} + 10^{-10} I_{16}$. | $R = \text{CSM} + 10^{-10} I_{16}$. | **None**. Exact identical diagonal loading. | Prevents singularity during eigendecomposition. | Baseline Verified |
| **Subspace Decomposition** | `np.linalg.eigh(R)`. Noise subspace: `eigvecs[:, :-1]` (columns 0..14). Signal: column 15. | `np.linalg.eigh(R)`. Noise subspace: `eigvecs[:, :-1]`. | **None**. Assumes 1 dominant acoustic source ($N_{sources}=1$). | Correct for single acoustic target; degraded for multiple concurrent sources. | Yes (Multi-source Study) |
| **Steering Projection & Inversion** | $\text{proj} = E_n^H A$. $\text{spectrum} = 1.0 / (\sum \|\text{proj}\|^2 + 10^{-12})$. | $\text{proj} = E_n^H A$. $\text{spectrum} = 1.0 / (\sum \|\text{proj}\|^2 + 10^{-12})$. | **None**. Identical MUSIC pseudo-spectrum formulation. | Orthogonal projection reciprocal creates sharp localization peaks. | Baseline Verified |
| **Per-Band Normalization** | Per-band Min-Max normalization to $[0, 1]$ before accumulation. | Per-band Min-Max normalization to $[0, 1]$ before accumulation. | **None**. Restored in current production. | Prevents high-energy low bands from overwhelming quieter high-frequency bands, but can amplify noise-only bands. | **Crucial Candidate** (Weighting Study) |
| **Broadband Accumulation** | Direct sum of 6 normalized band spectra: $P_{total} = \sum_{k=1}^6 P_k$. | Direct sum of 6 normalized band spectra: $P_{total} = \sum_{k=1}^6 P_k$. | **None**. Identical unweighted sum. | Equal weighting across all 6 bands regardless of SNR or spatial coherence. | **Crucial Candidate** (Weighting Study) |
| **Post-Accumulation Contrast** | Global Min-Max $\to$ Sigmoid $\frac{1}{1 + e^{-3.0(x - 0.5)}}$ $\to$ Global Min-Max. | Global Min-Max $\to$ Sigmoid $\frac{1}{1 + e^{-3.0(x - 0.5)}}$ $\to$ Global Min-Max. | **None**. Identical non-linear contrast shaping. | Suppresses low-level sidelobes, sharpening visualization. | Baseline Verified |
| **Noise Floor Gating** | $20 \log_{10}(\text{RMS}) + 90.0 \ge 15.0\text{ dB}$. If below, clears spectrum and resets CSM history. | Identical $20 \log_{10}(\text{RMS}) + 90.0 \ge 15.0\text{ dB}$. Resets CSM state on gate drop. | **None**. Identical energy calculation and gate threshold. | Rejects ambient silence; prevents stale noise accumulation. | Yes (Gate Study) |
| **Audio Acquisition Queueing** | `while not audio_queue.empty():` drains ALL buffered blocks into rolling `audio_buffer`. | Previously: FIFO pop 1 frame. **Fixed**: Latest-frame drain returning newest 4096 samples from buffer tail. | Production now matches reference's real-time latest-frame philosophy. | Eliminates queue drift and multi-second latency lag under slow CPU consumer. | Yes (Queue Study) |
| **Distance Estimation Heuristic** | Blend of 60% normalized dB + 40% beam sharpness (peak-to-mean), inverted $\to [0.5, 5.0]\text{ m}$. | Not present in current production `music.py`. | Reference had empirical distance heuristic. | Interesting heuristic combining energy and spatial concentration. | Yes (Candidate for Study) |
| **Frequency Presets** | Keyboard 1/2/3: Low (300-1000), Speech (500-3000), High (2000-4000). Recomputes steering on fly. | Fixed 6 bands (500-3000 Hz). Precomputed once. | Reference allowed dynamic preset switching. | Dynamic presets provide adaptability for different acoustic environments. | Future Roadmap |
| **Heatmap Visualization Rendering** | Matplotlib `FigureCanvasAgg` rendered to RGBA buffer then converted to BGR via OpenCV (~25 ms). | Direct OpenCV `cv2.applyColorMap(h_u8, cv2.COLORMAP_INFERNO)` (~0.4 ms). | OpenCV direct colormapping is ~60× faster than Matplotlib. | Critical optimization for maintaining 20+ FPS on CPU. | Preserved Production |

---

## 3. Conclusions from the Reference Code Audit

1. **Algorithmic Equivalence**: The core mathematical MUSIC algorithm in `acoustic/music.py` is identical to `reference_original/uma_live_camera.py`:
   - Exact same microphone positions in Acoular coordinate frame.
   - Exact same steering vector formulation (`steer_type="true level"`).
   - Exact same sub-band center frequencies $(500, 1000, 1500, 2000, 2500, 3000)$ Hz and bandwidths (500 Hz).
   - Exact same Hann windowing, STFT, CSM computation, EMA smoothing ($\alpha=0.30$), regularization ($10^{-10}$), noise subspace extraction ($N-1=15$), and pseudo-spectrum inversion.
   - Exact same per-band min-max normalization, global summation, and sigmoid contrast shaping ($g=3.0, m=0.5$).
2. **Key Areas for Synthetic Investigation**:
   - **Per-Band Min-Max Normalization**: Does normalizing each band independently force a band containing pure noise to produce spurious peaks with equal weight to a band containing a clean acoustic signal?
   - **Band Accumulation Strategy**: Can energy weighting, dynamic-range weighting, or active-band gating outperform equal summation?
   - **CSM Smoothing Tradeoffs**: What is the quantitative tradeoff between spatial stability (low $\alpha$) and transient response speed (high $\alpha$)?
   - **Audio Queueing**: Quantitative synthetic proof demonstrating why latest-frame draining guarantees real-time transient detection whereas FIFO accumulation causes unbounded latency under CPU-bound pipelines.
