# Real-Time Multimodal Acoustic Camera

A multimodal sensor fusion platform that integrates **16-channel phased-array acoustic direction-of-arrival (DoA) localization** with **real-time visual instance segmentation** to localize and identify sound emitters in physical environments.

---

## Overview

The system combines acoustic array signal processing with computer vision into a concurrent sensing pipeline:

- **miniDSP UMA-16 v2**: 16-channel uniform rectangular microphone array capturing synchronized multi-channel audio at 48 kHz.
- **ESP32-CAM (AI Thinker)**: Outward-facing wireless video streamer providing 640×480 optical imagery.
- **Multi-Band MUSIC Localization**: Sub-band Cross-Spectral Matrix (CSM) formation, Exponential Moving Average (EMA) smoothing, and eigen-space noise subspace decomposition across 6 discrete frequency bands (500–3000 Hz) over a 7,676-point angular grid.
- **YOLO11n-seg Vision Engine**: Instance segmentation for target object classification, bounding box tracking, and centroid extraction.
- **Temporal Synchronization Layer**: Queue-based monotonic timestamp matching achieving sub-100 ms cross-modal alignment (median latency delta = 28.5 ms).
- **Multimodal Acoustic/Visual Fusion**: Pinhole angular-to-pixel projection and dynamic false-color overlay rendering acoustic heatmaps directly onto the optical stream.

---

## Architecture

```text
Audio Acquisition (UMA-16)  --> Multi-Band MUSIC --> Acoustic Direction-of-Arrival (DoA)
                                                                 |
                                                                 v
                                                    Timestamp Synchronization (Δt < 100 ms)
                                                                 ^
                                                                 |
Camera Ingestion (ESP32-CAM) --> YOLO11n-seg   --> Visual Scene / Object Centroids
                                                                 |
                                                                 v
                                                        Multimodal Fusion
                                                                 |
                                                                 v
                                                    Acoustic Heatmap Overlay (OpenCV)
```

---

## Hardware

- **Acoustic Array**: miniDSP UMA-16 v2
  - 16 Knowles SPH0645LM4H MEMS microphones arranged in a planar geometry.
  - Inter-element spacing: $d = 42\text{ mm}$ (spatial aliasing threshold $\approx 4.08\text{ kHz}$).
  - Coordinate convention: X-Z array plane, +Y broadside.
  - Acquisition: 16 channels, 48,000 Hz, 24-bit resolution, USB Audio Class 2.0 (ASIO / PortAudio).
- **Camera Streamer**: ESP32-CAM AI Thinker
  - Sensor: Omnivision OV2640.
  - Lens Field of View: 50.0° Horizontal, 37.5° Vertical.
  - Resolution: 640×480 pixels via HTTP MJPEG stream over dedicated 2.4 GHz Wi-Fi.
- **Host Platform**: Windows 11 workstation running Python 3.12 (CPython x86_64).

---

## Software

The codebase is organized into modular packages:

- **`acoustic/`**: Low-level audio acquisition driver (`audio.py`), UMA-16 physical coordinate definitions (`geometry.py`), Acoular-equivalent steering vector precomputation (`steering.py`), multi-band MUSIC localizer (`music.py`), and data models (`models.py`).
- **`vision/`**: Thread-isolated non-blocking HTTP MJPEG ingestion reader (`camera.py`), Ultralytics YOLO11n-seg segmentation wrapper (`segmentation.py`), and data models (`models.py`).
- **`fusion/`**: Monotonic queue timestamp synchronizer (`synchronization.py`), angular-to-pixel coordinate projection visualizer (`visualization.py`), multimodal fusion engine (`engine.py`), and data models (`models.py`).
- **`pipeline/`**: Orchestration pipeline (`multimodal.py`) coupling concurrent capture, processing, and rendering threads.
- **`config/`**: Centralized configuration repository (`settings.py`) governing all hardware, algorithmic, and calibration parameters.

---

## Validation Status

> **Core multimodal functionality has been experimentally validated under the tested conditions. Horizontal and vertical directional ordering, audio acquisition, camera acquisition, synthetic MUSIC behavior, and multimodal timestamp synchronization were demonstrated. Pixel-level acoustic-to-camera projection calibration remains ongoing.**

- **Acoustic Directional Tracking**:
  - Horizontal ordering validated across three physical positions: LEFT ($106.0^\circ$) > CENTER ($90.8^\circ$) > RIGHT ($65.0^\circ$).
  - Vertical ordering validated across three physical positions: ABOVE ($+12.25^\circ$) > LEVEL ($-6.00^\circ$) > BELOW ($-17.00^\circ$).
  - Synthetic DoA benchmark: 11/11 tests passed within $0.25^\circ$ discretization resolution.
- **Acoustic Multipath Investigation**:
  - Unshielded indoor tabletop trials revealed coherent multipath reflections biasing elevation estimates downward toward the $-18.8^\circ$ boundary.
  - Deploying an acoustic absorption towel beneath the source substantially reduced the reflection effect, restoring level elevation tracking to $-7.8^\circ$ to $-8.75^\circ$.
- **Temporal Alignment**:
  - Evaluated across continuous streaming campaigns: 100% of frames satisfied the $\Delta t < 100\text{ ms}$ threshold (median delta = $28.5\text{ ms}$).
- **Projection Calibration Limitation**:
  - Pinhole visual projection achieved a median Euclidean error of $77.3\text{ px}$ under towel-shielded conditions ($dx = 1\text{ px}$ on best frame, residual $dy \approx 59\text{ px}$).
  - Non-linear camera lens distortion correction and rigid physical transformation calibration remain identified areas for future refinement.

---

## Repository Structure

```text
multimodal_acoustic_camera/
├── acoustic/               # Audio acquisition, array geometry, MUSIC localization
├── vision/                 # ESP32-CAM stream reader, YOLO11 instance segmentation
├── fusion/                 # Timestamp synchronization, coordinate projection, heatmap overlay
├── pipeline/               # Concurrent multimodal orchestration pipeline
├── config/                 # Centralized hardware and algorithmic settings (settings.py)
├── models/                 # Neural network weights (yolo11n-seg.pt)
├── test/                   # Standalone unit tests and diagnostic test harnesses
├── scratch/                # Forensic validation utilities, diagnostic tools, and raw evidence
│   └── report_generation/  # Report compilation pipeline, plot generators, and QC audit scripts
├── reference_original/     # READ-ONLY golden reference benchmark implementation
├── Reports/                # Historical milestone reports and validation documentation
├── PROJECT_FINAL_INTEGRATED_VALIDATION_REPORT.pdf  # Comprehensive 14-page engineering validation report
├── requirements.txt        # Python dependency manifest
├── .gitignore              # Repository exclusion rules
└── README.md               # Project documentation
```

---

## Running the Project

### 1. Environment Verification
Verify that all core modules compile cleanly:
```powershell
.venv\Scripts\python.exe -m compileall acoustic vision fusion pipeline config
```

### 2. Standalone Subsystem Tests
Run data model contracts and verification tests:
```powershell
.venv\Scripts\python.exe -m test.test_acoustic_models
.venv\Scripts\python.exe -m test.test_vision_models
```

### 3. Standalone Acoustic Driver & MUSIC Validation (Hardware Connected)
Verify 16-channel PortAudio stream acquisition:
```powershell
.venv\Scripts\python.exe test_audio_acquisition.py
```
Run standalone multi-band MUSIC processing on a captured audio frame:
```powershell
.venv\Scripts\python.exe test_music_audioframe.py
```

### 4. Forensic Diagnostics & Report Generation
Recompile the 14-page engineering validation report:
```powershell
.venv\Scripts\python.exe scratch/report_generation/build_revised_report.py
```
Execute the automated technical claim and formatting quality audit:
```powershell
.venv\Scripts\python.exe scratch/report_generation/qc_audit.py
```

### 5. Live Multimodal Pipeline Execution (Hardware Connected)
Execute the synchronized 20-frame integrated multimodal validation suite:
```powershell
.venv\Scripts\python.exe scratch/final_integrated_validation.py
```
Run interactive real-time visualizer with heatmap fusion:
```powershell
.venv\Scripts\python.exe scratch/live_multimodal_heatmap_fusion.py
```

---

## Experimental Validation Report

For complete mathematical derivations, Acoular equivalence proofs, multi-band covariance formulations, empirical telemetry tables, and forensic debugging analyses, refer to the project report:

**[`PROJECT_FINAL_INTEGRATED_VALIDATION_REPORT.pdf`](PROJECT_FINAL_INTEGRATED_VALIDATION_REPORT.pdf)**

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
