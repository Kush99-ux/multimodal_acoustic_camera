# Manual Testing & Faculty Demonstration Guide

This manual details how to launch, operate, and systematically verify every capability of the **Real-Time Multimodal Acoustic Camera** using the presentation-ready Professor Demo application.

---

## A. Installation & Environment

The project runs in a dedicated Python 3.12 virtual environment (`.venv`) with PyTorch, Ultralytics YOLO, OpenCV, Acoular, NumPy, and SoundDevice.

1. Open PowerShell in the project root:
   ```powershell
   cd C:\Users\kushs\OneDrive\Desktop\Projects\multimodal_acoustic_camera
   ```
2. Verify Python virtual environment:
   ```powershell
   .venv\Scripts\python.exe --version
   ```
3. Verify syntax and dependencies:
   ```powershell
   .venv\Scripts\python.exe -m compileall -q demo acoustic vision fusion pipeline config
   ```

---

## B. Launching LIVE Mode

Use LIVE mode when the ESP32-CAM and miniDSP UMA-16 hardware are connected.

### Standard Launch
```powershell
.venv\Scripts\python.exe demo\app.py
```

### Custom Camera IP Launch
If the ESP32-CAM receives a new DHCP IP address, pass `--stream-url`:
```powershell
.venv\Scripts\python.exe demo\app.py --stream-url http://<ESP32_IP>:81/stream
```

---

## C. Launching SIMULATION Mode

Use SIMULATION mode to demonstrate the complete acoustic, vision, fusion, and diagnostic pipeline on any PC without physical hardware.

```powershell
.venv\Scripts\python.exe demo\app.py --simulate
```

**Simulation Characteristics:**
- Replays authentic 16-channel 48 kHz recordings (`scratch/moderate_left_audio_frames.npy`) and calibrated camera imagery (`scratch/left_representative.jpg`).
- **All signal processing is computed live**: 6-band CSM covariance matrices, eigen-decomposition, 2D MUSIC spatial pseudo-spectrum (7,676 points), peak detection, YOLO11n-seg inference, timestamp alignment, and projection are executed dynamically on every frame.
- Marked with an amber `[SIMULATION]` status badge in the header.

---

## D. Keyboard & Mouse Controls

| Action | Control | Description |
| :--- | :--- | :--- |
| **Select Mode 1 to 9** | Keys `1` to `9` | Switches directly to presentation Modes 1 through 9. |
| **Select Mode 10** | Key `0` | Switches to Mode 10: Limitations & Research Roadmap. |
| **Tab Navigation** | Mouse Left Click | Click any of the 10 tabs across the top header bar ($y \in [34, 62]$ px). |
| **Set Ground Truth** | Mouse Click (Mode 1) | Click anywhere inside the camera video frame to set the target crosshair. |
| **Clear Ground Truth** | Key `C` | Clears the active Ground Truth target and resets error metrics. |
| **Save Snapshot** | Key `S` | Saves high-res canvas image (`.jpg`) and full structured telemetry (`.json`) to `scratch/demo_results/`. |
| **Toggle Session Record**| Key `R` | Starts/stops multi-frame session logging to structured JSON in `scratch/demo_results/`. |
| **Exit Application** | Key `Q` or `ESC` | Safely closes camera and audio threads, releases devices, and exits. |

---

## E. Explanation of Every UI Mode

### Mode 1: FUSION (Live Multimodal Fusion)
- **Display**: Live 640x480 ESP32-CAM optical stream blended with 2D MUSIC pseudo-spectrum heatmap (`colormap=INFERNO`), YOLO11n-seg instance bounding boxes, optical center crosshair $(320, 240)$, acoustic peak crosshair, and Ground Truth target marker.
- **Data Source**: Live ESP32-CAM + UMA-16 (or simulation replay).
- **Controls**: Mouse click on video viewport sets Ground Truth; `C` clears it.

### Mode 2: ACOUSTIC (2D MUSIC Spatial Spectrum)
- **Display**: Raw high-contrast 2D pseudo-spectrum heatmap plotted across the full $101 \times 76$ search grid ($40^\circ \le \text{Az} \le 140^\circ$, $-30^\circ \le \text{El} \le 30^\circ$). Overlaid with angular grid axes, peak crosshair, dynamic range, and colorbar.
- **Data Source**: Live 6-band MUSIC calculation.

### Mode 3: VISION (YOLO11n-seg Instance Segmentation)
- **Display**: Pure optical camera stream with YOLO11n-seg instance segmentation mask, bounding box, detected object class, confidence score, centroid coordinates, and inference latency (ms).
- **Data Source**: ESP32-CAM stream processed by `models/yolo11n-seg.pt`.

### Mode 4: HORIZONTAL (Guided Azimuth Benchmark Test)
- **Display**: 3-position horizontal test suite (**Left / Center / Right**). Displays current azimuth, historical project benchmarks, and real-time statistical summary (median, mean, std, min, max).
- **Controls**: Interactive buttons: `[1] LEFT`, `[2] CENTER`, `[3] RIGHT`, and `START 20-FRAME TEST`. Automatically gathers 20 frames and exports structured JSON.

### Mode 5: VERTICAL (Guided Elevation Benchmark Test)
- **Display**: 3-position vertical elevation test suite (**Above / Level / Below**). Displays current elevation, historical reflection-suppressed benchmarks, and real-time statistical summary.
- **Controls**: Interactive buttons: `[1] ABOVE`, `[2] LEVEL`, `[3] BELOW`, and `START 20-FRAME TEST`. Automatically gathers 20 frames and exports structured JSON.

### Mode 6: SYNC JITTER (Timestamp Synchronization Monitor)
- **Display**: Real-time strip-chart plotting optical vs. acoustic frame arrival time delta ($\Delta t = |t_{\text{cam}} - t_{\text{mic}}|$) against the $100$ ms synchronization tolerance. Shows rolling jitter mean, std, min, max, and 95th percentile.
- **Data Source**: Monotonic timestamps from camera stream and audio ring buffer.

### Mode 7: ARRAY & FFT (16-Mic Physical Layout & Spectral Meters)
- **Display**: Left: $4 \times 4$ physical arrangement of the 16 miniDSP UMA-16 microphones with live color-coded RMS activity heat. Right: 6 sub-band FFT energy meters ($500$, $1000$, $1500$, $2000$, $2500$, $3000$ Hz) in dBFS.
- **Data Source**: Raw multi-channel audio stream from UMA-16.

### Mode 8: BENCHMARKS (Scientific Validation Results)
- **Display**: Academic tables summarizing project milestones: 12/12 Synthetic MUSIC test pass ($\le 0.25^\circ$ discretization error), steering vector mathematical equivalence proof, and historical experimental runs (Runs 2, 4, 5, 6, 7).
- **Data Source**: Empirical benchmark database compiled from project reports.

### Mode 9: ARCHITECTURE (Multimodal System Dataflow Diagram)
- **Display**: Visual end-to-end architecture diagram showing the concurrent acoustic array branch (UMA-16 $\to$ CSM $\to$ Steering $\to$ MUSIC) and optical branch (ESP32-CAM $\to$ JPEG $\to$ YOLO11n-seg), converging at the Timestamp Synchronizer, Projection Visualizer, and OpenCV UI.
- **Data Source**: Static architectural schematic.

### Mode 10: ROADMAP (Empirical Limitations & Future Extensions)
- **Display**: Transparent evaluation of current prototype constraints (spatial resolution limits, multipath reflections, single-frequency bias) and 7 future research extension phases for an OUR project proposal.
- **Data Source**: Static academic research roadmap.

---

## F. How to Perform the CENTER Test

1. Set the acoustic source (phone playing 1.5–2.5 kHz tone or chirp) directly in front of the array at $\approx 1.5$ meters distance.
2. In the demo application, press `4` or click `[4] HORIZ TEST`.
3. Click the `[2] CENTER` button.
4. Click `START 20-FRAME TEST`.
5. Observe the progress bar accumulate 20 valid acoustic frames.
6. Verify the results:
   - Median Azimuth should be approximately $90^\circ \pm 3^\circ$.
   - A JSON test summary is automatically saved to `scratch/demo_results/horizontal_test_CENTER_<timestamp>.json`.

---

## G. How to Perform LEFT & RIGHT Tests

1. Move the sound source to the **Left** of broadside ($\approx 25^\circ$ off-center):
   - In Mode 4, click `[1] LEFT`.
   - Click `START 20-FRAME TEST`.
   - Verify: Azimuth should track between $100^\circ$ and $112^\circ$.
2. Move the sound source to the **Right** of broadside ($\approx 25^\circ$ off-center):
   - In Mode 4, click `[3] RIGHT`.
   - Click `START 20-FRAME TEST`.
   - Verify: Azimuth should track between $65^\circ$ and $75^\circ$.

---

## H. How to Perform ABOVE & BELOW Tests

1. In the demo application, press `5` or click `[5] VERT TEST`.
2. **Level Test**:
   - Place source at eye level with the camera/array.
   - Click `[2] LEVEL` $\to$ `START 20-FRAME TEST`.
   - Expected Elevation: $-5^\circ$ to $-7^\circ$ (due to physical $-6^\circ$ array pitch).
3. **Above Test**:
   - Raise source $20$–$30$ cm above broadside level.
   - Click `[1] ABOVE` $\to$ `START 20-FRAME TEST`.
   - Expected Elevation: $+3^\circ$ to $+10^\circ$.
4. **Below Test**:
   - Lower source $20$–$30$ cm below broadside level.
   - Click `[3] BELOW` $\to$ `START 20-FRAME TEST`.
   - Expected Elevation: $-12^\circ$ to $-18^\circ$.

> **Note on Reflections**: As demonstrated in forensic tests, hard desk surfaces create acoustic multipath reflections. Placing a towel or acoustic foam on the desk below the source significantly improves vertical accuracy.

---

## I. How to Use Ground-Truth Clicking

1. Switch to Mode 1 by pressing `1` or clicking `[1] FUSION`.
2. Locate the visible sound source (e.g. smartphone speaker) on the camera image.
3. Left-click directly on the center of the speaker.
4. An orange circle with label `GT (x, y)` appears at that exact pixel.
5. The application instantly renders an error line from the acoustic peak crosshair to the GT click and reports:
   - `Pixel Error (GT)` in pixels.
   - Ground truth camera coordinates $(u, v)$ in the sidebar.
6. Press `C` to clear the target point.

---

## J. How to Interpret Synchronization

In Mode 6 (`SYNC JITTER`) or the persistent sidebar:
- **Green Badge `SYNC: PASS`**: Camera frame and audio frame arrived within $\le 100.0$ ms of each other ($|\Delta t| \le 100$ ms).
- **Amber/Red Badge `SYNC: FAIL`**: Network congestion or buffer delay exceeded $100$ ms tolerance.
- **Jitter Metrics**:
  - `Mean Delta`: Average temporal offset between modalities.
  - `Std (Jitter)`: Frame-to-frame timing jitter.
  - `P95`: 95th percentile worst-case latency gap.

---

## K. How to Interpret Acoustic Azimuth & Elevation

The coordinate convention follows standard spherical broadside geometry:
- **Azimuth ($\theta$)**:
  - Broadside center: $\theta = 90.0^\circ$
  - Physical Left: $\theta > 90^\circ$ (e.g. $105^\circ$)
  - Physical Right: $\theta < 90^\circ$ (e.g. $65^\circ$)
- **Elevation ($\phi$)**:
  - Array plane level: $\phi = 0.0^\circ$
  - Physical Above: $\phi > 0^\circ$ (e.g. $+10^\circ$)
  - Physical Below: $\phi < 0^\circ$ (e.g. $-15^\circ$)
  - Default broadside level with $-6^\circ$ pitch: $\phi \approx -6.0^\circ$

---

## L. How to Interpret Projection

Acoustic direction $(\text{Az}, \text{El})$ is projected into image plane $(u, v)$ using pinhole camera optics:
$$u = \frac{W}{2} - \frac{\text{Az} - 90.0^\circ}{\text{FOV}_H} \cdot W$$
$$v = \frac{H}{2} - \frac{\text{El} - (-6.0^\circ)}{\text{FOV}_V} \cdot H$$
- Center $(u=320, v=240)$ corresponds to $\text{Az} = 90^\circ, \text{El} = -6^\circ$.
- $u < 320$: Left half of video frame.
- $u > 320$: Right half of video frame.
- $v < 240$: Upper half of video frame.
- $v > 240$: Lower half of video frame.

---

## M. Known Empirical Limitations

1. **Rayleigh Spatial Resolution Limit**: At $42$ mm microphone spacing, array aperture is $126$ mm. Angular beamwidth is $\approx 15^\circ$ at $2$ kHz.
2. **Environmental Multipath**: Acoustic reflections from desks/floors cause elevation pull toward hard reflective surfaces unless absorbed or isolated.
3. **Single Dominant Source Assumption**: Standard MUSIC rank-1 covariance assumes one primary sound source per sub-band.

---

## N. Troubleshooting Common Startup Errors

### 1. "ESP32-CAM connection deferred" / Black Camera Feed
- Verify ESP32-CAM is connected to the same Wi-Fi subnet.
- Open a browser and navigate to `http://10.243.98.165:81/stream`. If the IP has changed, run:
  ```powershell
  .venv\Scripts\python.exe demo\app.py --stream-url http://<NEW_IP>:81/stream
  ```

### 2. "UMA-16 audio stream deferred" / Audio Disconnected
- Verify the miniDSP UMA-16 is plugged into a USB port.
- Check Windows Sound Settings: input device must appear as `Line (UMA16v2)`.
- Test sounddevice detection:
  ```powershell
  .venv\Scripts\python.exe -c "import sounddevice as sd; print(sd.query_devices())"
  ```

### 3. OpenBLAS / Acoular Thread Warning
- Acoular prints a standard advisory regarding OpenBLAS and Numba threads. This is normal and does not affect operation or accuracy.
