# Multimodal Acoustic Camera — Professor Demo Quick Start

Quick reference for launching and presenting the Real-Time Multimodal Acoustic Camera faculty demonstration.

---

## 1. Quick Launch Commands

### Option A: Live Hardware Mode (ESP32-CAM + UMA-16)
```powershell
.venv\Scripts\python.exe demo\app.py
```
*If your ESP32-CAM stream URL has changed:*
```powershell
.venv\Scripts\python.exe demo\app.py --stream-url http://10.243.98.165:81/stream
```

### Option B: Offline Simulation Mode (No Hardware Required)
```powershell
.venv\Scripts\python.exe demo\app.py --simulate
```
*Replays authentic 16-channel 48 kHz recordings with dynamic live MUSIC, YOLO, and synchronization computation.*

### Option C: Quick Headless Software Sanity Check
```powershell
.venv\Scripts\python.exe demo\app.py --simulate --headless-frames 5
```
*Processes exactly 5 frames in memory and safely shuts down without opening a GUI window.*

---

## 2. Interactive Hotkeys

| Key | Mode / Action | Description |
| :---: | :--- | :--- |
| `1` | `[1] FUSION` | Live fused camera feed, acoustic heatmap, YOLO boxes, GT error |
| `2` | `[2] 2D MUSIC` | 2D Spatial Pseudo-Spectrum heatmap ($101 \times 76$ grid, 7,676 points) |
| `3` | `[3] YOLO SEG` | YOLO11n-seg instance segmentation mask, confidence, latency |
| `4` | `[4] HORIZ TEST` | Guided 3-position horizontal test (Left, Center, Right) |
| `5` | `[5] VERT TEST` | Guided 3-position vertical elevation test (Above, Level, Below) |
| `6` | `[6] SYNC JITTER`| Cross-modal synchronization strip-chart & rolling jitter stats |
| `7` | `[7] ARRAY & FFT`| 16-mic physical array layout + 6-band spectral power meters |
| `8` | `[8] BENCHMARKS` | Scientific validation benchmark tables & historical run summaries |
| `9` | `[9] ARCHITECTURE`| System architecture and data flow block diagram |
| `0` | `[0] ROADMAP` | Empirical limitations & 7-phase OUR research roadmap |
| `S` | Snapshot | Save current image (`.jpg`) and telemetry (`.json`) to `scratch/demo_results/` |
| `R` | Session Record | Toggle continuous telemetry session recording on/off |
| `C` | Clear Target | Clear Ground Truth target point in Mode 1 |
| `Q` / `ESC` | Quit | Graceful shutdown and hardware resource release |

---

## 3. Mouse Interactions

- **Tab Buttons**: Click any of the 10 header tabs to switch modes.
- **Ground Truth Target** (*Mode 1*): Click anywhere on the camera viewport to place an orange target crosshair. The application instantly computes the Euclidean pixel distance error:
  $$\Delta_{\text{px}} = \sqrt{(u_{\text{acoustic}} - u_{\text{GT}})^2 + (v_{\text{acoustic}} - v_{\text{GT}})^2}$$
- **Automated Tests** (*Modes 4 & 5*): Click position buttons (`[1] LEFT`, `[2] CENTER`, `[3] RIGHT` or `[1] ABOVE`, `[2] LEVEL`, `[3] BELOW`) and click `START 20-FRAME TEST`. Results automatically save to `scratch/demo_results/`.

---

## 4. Documentation

For detailed step-by-step experiment instructions and troubleshooting, see [MANUAL_TESTING.md](../MANUAL_TESTING.md).
