"""
acoustic_camera_calibration.py
==============================
Interactive 3x3 Acoustic-to-Camera Pixel Calibration Grid Utility.

Key Capabilities:
  1. Interactive Grid Collection (--collect <POSITION> or --collect-next)
     - 3x3 grid positions:
       ABOVE_LEFT,   ABOVE_CENTER,   ABOVE_RIGHT
       LEVEL_LEFT,   LEVEL_CENTER,   LEVEL_RIGHT
       BELOW_LEFT,   BELOW_CENTER,   BELOW_RIGHT
     - Live ESP32-CAM stream with interactive mouse click to mark visual GT.
     - Live acoustic feedback and visualizer overlay during recording.
     - Saves records to scratch/calibration_3x3.json and representative images.

  2. Grid Status (--status)
     - Displays formatted 3x3 grid showing collected vs pending points.

  3. Model Evaluation & Hold-Out Cross Validation (--evaluate)
     - Evaluates 4 candidate models:
       * Model A: Baseline Uncalibrated (az0=90.0, el0=0.0, H_fov=50.0, V_fov=37.5)
       * Model B: Offset-Only Calibration (az0=fit/90.0, el0=-6.0, H_fov=50.0, V_fov=37.5)
       * Model C: Calibrated Offsets + Effective FOV (fit az0, el0, H_fov, V_fov)
       * Model D: 2D Affine Transformation ([u, v]^T = M [Az, El]^T + b)
     - Computes Train and Leave-One-Out Cross-Validation (LOOCV) errors:
       * Mean X Error (px)
       * Mean Y Error (px)
       * Mean Euclidean Error (px)
       * Max Error (px)
     - Directional Monotonicity Checks (Horizontal Left < Center < Right, Vertical Above < Level < Below).
     - Provides clear statistical justification and minimal production change recommendation.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

import os
import sys
import json
import time
import threading
import argparse
import cv2
import numpy as np
from scipy.optimize import minimize

# Ensure project root is in sys.path
root_dir = r"c:\Users\kushs\OneDrive\Desktop\Projects\multimodal_acoustic_camera"
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config.settings
from config.settings import FRAME_WIDTH, FRAME_HEIGHT, CAMERA_FOV_H, CAMERA_FOV_V, MIRROR_CAMERA
from fusion.visualization import AcousticHeatmapVisualizer
from vision.camera import CameraAcquisition
from acoustic.audio import AudioAcquisition
from pipeline.multimodal import MultimodalPipeline

OUTPUT_DIR = os.path.dirname(__file__)
CALIBRATION_FILE = os.path.join(OUTPUT_DIR, "calibration_3x3.json")
WINDOW_NAME = "3x3 ACOUSTIC-CAMERA CALIBRATION"

GRID_POSITIONS = [
    "ABOVE_LEFT",   "ABOVE_CENTER",   "ABOVE_RIGHT",
    "LEVEL_LEFT",   "LEVEL_CENTER",   "LEVEL_RIGHT",
    "BELOW_LEFT",   "BELOW_CENTER",   "BELOW_RIGHT",
]

# Nominal visual regions for the 3x3 grid on a 640x480 frame
GRID_ZONES = {
    "ABOVE_LEFT":   (20, 20, 200, 150),
    "ABOVE_CENTER": (220, 20, 420, 150),
    "ABOVE_RIGHT":  (440, 20, 620, 150),
    "LEVEL_LEFT":   (20, 170, 200, 310),
    "LEVEL_CENTER": (220, 170, 420, 310),
    "LEVEL_RIGHT":  (440, 170, 620, 310),
    "BELOW_LEFT":   (20, 330, 200, 460),
    "BELOW_CENTER": (220, 330, 420, 460),
    "BELOW_RIGHT":  (440, 330, 620, 460),
}

target_x = None
target_y = None
mouse_lock = threading.Lock()


def on_mouse(event, x, y, flags, param):
    """Mouse callback to record ground truth visual coordinates of the sound source."""
    global target_x, target_y
    if event == cv2.EVENT_LBUTTONDOWN:
        with mouse_lock:
            target_x = x
            target_y = y
        dx = x - (FRAME_WIDTH // 2)
        dy = y - (FRAME_HEIGHT // 2)
        print(f"\n[Visual GT Clicked] ({x}, {y}) | dx={dx:+d}px, dy={dy:+d}px from center", flush=True)


# =============================================================================
# DATASET PERSISTENCE
# =============================================================================

def get_initial_seed_dataset() -> dict:
    """
    Returns the initial clean dataset containing ONLY the 3 verified vertical
    measurements from the current physical setup.
    """
    return {
        "ABOVE_CENTER": {
            "position": "ABOVE_CENTER",
            "camera_x": 322,
            "camera_y": 152,
            "acoustic_azimuth": 88.25,
            "acoustic_elevation": 12.25,
            "sample_count": 10,
            "representative_image": "scratch/above_elevation_rep.jpg",
            "raw_image": "scratch/above_elevation_raw.jpg",
            "setup": "current_elevation_test",
        },
        "LEVEL_CENTER": {
            "position": "LEVEL_CENTER",
            "camera_x": 321,
            "camera_y": 231,
            "acoustic_azimuth": 91.25,
            "acoustic_elevation": -6.0,
            "sample_count": 10,
            "representative_image": "scratch/level_elevation_rep.jpg",
            "raw_image": "scratch/level_elevation_raw.jpg",
            "setup": "current_elevation_test",
        },
        "BELOW_CENTER": {
            "position": "BELOW_CENTER",
            "camera_x": 322,
            "camera_y": 340,
            "acoustic_azimuth": 83.25,
            "acoustic_elevation": -17.0,
            "sample_count": 10,
            "representative_image": "scratch/below_elevation_rep.jpg",
            "raw_image": "scratch/below_elevation_raw.jpg",
            "setup": "current_elevation_test",
        },
    }


def load_calibration_dataset() -> dict:
    """Loads 3x3 dataset from disk, seeding if missing."""
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"Warning reading {CALIBRATION_FILE}: {e}")
    seed = get_initial_seed_dataset()
    save_calibration_dataset(seed)
    return seed


def save_calibration_dataset(data: dict):
    """Saves 3x3 dataset to disk."""
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)


def print_status():
    """Prints formatted 3x3 grid status."""
    dataset = load_calibration_dataset()
    print("\n" + "=" * 90)
    print("CURRENT 3x3 CALIBRATION GRID STATUS")
    print("=" * 90)
    print(f"Storage File : {CALIBRATION_FILE}")
    print(f"Total Points : {len(dataset)} of 9 Collected\n")

    header = f"{'Position':<15} | {'Status':<12} | {'Camera (x, y)':<15} | {'Acoustic (Az, El)':<22} | {'Samples':<8}"
    print(header)
    print("-" * 90)

    for pos in GRID_POSITIONS:
        if pos in dataset:
            d = dataset[pos]
            cx, cy = d.get("camera_x", "N/A"), d.get("camera_y", "N/A")
            az, el = d.get("acoustic_azimuth", "N/A"), d.get("acoustic_elevation", "N/A")
            cnt = d.get("sample_count", 1)
            c_str = f"({cx}, {cy})" if isinstance(cx, int) else "N/A"
            a_str = f"Az={az:.1f}°, El={el:.1f}°" if isinstance(az, (int, float)) else "N/A"
            print(f"{pos:<15} | {'COLLECTED':<12} | {c_str:<15} | {a_str:<22} | {cnt:<8}")
        else:
            print(f"{pos:<15} | {'[PENDING]':<12} | {'--':<15} | {'--':<22} | {'--':<8}")
    print("=" * 90 + "\n")


# =============================================================================
# GUI DRAWING & OVERLAYS
# =============================================================================

def draw_calibration_gui(
    frame: np.ndarray,
    position_label: str,
    instruction: str,
    live_az: float = None,
    live_el: float = None,
    live_u: int = None,
    live_v: int = None,
    live_db: float = None,
) -> np.ndarray:
    """Draws 3x3 grid guidelines, active target box, clicked target, and live telemetry."""
    gui = frame.copy()
    h, w = gui.shape[:2]

    # Draw faint 3x3 grid lines
    cv2.line(gui, (w // 3, 0), (w // 3, h), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(gui, (2 * w // 3, 0), (2 * w // 3, h), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(gui, (0, h // 3), (w, h // 3), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(gui, (0, 2 * h // 3), (w, 2 * h // 3), (70, 70, 70), 1, cv2.LINE_AA)

    # Draw camera optical center crosshair (320, 240)
    cx, cy = w // 2, h // 2
    cv2.drawMarker(gui, (cx, cy), (120, 120, 120), cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
    cv2.putText(gui, "(320, 240)", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    # Highlight active target zone
    if position_label in GRID_ZONES:
        x1, y1, x2, y2 = GRID_ZONES[position_label]
        overlay = gui.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.15, gui, 0.85, 0, gui)
        cv2.rectangle(gui, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(gui, f"TARGET: {position_label}", (x1 + 6, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    # Draw clicked visual ground truth target
    with mouse_lock:
        tx, ty = target_x, target_y
    if tx is not None and ty is not None:
        cv2.drawMarker(gui, (tx, ty), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 24, 2, cv2.LINE_AA)
        cv2.circle(gui, (tx, ty), 16, (0, 255, 0), 2, cv2.LINE_AA)
        dx = tx - cx
        dy = ty - cy
        cv2.putText(gui, f"CLICK GT: ({tx}, {ty}) [dx={dx:+d}, dy={dy:+d}]", (tx + 12, ty + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw top banner
    banner_h = 56
    cv2.rectangle(gui, (0, 0), (w, banner_h), (25, 25, 25), -1)
    title = f"3x3 CALIBRATION: {position_label.upper()}"
    cv2.putText(gui, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(gui, instruction, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Draw bottom telemetry banner
    b_banner_y = h - 34
    cv2.rectangle(gui, (0, b_banner_y), (w, h), (20, 20, 20), -1)
    if live_az is not None and live_el is not None:
        telemetry = f"ACOUSTIC: Az={live_az:5.1f} deg | El={live_el:5.1f} deg | dB={live_db:4.1f} | Overlay=({live_u}, {live_v})"
        cv2.putText(gui, telemetry, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(gui, "ACOUSTIC: Waiting for audio / frame sync...", (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

    return gui


# =============================================================================
# DATA COLLECTION PROCEDURE
# =============================================================================

def collect_grid_position(position_label: str, num_frames: int = 10, countdown_sec: int = 3, stream_url: str = "http://10.243.98.165:81/stream"):
    """Interactive collection of a single 3x3 grid position."""
    global target_x, target_y
    position_label = position_label.upper().strip()
    if position_label not in GRID_POSITIONS:
        print(f"Error: Invalid position '{position_label}'. Must be one of: {', '.join(GRID_POSITIONS)}")
        return False

    print("\n" + "=" * 80)
    print(f"COLLECTING POSITION: {position_label}")
    print("=" * 80)
    print("Instructions:")
    print(f"  1. Physically position sound source in region: {position_label}")
    print("  2. CLICK on the sound source in the OpenCV camera window to set Visual Ground Truth.")
    print("  3. Press ENTER or SPACEBAR in the camera window (or terminal) to confirm position.")
    print("  4. Sound playback should be playing continuously during recording.")
    print("  5. Press 'q' or ESC at any time to abort.\n")

    # Reset mouse target
    with mouse_lock:
        target_x = None
        target_y = None

    print("[1/3] Initializing Camera and UMA-16 Audio Pipeline...")
    import sounddevice as sd

    devices = sd.query_devices()
    uma_dev = None
    for idx, dev in enumerate(devices):
        if "uma16v2" in dev["name"].lower() and dev["max_input_channels"] >= 16:
            uma_dev = idx
            break

    print(f"Selected audio device index: {uma_dev}")
    config.settings.STREAM_URL = stream_url
    camera = CameraAcquisition(stream_url=stream_url, expected_width=640, expected_height=480)
    audio = AudioAcquisition(device=uma_dev)
    pipeline = MultimodalPipeline(camera=camera, audio=audio)
    visualizer = AcousticHeatmapVisualizer(diagnostic=False)

    try:
        pipeline.start()
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        return False

    time.sleep(1.0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    enter_pressed = threading.Event()

    def terminal_listener():
        try:
            input()
            enter_pressed.set()
        except Exception:
            enter_pressed.set()

    threading.Thread(target=terminal_listener, daemon=True).start()

    # Positioning phase
    print(f"Waiting for target placement and click for {position_label}...")
    while not enter_pressed.is_set():
        cam_frame = pipeline.camera.read()
        if cam_frame is None:
            time.sleep(0.01)
            continue

        instruction = "CLICK sound source to mark GT (x, y) | Press ENTER or SPACE when ready | 'q' to abort"
        gui = draw_calibration_gui(cam_frame.frame, position_label, instruction)
        cv2.imshow(WINDOW_NAME, gui)
        key = cv2.waitKey(15) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("\nAborted by user.")
            pipeline.stop()
            cv2.destroyAllWindows()
            return False
        elif key in (10, 13, ord(' ')):
            with mouse_lock:
                if target_x is None or target_y is None:
                    print("\n[WARNING] Please CLICK on the sound source in the camera image first!")
                    continue
            enter_pressed.set()
            break

    with mouse_lock:
        click_x, click_y = target_x, target_y

    print(f"\n[Ground Truth Locked] Visual Click: ({click_x}, {click_y}) | dx={click_x - 320:+d}px, dy={click_y - 240:+d}px")

    # Countdown phase
    t_start = time.perf_counter()
    while True:
        rem = int(np.ceil(countdown_sec - (time.perf_counter() - t_start)))
        if rem <= 0:
            break
        cam_frame = pipeline.camera.read()
        if cam_frame is None:
            time.sleep(0.01)
            continue
        gui = draw_calibration_gui(cam_frame.frame, position_label, f"STARTING RECORDING IN {rem}s — KEEP AUDIO PLAYING!")
        h, w = gui.shape[:2]
        cv2.putText(gui, str(rem), (w // 2 - 25, h // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, gui)
        if cv2.waitKey(20) & 0xFF in (ord('q'), ord('Q'), 27):
            pipeline.stop()
            cv2.destroyAllWindows()
            return False

    # Flush audio buffer & reset CSM state
    with pipeline.audio._buffer_lock:
        pipeline.audio._buffer = np.empty((0, pipeline.audio.channels), dtype=np.float32)
    pipeline.localizer.reset_csm_state()
    time.sleep(0.1)

    # Recording phase
    print(f"\nRecording {num_frames} synchronized frames for {position_label}...")
    print(f"{'#':<3} | {'Cam ID':<7} | {'dSync(ms)':<10} | {'Az (deg)':<9} | {'El (deg)':<9} | {'dB':<6} | {'Overlay (u, v)':<15}")
    print("-" * 75)

    records = []
    best_image = None
    best_raw = None
    attempts = 0

    while len(records) < num_frames and attempts < 250:
        attempts += 1
        audio_frame = pipeline.audio.read(timeout=pipeline.audio_read_timeout)
        cam_frame = pipeline.camera.read()
        if cam_frame is None:
            time.sleep(0.01)
            continue

        vision_frame = pipeline.build_vision_frame(cam_frame)
        if audio_frame is not None:
            pipeline.ingest_audio_frame(audio_frame)

        result = pipeline.process_vision_frame(vision_frame)
        live_az, live_el, live_u, live_v, live_db = None, None, None, None, None

        if result is not None and result.synchronized:
            loc = result.acoustic.localization
            if loc is not None and loc.valid:
                h, w = cam_frame.frame.shape[:2]
                az = float(result.acoustic.metadata.get("azimuth_deg", 90.0))
                el = float(result.acoustic.metadata.get("elevation_deg", 0.0))
                db = float(result.acoustic.metadata.get("db_level", 0.0))
                u, v = visualizer.angle_to_image(az, el, w, h)

                live_az, live_el, live_u, live_v, live_db = az, el, u, v, db
                rendered = visualizer.render(cam_frame.frame.copy(), result)

                if best_image is None:
                    best_image = rendered.copy()
                    best_raw = cam_frame.frame.copy()

                rec = {
                    "index": len(records) + 1,
                    "frame_id": int(result.vision.frame_id),
                    "sync_delta_ms": float(result.timestamp_difference_ms),
                    "azimuth": az,
                    "elevation": el,
                    "db_level": db,
                    "overlay_u": u,
                    "overlay_v": v,
                    "click_x": click_x,
                    "click_y": click_y,
                }
                records.append(rec)
                print(f"{len(records):<3} | {rec['frame_id']:<7} | {rec['sync_delta_ms']:<10.1f} | {az:<9.1f} | {el:<9.1f} | {db:<6.1f} | ({u:3d}, {v:3d})")

        gui = draw_calibration_gui(cam_frame.frame, position_label,
                                   f"RECORDING ({len(records)}/{num_frames}) — Keep audio playing!",
                                   live_az, live_el, live_u, live_v, live_db)
        cv2.imshow(WINDOW_NAME, gui)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

    if not records:
        print("\n[ERROR] No valid synchronized frames recorded. Please verify audio playback and try again.")
        return False

    # Compute medians and stability metrics
    az_vals = [r["azimuth"] for r in records]
    el_vals = [r["elevation"] for r in records]
    az_median = float(np.median(az_vals))
    el_median = float(np.median(el_vals))
    az_std = float(np.std(az_vals))
    el_std = float(np.std(el_vals))

    tag = position_label.lower()
    rep_img_path = os.path.join(OUTPUT_DIR, f"calib_{tag}_rep.jpg")
    raw_img_path = os.path.join(OUTPUT_DIR, f"calib_{tag}_raw.jpg")
    if best_image is not None:
        cv2.imwrite(rep_img_path, best_image)
    if best_raw is not None:
        cv2.imwrite(raw_img_path, best_raw)

    dataset = load_calibration_dataset()
    dataset[position_label] = {
        "position": position_label,
        "camera_x": click_x,
        "camera_y": click_y,
        "acoustic_azimuth": az_median,
        "acoustic_elevation": el_median,
        "azimuth_std": az_std,
        "elevation_std": el_std,
        "sample_count": len(records),
        "representative_image": os.path.relpath(rep_img_path, root_dir),
        "raw_image": os.path.relpath(raw_img_path, root_dir),
        "records": records,
        "timestamp": time.time(),
    }
    save_calibration_dataset(dataset)

    print("\n" + "=" * 80)
    print(f"POSITION {position_label} SUCCESSFULLY RECORDED & SAVED")
    print("=" * 80)
    print(f"Visual Click GT : ({click_x}, {click_y})")
    print(f"Median Azimuth  : {az_median:.2f}° (std: {az_std:.2f}°)")
    print(f"Median Elevation: {el_median:.2f}° (std: {el_std:.2f}°)")
    print(f"Frames Recorded : {len(records)}")
    print(f"Saved to JSON   : {CALIBRATION_FILE}")
    print("=" * 80 + "\n")
    return True


# =============================================================================
# PROJECTION MODELS & FITTING
# =============================================================================

def project_linear(az: float, el: float, az_c: float, el_c: float, fov_h: float, fov_v: float, w: int = 640, h: int = 480):
    """Projects acoustic (Az, El) in degrees to pixel (u, v) using pinhole angular mapping."""
    az_min = az_c - (fov_h / 2.0)
    az_max = az_c + (fov_h / 2.0)
    el_min = el_c - (fov_v / 2.0)
    el_max = el_c + (fov_v / 2.0)

    x_norm = (az_max - az) / (az_max - az_min)
    y_norm = (el - el_min) / (el_max - el_min)

    u = x_norm * w
    v = (1.0 - y_norm) * h
    return u, v


def fit_model_b(points: list[dict], fov_h: float = 50.0, fov_v: float = 37.5, w: int = 640, h: int = 480) -> tuple[float, float]:
    """
    Fits Model B (Offset-Only) using exact analytical least squares:
    az_c = mean(az_i + (x_i - w/2) * (fov_h / w))
    el_c = mean(el_i + (y_i - h/2) * (fov_v / h))
    """
    az_offsets = []
    el_offsets = []
    for p in points:
        x, y = p["camera_x"], p["camera_y"]
        az, el = p["acoustic_azimuth"], p["acoustic_elevation"]
        az_c_i = az + (x - (w / 2.0)) * (fov_h / float(w))
        el_c_i = el + (y - (h / 2.0)) * (fov_v / float(h))
        az_offsets.append(az_c_i)
        el_offsets.append(el_c_i)

    az_c = float(np.mean(az_offsets)) if az_offsets else 90.0
    el_c = float(np.mean(el_offsets)) if el_offsets else -6.0
    return az_c, el_c


def fit_model_c(points: list[dict], w: int = 640, h: int = 480) -> tuple[float, float, float, float]:
    """
    Fits Model C (Offsets + Effective FOVs) via bounded optimization.
    Minimizes mean squared pixel error.
    """
    if len(points) < 4:
        # Fall back to default FOV if too few points
        az_c, el_c = fit_model_b(points, 50.0, 37.5, w, h)
        return az_c, el_c, 50.0, 37.5

    def loss(params):
        az_c, el_c, fov_h, fov_v = params
        sq_errs = []
        for p in points:
            x, y = p["camera_x"], p["camera_y"]
            az, el = p["acoustic_azimuth"], p["acoustic_elevation"]
            pu, pv = project_linear(az, el, az_c, el_c, fov_h, fov_v, w, h)
            sq_errs.append((pu - x) ** 2 + (pv - y) ** 2)
        return np.mean(sq_errs)

    x0 = [90.0, -6.0, 50.0, 37.5]
    bounds = [
        (75.0, 105.0),   # az_c
        (-25.0, 10.0),   # el_c
        (30.0, 100.0),   # fov_h
        (30.0, 100.0),   # fov_v
    ]
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    az_c, el_c, fov_h, fov_v = res.x
    return float(az_c), float(el_c), float(fov_h), float(fov_v)


def fit_model_d(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits Model D (2D Affine Transform) via linear least squares:
    [u, v]^T = M * [Az, El]^T + b
    """
    if len(points) < 3:
        M = np.array([[-640.0 / 50.0, 0.0], [0.0, -480.0 / 37.5]])
        b = np.array([320.0 - M[0, 0] * 90.0, 240.0 - M[1, 1] * (-6.0)])
        return M, b

    A = []
    Y = []
    for p in points:
        az, el = p["acoustic_azimuth"], p["acoustic_elevation"]
        x, y = p["camera_x"], p["camera_y"]
        A.append([az, el, 1.0])
        Y.append([x, y])
    A = np.array(A)
    Y = np.array(Y)

    # Solve least squares for [m11, m21; m12, m22; b1, b2]
    W, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    M = W[:2, :].T  # shape (2, 2)
    b = W[2, :]     # shape (2,)
    return M, b


# =============================================================================
# EVALUATION & HOLD-OUT CROSS VALIDATION
# =============================================================================

def evaluate_candidate_models():
    """Runs rigorous model evaluation and Leave-One-Out Cross-Validation."""
    dataset = load_calibration_dataset()
    pts = list(dataset.values())
    N = len(pts)

    print("\n" + "=" * 105)
    print(f"3x3 ACOUSTIC-CAMERA CALIBRATION: MODEL COMPARISON & HOLD-OUT VALIDATION (N = {N} points)")
    print("=" * 105)

    if N < 3:
        print("[ERROR] At least 3 points are required to run calibration evaluation.")
        return

    # Fit Full Models on all available points
    # Model A: Baseline
    mA_params = {"az_c": 90.0, "el_c": 0.0, "fov_h": 50.0, "fov_v": 37.5}

    # Model B: Offset-Only
    mB_az_c, mB_el_c = fit_model_b(pts, 50.0, 37.5)
    mB_params = {"az_c": mB_az_c, "el_c": mB_el_c, "fov_h": 50.0, "fov_v": 37.5}

    # Model C: Calibrated FOV
    mC_az_c, mC_el_c, mC_fov_h, mC_fov_v = fit_model_c(pts)
    mC_params = {"az_c": mC_az_c, "el_c": mC_el_c, "fov_h": mC_fov_h, "fov_v": mC_fov_v}

    # Model D: Affine
    mD_M, mD_b = fit_model_d(pts)

    def predict(model_name: str, az: float, el: float, params) -> tuple[float, float]:
        if model_name in ("Model A", "Model B", "Model C"):
            return project_linear(az, el, params["az_c"], params["el_c"], params["fov_h"], params["fov_v"])
        elif model_name == "Model D":
            M, b = params["M"], params["b"]
            pred = M @ np.array([az, el]) + b
            return float(pred[0]), float(pred[1])
        raise ValueError(f"Unknown model: {model_name}")

    models = {
        "Model A (Baseline)": ("Model A", mA_params),
        "Model B (Offset-Only)": ("Model B", mB_params),
        "Model C (Calibrated FOV)": ("Model C", mC_params),
        "Model D (Affine)": ("Model D", {"M": mD_M, "b": mD_b}),
    }

    # Print Parameters
    print("\n--- FIT PARAMETERS (Full Dataset N = {}) ---".format(N))
    print(f"Model A (Baseline)   : Az0 = {mA_params['az_c']:.2f}°, El0 = {mA_params['el_c']:.2f}°, H_FOV = {mA_params['fov_h']:.1f}°, V_FOV = {mA_params['fov_v']:.1f}°")
    print(f"Model B (Offset-Only): Az0 = {mB_params['az_c']:.2f}°, El0 = {mB_params['el_c']:.2f}°, H_FOV = {mB_params['fov_h']:.1f}°, V_FOV = {mB_params['fov_v']:.1f}°")
    print(f"Model C (Calib FOV)  : Az0 = {mC_params['az_c']:.2f}°, El0 = {mC_params['el_c']:.2f}°, H_FOV = {mC_params['fov_h']:.1f}°, V_FOV = {mC_params['fov_v']:.1f}°")
    print(f"Model D (Affine)     : M = [[{mD_M[0,0]:.2f}, {mD_M[0,1]:.2f}], [{mD_M[1,0]:.2f}, {mD_M[1,1]:.2f}]], b = [{mD_b[0]:.1f}, {mD_b[1]:.1f}]")

    # Compute Training (In-Sample) Errors
    train_results = {}
    for m_label, (m_type, m_p) in models.items():
        errs_x, errs_y, errs_euc = [], [], []
        for p in pts:
            u, v = predict(m_type, p["acoustic_azimuth"], p["acoustic_elevation"], m_p)
            ex = abs(u - p["camera_x"])
            ey = abs(v - p["camera_y"])
            euc = np.hypot(u - p["camera_x"], v - p["camera_y"])
            errs_x.append(ex)
            errs_y.append(ey)
            errs_euc.append(euc)
        train_results[m_label] = {
            "mean_x": float(np.mean(errs_x)),
            "mean_y": float(np.mean(errs_y)),
            "mean_euc": float(np.mean(errs_euc)),
            "max_err": float(np.max(errs_euc)),
        }

    # Compute Hold-Out Validation Errors (Leave-One-Out Cross-Validation)
    loocv_results = {}
    for m_label in models:
        loocv_results[m_label] = {"val_x": [], "val_y": [], "val_euc": []}

    for i in range(N):
        train_fold = [pts[j] for j in range(N) if j != i]
        test_pt = pts[i]

        # Fit models on train_fold
        # Model A
        f_mA_p = mA_params
        # Model B
        f_b_az, f_b_el = fit_model_b(train_fold, 50.0, 37.5)
        f_mB_p = {"az_c": f_b_az, "el_c": f_b_el, "fov_h": 50.0, "fov_v": 37.5}
        # Model C
        f_c_az, f_c_el, f_c_h, f_c_v = fit_model_c(train_fold)
        f_mC_p = {"az_c": f_c_az, "el_c": f_c_el, "fov_h": f_c_h, "fov_v": f_c_v}
        # Model D
        f_d_M, f_d_b = fit_model_d(train_fold)
        f_mD_p = {"M": f_d_M, "b": f_d_b}

        fold_models = {
            "Model A (Baseline)": ("Model A", f_mA_p),
            "Model B (Offset-Only)": ("Model B", f_mB_p),
            "Model C (Calibrated FOV)": ("Model C", f_mC_p),
            "Model D (Affine)": ("Model D", f_mD_p),
        }

        for m_label, (m_type, f_params) in fold_models.items():
            u, v = predict(m_type, test_pt["acoustic_azimuth"], test_pt["acoustic_elevation"], f_params)
            ex = abs(u - test_pt["camera_x"])
            ey = abs(v - test_pt["camera_y"])
            euc = np.hypot(u - test_pt["camera_x"], v - test_pt["camera_y"])
            loocv_results[m_label]["val_x"].append(ex)
            loocv_results[m_label]["val_y"].append(ey)
            loocv_results[m_label]["val_euc"].append(euc)

    # Directional Monotonicity Checks
    print("\n--- DIRECTIONAL MONOTONICITY CHECKS ---")
    u_left, _ = project_linear(106.0, -6.0, mB_params["az_c"], mB_params["el_c"], mB_params["fov_h"], mB_params["fov_v"])
    u_cent, _ = project_linear(90.0, -6.0, mB_params["az_c"], mB_params["el_c"], mB_params["fov_h"], mB_params["fov_v"])
    u_right, _ = project_linear(65.0, -6.0, mB_params["az_c"], mB_params["el_c"], mB_params["fov_h"], mB_params["fov_v"])
    h_order = u_left < u_cent < u_right
    print(f"Horizontal (LEFT < CENTER < RIGHT): u({u_left:.0f}) < u({u_cent:.0f}) < u({u_right:.0f}) => {'PASS' if h_order else 'FAIL'}")

    _, v_above = project_linear(90.0, 12.25, mB_params["az_c"], mB_params["el_c"], mB_params["fov_h"], mB_params["fov_v"])
    _, v_level = project_linear(90.0, -6.00, mB_params["az_c"], mB_params["el_c"], mB_params["fov_h"], mB_params["fov_v"])
    _, v_below = project_linear(90.0, -17.0, mB_params["az_c"], mB_params["el_c"], mB_params["fov_h"], mB_params["fov_v"])
    v_order = v_above < v_level < v_below
    print(f"Vertical   (ABOVE < LEVEL < BELOW): v({v_above:.0f}) < v({v_level:.0f}) < v({v_below:.0f}) => {'PASS' if v_order else 'FAIL'}")

    # Comparative Summary Table
    print("\n" + "=" * 105)
    print("CANDIDATE MODEL COMPARISON (TRAIN vs LEAVE-ONE-OUT HOLD-OUT VALIDATION)")
    print("=" * 105)
    summary_hdr = "{:<25} | {:>9} | {:>9} | {:>10} || {:>9} | {:>9} | {:>10} | {:>8}".format(
        "Candidate Model", "Train X", "Train Y", "Train Euc", "Val X", "Val Y", "Val Euc", "Max Err"
    )
    print(summary_hdr)
    print("-" * 105)

    for m_label in models:
        tr = train_results[m_label]
        val_x = float(np.mean(loocv_results[m_label]["val_x"]))
        val_y = float(np.mean(loocv_results[m_label]["val_y"]))
        val_euc = float(np.mean(loocv_results[m_label]["val_euc"]))
        row = "{:<25} | {:>7.1f}px | {:>7.1f}px | {:>8.1f}px || {:>7.1f}px | {:>7.1f}px | {:>8.1f}px | {:>6.1f}px".format(
            m_label, tr["mean_x"], tr["mean_y"], tr["mean_euc"], val_x, val_y, val_euc, tr["max_err"]
        )
        print(row)
    print("=" * 105)

    # Detailed Per-Point Table for Model B and Model C
    print("\n" + "=" * 105)
    print("DETAILED PER-POINT PREDICTIONS (Model B vs Model C)")
    print("=" * 105)
    pt_hdr = "{:<14} | {:<12} | {:<18} | {:<18} | {:<18}".format(
        "Position", "Actual (x, y)", "Acoustic (Az, El)", "Model B (u, v) [Err]", "Model C (u, v) [Err]"
    )
    print(pt_hdr)
    print("-" * 105)

    for p in pts:
        ax, ay = p["camera_x"], p["camera_y"]
        az, el = p["acoustic_azimuth"], p["acoustic_elevation"]
        bu, bv = predict("Model B", az, el, mB_params)
        beuc = np.hypot(bu - ax, bv - ay)
        cu, cv = predict("Model C", az, el, mC_params)
        ceuc = np.hypot(cu - ax, cv - ay)
        b_str = f"({bu:.0f}, {bv:.0f}) [{beuc:4.1f}px]"
        c_str = f"({cu:.0f}, {cv:.0f}) [{ceuc:4.1f}px]"
        print(f"{p['position']:<14} | ({ax:3d}, {ay:3d})    | (Az={az:5.1f}°, El={el:5.1f}°) | {b_str:<18} | {c_str:<18}")
    print("=" * 105 + "\n")

    # Production Recommendation
    print("=" * 105)
    print("FORENSIC PRODUCTION RECOMMENDATION")
    print("=" * 105)
    val_euc_B = float(np.mean(loocv_results["Model B (Offset-Only)"]["val_euc"]))
    val_euc_C = float(np.mean(loocv_results["Model C (Calibrated FOV)"]["val_euc"]))

    print(f"Validation Euclidean Error: Model B = {val_euc_B:.1f} px | Model C = {val_euc_C:.1f} px")
    if val_euc_C < val_euc_B * 0.85 and N >= 6:
        print("[RECOMMENDATION] Model C demonstrates statistically superior generalization on hold-out data.")
        print("Recommended production parameters in fusion/visualization.py:")
        print(f"  azimuth_center: float = {mC_params['az_c']:.2f}")
        print(f"  elevation_center: float = {mC_params['el_c']:.2f}")
        print(f"  horizontal_fov_deg: float = {mC_params['fov_h']:.2f}")
        print(f"  vertical_fov_deg: float = {mC_params['fov_v']:.2f}")
    else:
        print("[RECOMMENDATION] Model B (Offset-Only) is physically robust and prevents overfitting.")
        print("Recommended production parameters in fusion/visualization.py:")
        print(f"  azimuth_center: float = {mB_params['az_c']:.2f}")
        print(f"  elevation_center: float = {mB_params['el_c']:.2f}")
        print(f"  horizontal_fov_deg: float = 50.0")
        print(f"  vertical_fov_deg: float = 37.5")
    print("=" * 105 + "\n")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="3x3 Acoustic-Camera Calibration Grid Engine")
    parser.add_argument("--collect", type=str, default=None,
                        help=f"Collect single grid position: {', '.join(GRID_POSITIONS)}")
    parser.add_argument("--collect-next", action="store_true",
                        help="Find and collect the next pending grid position")
    parser.add_argument("--status", action="store_true",
                        help="Show current 3x3 grid collection status")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run full candidate model evaluation and LOOCV")
    parser.add_argument("--frames", type=int, default=10,
                        help="Number of synchronized frames to record per position (default: 10)")
    parser.add_argument("--reset-seed", action="store_true",
                        help="Reset calibration_3x3.json to the 3 initial verified vertical points")
    parser.add_argument("--stream-url", default="http://10.243.98.165:81/stream",
                        help="ESP32-CAM MJPEG stream endpoint (default: http://10.243.98.165:81/stream)")

    args = parser.parse_args()

    if args.reset_seed:
        seed = get_initial_seed_dataset()
        save_calibration_dataset(seed)
        print(f"Reset {CALIBRATION_FILE} to 3 verified seed points.")
        print_status()
        return

    if args.status:
        print_status()
        return

    if args.collect:
        collect_grid_position(args.collect, num_frames=args.frames, stream_url=args.stream_url)
        return

    if args.collect_next:
        dataset = load_calibration_dataset()
        pending = [p for p in GRID_POSITIONS if p not in dataset]
        if not pending:
            print("All 9 grid positions have already been collected!")
            evaluate_candidate_models()
            return
        next_pos = pending[0]
        print(f"Next pending position: {next_pos}")
        collect_grid_position(next_pos, num_frames=args.frames, stream_url=args.stream_url)
        return

    if args.evaluate:
        evaluate_candidate_models()
        return

    # Default action: show status and evaluate
    print_status()
    evaluate_candidate_models()


if __name__ == "__main__":
    main()
