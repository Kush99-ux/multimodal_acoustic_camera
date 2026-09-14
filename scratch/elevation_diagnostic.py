"""
elevation_diagnostic.py
=======================
Diagnostic script for investigating acoustic elevation estimation and camera projection:
  Part 1: Synthetic Elevation Sweep (-15°, -10°, -5°, 0°, +5°, +10°, +15°)
  Part 2: Acoular Coordinate System & Physical Convention Inspection
  Part 3: Camera-Array Extrinsic & Projection Verification
  Part 4: Real-Hardware Elevation Measurement Procedure (Level, Above, Below)
          Includes 2D heatmap export, vertical elevation profile analysis, and image saving.

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

# Ensure project root is in sys.path
root_dir = r"c:\Users\kushs\OneDrive\Desktop\Projects\multimodal_acoustic_camera"
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config.settings
from config.settings import FRAME_WIDTH, FRAME_HEIGHT, CAMERA_FOV_H, CAMERA_FOV_V, MIRROR_CAMERA
from acoustic.steering import SteeringFactory, ORIGINAL_MIC_RAW, build_microphone_positions
from acoustic.music import MusicLocalizer
from fusion.visualization import AcousticHeatmapVisualizer
from vision.camera import CameraAcquisition
from acoustic.audio import AudioAcquisition
from pipeline.multimodal import MultimodalPipeline

OUTPUT_DIR = os.path.dirname(__file__)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "elevation_results.json")
WINDOW_NAME = "ELEVATION DIAGNOSTIC LIVE CAMERA"

target_x = None
target_y = None
mouse_lock = threading.Lock()


def on_mouse(event, x, y, flags, param):
    """OpenCV mouse callback for interactive sound source targeting."""
    global target_x, target_y
    if event == cv2.EVENT_LBUTTONDOWN:
        with mouse_lock:
            target_x = x
            target_y = y
        print(f"\n[Clicked Target] x = {x}, y = {y} (dx={x - 320:+d}px, dy={y - 240:+d}px)", flush=True)


# =============================================================================
# PART 1: SYNTHETIC ELEVATION SWEEP
# =============================================================================

def run_synthetic_elevation_sweep():
    """
    Run a synthetic broadband elevation sweep to test if MUSIC/steering can
    accurately recover elevation across the configured grid.
    """
    print("\n" + "=" * 80)
    print("PART 1: SYNTHETIC BROADBAND ELEVATION SWEEP")
    print("=" * 80)

    factory = SteeringFactory()
    factory.precompute_matrices()
    localizer = MusicLocalizer()

    fs = 48000
    frame_size = 4096
    c = 343.0
    t = np.arange(frame_size) / fs

    # 6 sub-band center frequencies
    tones = [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0]

    test_cases = [
        # (Azimuth, Elevation, Distance)
        (90.0, -15.0, 1.5),
        (90.0, -10.0, 1.5),
        (90.0,  -5.0, 1.5),
        (90.0,   0.0, 1.5),
        (90.0,  +5.0, 1.5),
        (90.0, +10.0, 1.5),
        (90.0, +15.0, 1.5),
        # Off-center azimuth test cases
        (80.0, -10.0, 1.5),
        (80.0, +10.0, 1.5),
        (100.0, -10.0, 1.5),
        (100.0, +10.0, 1.5),
    ]

    print(f"\n{'True Az':>8} | {'True El':>8} | {'Dist (m)':>8} | {'Rec Az':>8} | {'Rec El':>8} | {'Az Err':>8} | {'El Err':>8} | {'Peak Val':>9} | {'Status':>8}")
    print("-" * 92)

    results = []
    all_passed = True

    for test_az, test_el, r_src in test_cases:
        az_rad = np.deg2rad(test_az)
        el_rad = np.deg2rad(test_el)

        # Source position in Acoular Cartesian coordinates (X-Z plane, +Y broadside)
        x_s = r_src * np.cos(el_rad) * np.cos(az_rad)
        y_s = r_src * np.cos(el_rad) * np.sin(az_rad)
        z_s = r_src * np.sin(el_rad)
        src_pos = np.array([x_s, y_s, z_s])

        mic_pos = factory.microphone_geometry.pos_total  # shape (3, 16)
        dist = np.linalg.norm(src_pos[:, None] - mic_pos, axis=0)  # shape (16,)
        dist_0 = np.linalg.norm(src_pos)
        tau = (dist - dist_0) / c  # time delay relative to array center

        # Generate broadband signal
        sig = np.zeros((frame_size, 16), dtype=np.float64)
        for f in tones:
            for ch in range(16):
                sig[:, ch] += (1.0 / len(tones)) * 0.5 * np.cos(2 * np.pi * f * (t - tau[ch]))
        # Add slight white noise (SNR ~ 40 dB)
        sig += 1e-4 * np.random.randn(*sig.shape)

        localizer.reset_csm_state()
        heatmap, peak = localizer.localize_with_peak(sig)

        rec_az = peak["azimuth"]
        rec_el = peak["elevation"]
        az_err = rec_az - test_az
        el_err = rec_el - test_el
        val = peak["value"]

        # 0.5 degree is grid step
        passed = (abs(az_err) <= 0.6) and (abs(el_err) <= 0.6)
        if not passed:
            all_passed = False

        status_str = "PASS" if passed else "FAIL"
        print(f"{test_az:8.1f}° | {test_el:8.1f}° | {r_src:8.1f} | {rec_az:8.1f}° | {rec_el:8.1f}° | {az_err:+7.2f}° | {el_err:+7.2f}° | {val:9.4f} | {status_str:>8}")

        results.append({
            "true_az": test_az,
            "true_el": test_el,
            "dist": r_src,
            "rec_az": rec_az,
            "rec_el": rec_el,
            "az_err": az_err,
            "el_err": el_err,
            "passed": passed,
        })

    print("-" * 92)
    if all_passed:
        print("[CONCLUSION] Synthetic Elevation Sweep: ALL PASSED.")
        print("Acoular steering vector equations, angular grid, and MUSIC argmax can perfectly recover elevation.")
    else:
        print("[WARNING] Some synthetic elevation test cases failed! Inspect grid or steering equations.")

    return results


# =============================================================================
# PART 2: ACOULAR COORDINATE SYSTEM & PHYSICAL CONVENTION INSPECTION
# =============================================================================

def inspect_acoular_conventions():
    """
    Mathematically inspect and document the Acoular coordinate system,
    microphone positions, and steering unit vectors.
    """
    print("\n" + "=" * 80)
    print("PART 2: ACOULAR COORDINATE SYSTEM & PHYSICAL CONVENTION INSPECTION")
    print("=" * 80)

    mic_pos = build_microphone_positions()  # shape (3, 16)
    print(f"Acoular Mic Positions shape: {mic_pos.shape} (X, Y, Z for 16 channels)")

    print("\nChannel-by-channel Acoular (X, Y, Z) coordinates (in meters):")
    print(f"{'Channel':<10} | {'Acoular X (m)':<15} | {'Acoular Y (m)':<15} | {'Acoular Z (m)':<15} | {'Physical Board Position'}")
    print("-" * 80)

    for ch in range(16):
        x, y, z = mic_pos[0, ch], mic_pos[1, ch], mic_pos[2, ch]
        x_desc = "LEFT (+X)" if x > 0 else "RIGHT (-X)"
        z_desc = "TOP (+Z)" if z > 0.04 else ("ROW2 (+Z)" if z > 0 else ("ROW3 (-Z)" if z > -0.04 else "BOTTOM (-Z)"))
        print(f"CH {ch+1:<7} | {x:+15.3f} | {y:+15.3f} | {z:+15.3f} | {x_desc}, {z_desc}")

    print("\nSpherical Unit Vector Equation in acoustic.steering:")
    print("  x_sphere = cos(el) * cos(az)")
    print("  y_sphere = cos(el) * sin(az)")
    print("  z_sphere = sin(el)")

    print("\nEvaluating directions:")
    test_dirs = [
        ("Broadside Center", 90.0, 0.0),
        ("Center Up (+15°)", 90.0, 15.0),
        ("Center Down (-15°)", 90.0, -15.0),
        ("Camera Left (az=110°)", 110.0, 0.0),
        ("Camera Right (az=70°)", 70.0, 0.0),
    ]
    print(f"{'Direction':<22} | {'Az':>6} | {'El':>6} | {'Unit X':>10} | {'Unit Y':>10} | {'Unit Z':>10} | {'Physical Meaning'}")
    print("-" * 90)
    for name, az, el in test_dirs:
        az_r, el_r = np.deg2rad(az), np.deg2rad(el)
        ux = np.cos(el_r) * np.cos(az_r)
        uy = np.cos(el_r) * np.sin(az_r)
        uz = np.sin(el_r)
        meaning = []
        if uy > 0.9: meaning.append("+Y forward")
        if ux < -0.1: meaning.append("-X Cam Left")
        if ux > 0.1: meaning.append("+X Cam Right")
        if uz > 0.1: meaning.append("+Z Higher (Up)")
        if uz < -0.1: meaning.append("-Z Lower (Down)")
        print(f"{name:<22} | {az:6.1f} | {el:6.1f} | {ux:+10.4f} | {uy:+10.4f} | {uz:+10.4f} | {', '.join(meaning)}")


# =============================================================================
# PART 3: PROJECTION MAPPING VERIFICATION
# =============================================================================

def inspect_projection_mapping():
    """
    Inspect the angle_to_image mapping in fusion.visualization.
    """
    print("\n" + "=" * 80)
    print("PART 3: CAMERA PROJECTION MAPPING VERIFICATION")
    print("=" * 80)

    visualizer = AcousticHeatmapVisualizer(
        horizontal_fov_deg=50.0,
        vertical_fov_deg=37.5,
        mirror_camera=False,
    )

    w, h = 640, 480
    print(f"Image Resolution: {w} x {h}")
    print(f"Acoustic Grid: Azimuth [{visualizer.azimuth_center - 25.0}° .. {visualizer.azimuth_center + 25.0}°]")
    print(f"Acoustic Grid: Elevation [{visualizer.elevation_center - 18.75}° .. {visualizer.elevation_center + 18.75}°]")

    test_angles = [
        ("Top of Grid (+18.75°)", 90.0, +18.75),
        ("Upper Region (+10.0°)", 90.0, +10.0),
        ("Center Level (0.0°)", 90.0, 0.0),
        ("Lower Region (-10.0°)", 90.0, -10.0),
        ("Bottom of Grid (-18.75°)", 90.0, -18.75),
    ]

    print(f"\n{'Description':<25} | {'Az':>6} | {'El':>8} | {'u (px)':>8} | {'v (px)':>8} | {'Image Region':<15}")
    print("-" * 75)
    for desc, az, el in test_angles:
        u, v = visualizer.angle_to_image(az, el, w, h)
        region = "TOP" if v < 160 else ("CENTER" if v < 320 else "BOTTOM")
        print(f"{desc:<25} | {az:6.1f} | {el:+8.2f}° | {u:8d} | {v:8d} | {region:<15}")

    print("\nProjection Formula check:")
    print("  y_norm = (elevation_deg - el_min) / (el_max - el_min)")
    print("  v = int((1.0 - y_norm) * height)")
    print("  When elevation = 0.0°   -> y_norm = 0.5  -> v = 240 px (EXACT CENTER)")
    print("  When elevation = +18.75° -> y_norm = 1.0  -> v = 0 px   (TOP EDGE)")
    print("  When elevation = -18.75° -> y_norm = 0.0  -> v = 479 px (BOTTOM EDGE)")


# =============================================================================
# PART 4: REAL HARDWARE ELEVATION TEST PROCEDURE
# =============================================================================

def draw_hardware_gui(frame: np.ndarray, phase: str, sub: str, live_az=None, live_el=None, live_u=None, live_v=None):
    display = frame.copy()
    h, w = display.shape[:2]

    # Horizontal thirds and center lines
    cv2.line(display, (0, h // 3), (w, h // 3), (180, 180, 180), 1, cv2.LINE_AA)
    cv2.line(display, (0, 2 * h // 3), (w, 2 * h // 3), (180, 180, 180), 1, cv2.LINE_AA)
    cv2.line(display, (0, h // 2), (w, h // 2), (0, 255, 0), 2, cv2.LINE_AA)  # Center line Y=240
    cv2.line(display, (w // 2, 0), (w // 2, h), (0, 255, 0), 2, cv2.LINE_AA)  # Center line X=320

    # Banner
    cv2.rectangle(display, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.putText(display, phase, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # Subtitle
    if sub:
        cv2.rectangle(display, (0, 40), (w, 68), (10, 10, 10), -1)
        cv2.putText(display, sub, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # Target marker if clicked
    with mouse_lock:
        tx, ty = target_x, target_y
    if tx is not None and ty is not None:
        cv2.drawMarker(display, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        cv2.circle(display, (tx, ty), 12, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, f"Target ({tx}, {ty}) dy={ty-240:+d}px", (tx + 10, ty - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # Acoustic overlay if available
    if live_u is not None and live_v is not None:
        cv2.drawMarker(display, (live_u, live_v), (0, 0, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        cv2.circle(display, (live_u, live_v), 16, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f"Acoustic: Az={live_az:.1f} El={live_el:.1f}", (live_u + 10, live_v + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    # HUD bottom
    cv2.rectangle(display, (0, h - 30), (w, h), (15, 15, 15), -1)
    cv2.putText(display, "L-Click: Mark Target | ENTER / SPACE: Proceed | Q/ESC: Abort",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return display


def analyze_vertical_profile(heatmap_2d: np.ndarray, peak_row: int, peak_col: int, el_values: np.ndarray):
    """
    Analyze the 1D vertical elevation slice through the peak azimuth column.
    """
    profile = heatmap_2d[:, peak_col]
    top_rows = np.argsort(profile)[::-1][:5]
    print(f"\n--- Vertical Slice Analysis at Peak Azimuth Column ({peak_col}) ---")
    print(f"Peak Row: {peak_row} (El = {el_values[peak_row]:.1f}°), Profile Value: {profile[peak_row]:.4f}")
    print(f"Profile range: min={np.min(profile):.4f}, max={np.max(profile):.4f}")
    print(f"Top 5 Elevation points along this column:")
    for r in top_rows:
        print(f"  Row {r:2d} (El = {el_values[r]:+5.1f}°): {profile[r]:.4f}")

    is_boundary = (peak_row == 0 or peak_row == len(el_values) - 1)
    if is_boundary:
        print("  [ALERT] Peak is at the GRID BOUNDARY! The physical sound source is likely outside the grid.")
    else:
        print("  [INFO] Peak is an INTERIOR local extremum within the grid.")


def record_elevation_position(pipeline: MultimodalPipeline, visualizer: AcousticHeatmapVisualizer, position_label: str, countdown_sec: int = 5, num_frames: int = 10):
    global target_x, target_y

    print("\n" + "=" * 80)
    print(f"HARDWARE ELEVATION TEST: {position_label.upper()}")
    print("=" * 80)
    print(f"1. Place the sound source at ~1.5 m distance.")
    print(f"2. Align vertically for: {position_label.upper()}")
    if position_label.lower() == "level":
        print("   -> Position source at approximately the SAME HEIGHT as array center (image y ~ 240)")
    elif position_label.lower() == "above":
        print("   -> Position source clearly ABOVE array center (image y < 160)")
    elif position_label.lower() == "below":
        print("   -> Position source clearly BELOW array center (image y > 320)")
    print("3. Click on the sound source in the OpenCV window to record visual ground truth.")
    print("4. Press ENTER or SPACEBAR in the live window when ready.")
    print("=" * 80)

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
    while not enter_pressed.is_set():
        cam_frame = pipeline.camera.read()
        if cam_frame is None:
            time.sleep(0.01)
            continue

        phase_str = f"POSITION: {position_label.upper()} | Click source to mark (x, y)"
        sub_str = f"Target height: {position_label.upper()} | Press ENTER/SPACEBAR when ready"
        gui = draw_hardware_gui(cam_frame.frame, phase_str, sub_str)
        cv2.imshow(WINDOW_NAME, gui)
        key = cv2.waitKey(15) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("\nAborted by user.")
            return None
        elif key in (10, 13, ord(' ')):  # ENTER or SPACE
            enter_pressed.set()
            break

    with mouse_lock:
        click_x, click_y = target_x, target_y

    print(f"\n[Ground Truth Confirmed] Visual click: x={click_x}, y={click_y} (Center is 320, 240)")

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
        gui = draw_hardware_gui(cam_frame.frame, f"{position_label.upper()} STARTING IN {rem}s", "Ensure speech/audio is actively playing!")
        h, w = gui.shape[:2]
        cv2.putText(gui, str(rem), (w//2 - 25, h//2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, gui)
        if cv2.waitKey(20) & 0xFF in (ord('q'), ord('Q'), 27):
            return None

    # Drain audio buffer and reset CSM
    with pipeline.audio._buffer_lock:
        pipeline.audio._buffer = np.empty((0, pipeline.audio.channels), dtype=np.float32)
    pipeline.localizer.reset_csm_state()
    time.sleep(0.1)

    # Recording phase
    print(f"\nRecording {num_frames} synchronized frames for {position_label.upper()}...")
    print(f"{'#':<3} | {'Cam ID':<7} | {'dSync(ms)':<10} | {'Az (deg)':<9} | {'El (deg)':<9} | {'dB':<6} | {'Overlay (u, v)':<15}")
    print("-" * 75)

    records = []
    best_image = None
    best_raw = None
    last_heatmap = None
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
        live_az, live_el, live_u, live_v = None, None, None, None

        if result is not None and result.synchronized:
            loc = result.acoustic.localization
            if loc is not None and loc.valid:
                h, w = cam_frame.frame.shape[:2]
                az = float(result.acoustic.metadata.get("azimuth_deg", 90.0))
                el = float(result.acoustic.metadata.get("elevation_deg", 0.0))
                db = float(result.acoustic.metadata.get("db_level", 0.0))
                u, v = visualizer.angle_to_image(az, el, w, h)

                live_az, live_el, live_u, live_v = az, el, u, v
                rendered = visualizer.render(cam_frame.frame.copy(), result)

                if best_image is None:
                    best_image = rendered.copy()
                    best_raw = cam_frame.frame.copy()

                last_heatmap = result.acoustic.metadata.get("heatmap")

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

        gui = draw_hardware_gui(cam_frame.frame, f"RECORDING: {position_label.upper()} ({len(records)}/{num_frames})",
                                "Keep audio playing continuously!", live_az, live_el, live_u, live_v)
        cv2.imshow(WINDOW_NAME, gui)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    if not records:
        print("No frames recorded!")
        return None

    # Save representative images
    tag = position_label.lower().replace(" ", "_")
    img_path = os.path.join(OUTPUT_DIR, f"{tag}_elevation_rep.jpg")
    raw_path = os.path.join(OUTPUT_DIR, f"{tag}_elevation_raw.jpg")
    if best_image is not None:
        cv2.imwrite(img_path, best_image)
    if best_raw is not None:
        cv2.imwrite(raw_path, best_raw)

    if last_heatmap is not None:
        np.save(os.path.join(OUTPUT_DIR, f"{tag}_heatmap.npy"), last_heatmap)
        peak_col = int(records[-1].get("overlay_u", 320) * 101 / 640)
        peak_col = int(np.clip(peak_col, 0, 100))
        peak_row = int(np.argmin(np.abs(pipeline.localizer.el_values - records[-1]["elevation"])))
        analyze_vertical_profile(last_heatmap, peak_row, peak_col, pipeline.localizer.el_values)

    az_med = float(np.median([r["azimuth"] for r in records]))
    el_med = float(np.median([r["elevation"] for r in records]))
    u_med = float(np.median([r["overlay_u"] for r in records]))
    v_med = float(np.median([r["overlay_v"] for r in records]))

    summary = {
        "position": position_label.upper(),
        "click_x": click_x,
        "click_y": click_y,
        "az_median": az_med,
        "el_median": el_med,
        "u_median": u_med,
        "v_median": v_med,
        "representative_image": img_path,
        "raw_image": raw_path,
        "records": records,
    }

    # Save to json
    all_data = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                all_data = json.load(f)
        except Exception:
            all_data = {}
    all_data[position_label.upper()] = summary
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\n[{position_label.upper()} Summary] Visual click: ({click_x}, {click_y}) | Acoustic: Az={az_med:.1f}°, El={el_med:.1f}° | Overlay: ({u_med:.0f}, {v_med:.0f})")
    return summary


def print_summary():
    if not os.path.exists(RESULTS_FILE):
        print("No elevation results found at:", RESULTS_FILE)
        return
    with open(RESULTS_FILE, "r") as f:
        res = json.load(f)

    print("\n" + "=" * 80)
    print("REAL HARDWARE ELEVATION TEST SUMMARY")
    print("=" * 80)
    for pos in ["LEVEL", "ABOVE", "BELOW"]:
        if pos in res:
            d = res[pos]
            print(f"\n[{pos}]: Clicked ({d['click_x']}, {d['click_y']})")
            print(f"  Azimuth Median:   {d['az_median']:5.1f}°")
            print(f"  Elevation Median: {d['el_median']:5.1f}°")
            print(f"  Overlay Median:   ({d['u_median']:.0f}, {d['v_median']:.0f})")
            print(f"  Representative:   {d.get('representative_image')}")
        else:
            print(f"\n[{pos}]: NOT RECORDED")

    if "ABOVE" in res and "LEVEL" in res and "BELOW" in res:
        a, l, b = res["ABOVE"], res["LEVEL"], res["BELOW"]
        print("\n" + "-" * 80)
        print("ELEVATION ORDERING CHECK (ABOVE > LEVEL > BELOW):")
        el_pass = a["el_median"] > l["el_median"] > b["el_median"]
        print(f"  {a['el_median']:.1f}° > {l['el_median']:.1f}° > {b['el_median']:.1f}°  ==> {'PASS' if el_pass else 'FAIL'}")
        v_pass = a["v_median"] < l["v_median"] < b["v_median"]
        print("OVERLAY V ORDERING CHECK (ABOVE < LEVEL < BELOW, smaller v = higher):")
        print(f"  {a['v_median']:.0f}px < {l['v_median']:.0f}px < {b['v_median']:.0f}px  ==> {'PASS' if v_pass else 'FAIL'}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Acoustic Camera Elevation Diagnostic")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic elevation sweep")
    parser.add_argument("--inspect", action="store_true", help="Inspect Acoular coordinate convention")
    parser.add_argument("--hardware", choices=["level", "above", "below", "all"], help="Run real hardware elevation test")
    parser.add_argument("--summary", action="store_true", help="Print summary of recorded results")
    parser.add_argument("--stream-url", default="http://10.243.98.165:81/stream", help="ESP32-CAM stream URL")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    # If no flags specified, run synthetic and inspection by default
    if not (args.synthetic or args.inspect or args.hardware):
        inspect_acoular_conventions()
        inspect_projection_mapping()
        run_synthetic_elevation_sweep()
        return

    if args.inspect:
        inspect_acoular_conventions()
        inspect_projection_mapping()

    if args.synthetic:
        run_synthetic_elevation_sweep()

    if args.hardware:
        config.settings.STREAM_URL = args.stream_url

        from vision.camera import CameraAcquisition
        from acoustic.audio import AudioAcquisition
        import sounddevice as sd

        devices = sd.query_devices()
        uma_dev = None
        for idx, dev in enumerate(devices):
            if "uma16v2" in dev["name"].lower() and dev["max_input_channels"] >= 16:
                uma_dev = idx
                break

        print(f"Selected audio device index: {uma_dev}")
        camera = CameraAcquisition(stream_url=args.stream_url, expected_width=640, expected_height=480)
        audio = AudioAcquisition(device=uma_dev)
        pipeline = MultimodalPipeline(camera=camera, audio=audio)
        visualizer = AcousticHeatmapVisualizer(horizontal_fov_deg=50.0, vertical_fov_deg=37.5, mirror_camera=False)

        pipeline.start()
        try:
            if args.hardware == "all":
                for pos in ["level", "above", "below"]:
                    res = record_elevation_position(pipeline, visualizer, pos)
                    if res is None:
                        break
            else:
                record_elevation_position(pipeline, visualizer, args.hardware)
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

        print_summary()


if __name__ == "__main__":
    main()
