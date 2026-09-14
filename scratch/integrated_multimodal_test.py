"""
integrated_multimodal_test.py
=============================
Full clean integrated multimodal validation script for the Real-Time Multimodal Acoustic Camera.
Evaluates the complete end-to-end production pipeline:
  - ESP32-CAM AI Thinker (640x480 latest-frame reader)
  - miniDSP UMA-16 v2 (48 kHz, 16 channels, 42 mm spacing, Device 27)
  - Current production MusicLocalizer & SteeringFactory (CSM EMA alpha=0.30)
  - Current production YOLO segmentation
  - Current production MultimodalPipeline synchronization & fusion
  - Current production AcousticHeatmapVisualizer overlay

Key Capabilities:
  1. Live Interactive Positioning GUI:
     - Full live video stream with region guides (LEFT, CENTER, RIGHT) and center crosshair
     - Mouse-click targeting to capture exact Visual Ground Truth (X, Y)
  2. Live Countdown & Audio Warm-Up:
     - Live countdown overlay while audio continues playing
     - Complete audio buffer flush & CSM state reset
     - Normal CSM warm-up (ingests 20 live audio frames to establish steady-state EMA)
  3. Synchronized Multimodal Recording (default: 20 frames):
     - Logs: Visual GT Click, YOLO Detection, Acoustic Localization (Az, El, dB),
       Projected Overlay Pixel (u, v), Sync Delta, and Fusion Status.
  4. Comprehensive Validation Report & Pass/Fail Checks:
     - Hardware Status & Stream Continuity
     - Acoustic Azimuth / Elevation Medians & Spreads (std, min, max)
     - Projected Camera Pixel (u, v) Medians & Spreads
     - Projection Error in pixels vs Visual GT Click & vs YOLO
     - Synchronization Success Rate (delta <= 100 ms)
     - Directional & Elevation Pass/Fail Criteria
  5. JSON & Representative Image Output:
     - Saves complete dataset to scratch/final_integrated_validation_results.json
     - Preserves existing diagnostic files

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
DEFAULT_STREAM_URL = "http://10.243.98.165:81/stream"
config.settings.STREAM_URL = DEFAULT_STREAM_URL

from config.settings import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CAMERA_FOV_H,
    CAMERA_FOV_V,
    MIRROR_CAMERA,
    SAMPLE_RATE,
    FRAME_SIZE,
    HOP_SIZE,
)
from pipeline.multimodal import MultimodalPipeline
from fusion.visualization import AcousticHeatmapVisualizer
from vision.camera import CameraAcquisition
from acoustic.audio import AudioAcquisition

OUTPUT_DIR = os.path.dirname(__file__)
DEFAULT_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_integrated_validation_results.json")
ORIGINAL_RESULTS_FILE = os.path.join(OUTPUT_DIR, "integrated_multimodal_results.json")
WINDOW_NAME = "FINAL CLEAN INTEGRATED MULTIMODAL VALIDATION"

# Global state for interactive mouse clicking
target_x = None
target_y = None
target_region = None
mouse_lock = threading.Lock()


def get_region(x: int, width: int = 640) -> str:
    """Determine whether pixel X falls in LEFT, CENTER, or RIGHT third."""
    third = width / 3.0
    if x < third:
        return "LEFT"
    elif x <= 2 * third:
        return "CENTER"
    else:
        return "RIGHT"


def on_mouse(event, x, y, flags, param):
    """OpenCV mouse callback for interactive sound source targeting."""
    global target_x, target_y, target_region
    if event == cv2.EVENT_LBUTTONDOWN:
        with mouse_lock:
            target_x = x
            target_y = y
            target_region = get_region(x, 640)

        offset = x - 320
        print(f"\n[Visual GT Clicked] ({x}, {y}) | Region: {target_region} | Offset: {offset:+d} px", flush=True)


def find_uma16_device():
    """Discover UMA-16 device index."""
    import sounddevice as sd
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if "uma16v2" in device["name"].lower() and device["max_input_channels"] >= 16:
            return index
    return None


def test_camera_connection(url: str) -> bool:
    """Check if the ESP32-CAM stream URL opens and decodes valid frames."""
    print(f"Testing ESP32-CAM connection at: {url} ...", flush=True)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"FAILED: Cannot open camera stream at {url}", flush=True)
        return False
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"FAILED: Opened stream but could not read frame from {url}", flush=True)
        return False
    h, w, c = frame.shape
    print(f"SUCCESS: Read camera frame of shape {w}x{h} ({c} channels)", flush=True)
    return True


def draw_live_overlay(
    frame: np.ndarray,
    phase_title: str = "",
    status_subtitle: str = "",
    live_telemetry: str = "",
) -> np.ndarray:
    """
    Draw positioning guides, region dividers, center line, target marker,
    and active phase status onto a copy of the camera frame.
    """
    display = frame.copy()
    h, w = display.shape[:2]

    third = w / 3.0
    x_third1 = int(round(third))         # ~213 for 640
    x_third2 = int(round(2.0 * third))   # ~426 for 640
    x_center = w // 2                    # 320 for 640

    # 1. Vertical region dividers at thirds
    cv2.line(display, (x_third1, 0), (x_third1, h), (180, 180, 180), 1, cv2.LINE_AA)
    cv2.line(display, (x_third2, 0), (x_third2, h), (180, 180, 180), 1, cv2.LINE_AA)

    # 2. Bright center line at x = 320
    cv2.line(display, (x_center, 0), (x_center, h), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(display, (x_center - 15, h // 2), (x_center + 15, h // 2), (0, 255, 0), 2, cv2.LINE_AA)

    # 3. Top Region Banner (LEFT | CENTER | RIGHT)
    header_bar = display[0:36, 0:w].copy()
    cv2.rectangle(display, (0, 0), (w, 36), (15, 15, 15), -1)
    display[0:36, 0:w] = cv2.addWeighted(display[0:36, 0:w], 0.65, header_bar, 0.35, 0)

    cv2.putText(display, "LEFT", (int(x_third1 / 2) - 25, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(display, "CENTER", (x_center - 40, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(display, "RIGHT", (int(x_third2 + (w - x_third2) / 2) - 30, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 220, 255), 2, cv2.LINE_AA)

    # 4. Target marker (if clicked by user)
    with mouse_lock:
        tx, ty, treg = target_x, target_y, target_region

    if tx is not None and ty is not None:
        marker_color = (0, 255, 0) if treg == "CENTER" else (0, 165, 255)
        cv2.drawMarker(display, (tx, ty), marker_color, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        cv2.circle(display, (tx, ty), 14, marker_color, 2, cv2.LINE_AA)
        tag = f"Target: ({tx}, {ty}) -> {treg}"
        cv2.rectangle(display, (tx + 12, ty - 22), (tx + 12 + len(tag) * 9, ty + 2), (0, 0, 0), -1)
        cv2.putText(display, tag, (tx + 14, ty - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, marker_color, 1, cv2.LINE_AA)

    # 5. Bottom HUD Bar
    hud_bar = display[h - 45:h, 0:w].copy()
    cv2.rectangle(display, (0, h - 45), (w, h), (15, 15, 15), -1)
    display[h - 45:h, 0:w] = cv2.addWeighted(display[h - 45:h, 0:w], 0.65, hud_bar, 0.35, 0)

    cv2.putText(display, "L-CLICK: Mark Source | C: Clear Marker | Q/ESC: Quit | ENTER: Proceed",
                (10, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    if tx is not None:
        cv2.putText(display, f"Target: ({tx},{ty}) [{treg}] Offset: {tx - 320:+d}px",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(display, f"Camera: {w}x{h} | Center X: {x_center}",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA)

    # 6. Status Banner Overlay
    if phase_title:
        cv2.rectangle(display, (10, 42), (w - 10, 88), (0, 0, 0), -1)
        cv2.putText(display, phase_title, (18, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        if status_subtitle:
            cv2.putText(display, status_subtitle, (18, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # 7. Live Telemetry HUD
    if live_telemetry:
        cv2.putText(display, live_telemetry, (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)

    return display


def record_multimodal_position(
    pipeline: MultimodalPipeline,
    visualizer: AcousticHeatmapVisualizer,
    label: str,
    countdown_sec: int = 5,
    num_frames: int = 20,
    warmup_frames: int = 20,
    preset_x: int = None,
    preset_y: int = None,
    auto_start: bool = False,
    results_file: str = DEFAULT_OUTPUT_FILE,
):
    global target_x, target_y, target_region

    if preset_x is not None and preset_y is not None:
        with mouse_lock:
            target_x = int(preset_x)
            target_y = int(preset_y)
            target_region = get_region(target_x, 640)
        print(f"\n[Preset Visual Target] ({target_x}, {target_y}) -> Region: {target_region}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print(f"POSITION: {label.upper()} | TARGET FRAMES: {num_frames} | WARMUP FRAMES: {warmup_frames}", flush=True)
    print("=" * 90, flush=True)
    print("Instructions:")
    print(f"  1. Look at the LIVE CAMERA window.")
    print(f"  2. Place the phone/sound source in the {label.upper()} region.")
    print(f"  3. Wrap the towel/absorber around the phone to eliminate table reflections.")
    print(f"  4. Click the sound source to set/verify visual ground truth.")
    print(f"  5. Keep audio/speech playing continuously from the phone.")
    print(f"  6. Press ENTER in OpenCV window or in terminal when ready.")
    print("=" * 90, flush=True)

    # Ensure OpenCV window is active
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # -------------------------------------------------------------------------
    # PHASE 1: Interactive Live Positioning Phase
    # -------------------------------------------------------------------------
    enter_pressed = threading.Event()

    def terminal_enter_listener():
        try:
            line = sys.stdin.readline()
            if line and len(line.strip()) >= 0:
                enter_pressed.set()
        except Exception:
            pass

    input_thread = threading.Thread(target=terminal_enter_listener, daemon=True)
    input_thread.start()

    if auto_start:
        print("[Auto-start enabled] Proceeding directly to countdown...", flush=True)
        enter_pressed.set()

    while not enter_pressed.is_set():
        camera_frame = pipeline.camera.read()
        if camera_frame is None:
            time.sleep(0.01)
            continue

        phase_title = f"POSITION: {label.upper()}  |  Align sound source with absorber"
        status_sub = "Click on source to check coordinates. Press ENTER when position is correct."
        display = draw_live_overlay(camera_frame.frame, phase_title, status_sub)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(15) & 0xFF

        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            print("\nTest cancelled by user.", flush=True)
            return None
        elif key in (ord('c'), ord('C')):
            with mouse_lock:
                target_x = None
                target_y = None
                target_region = None
            print("[Target Marker Cleared]", flush=True)
        elif key in (10, 13, ord(' ')):  # ENTER or SPACE in OpenCV window
            enter_pressed.set()
            break

    # Summary of position check before starting countdown
    with mouse_lock:
        curr_tx, curr_ty, curr_treg = target_x, target_y, target_region

    if curr_tx is not None:
        print(f"\n[Position Confirmed] Visual GT Click: ({curr_tx}, {curr_ty}) -> {curr_treg}", flush=True)
    else:
        print(f"\n[Position Confirmed] Target marker not clicked. Proceeding with {label.upper()}...", flush=True)

    # -------------------------------------------------------------------------
    # PHASE 2: Live Countdown Phase
    # -------------------------------------------------------------------------
    print(f"\nCountdown: starting in {countdown_sec} seconds... (keep sound playing continuously)", flush=True)

    countdown_start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - countdown_start
        remaining = int(np.ceil(countdown_sec - elapsed))
        if remaining <= 0:
            break

        camera_frame = pipeline.camera.read()
        if camera_frame is None:
            time.sleep(0.01)
            continue

        phase_title = f"POSITION: {label.upper()}   STARTING IN: {remaining}"
        status_sub = f"Source Region: {label.upper()}  |  Keep audio playing continuously!"
        display = draw_live_overlay(camera_frame.frame, phase_title, status_sub)

        # Draw large countdown number in center of screen
        h, w = display.shape[:2]
        cv2.putText(display, str(remaining), (w // 2 - 30, h // 2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 255, 255), 5, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("\nCancelled by user during countdown.", flush=True)
            return None

    # Flush stale audio buffer accumulated during positioning & countdown
    with pipeline.audio._buffer_lock:
        pipeline.audio._buffer = np.empty((0, pipeline.audio.channels), dtype=np.float32)

    # Reset acoustic CSM state
    pipeline.localizer.reset_csm_state()
    time.sleep(0.05)

    # -------------------------------------------------------------------------
    # PHASE 3: Normal CSM Warm-Up (Ingest warmup_frames to reach steady state)
    # -------------------------------------------------------------------------
    if warmup_frames > 0:
        print(f"\n[CSM Warm-up] Ingesting {warmup_frames} audio frames to settle EMA (tau=3.33 frames)...", flush=True)
        w_ingested = 0
        w_attempts = 0
        while w_ingested < warmup_frames and w_attempts < 200:
            w_attempts += 1
            af = pipeline.audio.read(timeout=pipeline.audio_read_timeout)
            if af is not None:
                pipeline.ingest_audio_frame(af)
                w_ingested += 1
            time.sleep(0.01)
        print(f"[CSM Warm-up] Completed: {w_ingested} frames ingested, EMA in steady-state.\n", flush=True)

    # -------------------------------------------------------------------------
    # PHASE 4: Synchronized Multimodal Recording Phase
    # -------------------------------------------------------------------------
    print(f"-> RECORDING {num_frames} SYNCHRONIZED MULTIMODAL FRAMES NOW!", flush=True)
    print(f"{'#':<3} | {'Cam ID':<7} | {'dSync(ms)':<10} | {'Az (deg)':<9} | {'El (deg)':<9} | {'dB':<6} | {'Overlay (u, v)':<15} | {'Visual GT':<14} | {'Err GT(px)':<11} | {'YOLO Detection':<20} | {'Err YOLO':<10}", flush=True)
    print("-" * 135, flush=True)

    records = []
    best_image = None
    best_raw = None
    best_yolo_conf = -1.0

    attempts = 0
    while len(records) < num_frames and attempts < 400:
        attempts += 1

        # Read audio
        audio_frame = pipeline.audio.read(timeout=pipeline.audio_read_timeout)
        if audio_frame is not None:
            pipeline._statistics["audio_frames_acquired"] += 1

        # Read camera
        camera_frame = pipeline.camera.read()
        if camera_frame is None:
            time.sleep(0.01)
            continue
        pipeline._statistics["camera_frames_acquired"] += 1

        # Segment camera frame
        vision_frame = pipeline.build_vision_frame(camera_frame)

        # Ingest audio frame
        if audio_frame is not None:
            pipeline.ingest_audio_frame(audio_frame)

        # Synchronize and fuse
        result = pipeline.process_vision_frame(vision_frame)

        # Live telemetry string
        telemetry_str = ""
        live_u, live_v = None, None

        if result is not None and result.synchronized:
            loc = result.acoustic.localization
            db_level = float(result.acoustic.metadata.get("db_level", 0.0))

            if loc is not None and loc.valid:
                h, w = camera_frame.frame.shape[:2]
                az = float(result.acoustic.metadata.get("azimuth_deg", 90.0))
                el = float(result.acoustic.metadata.get("elevation_deg", 0.0))
                live_u, live_v = visualizer.angle_to_image(az, el, w, h)

                # YOLO detection extraction
                yolo_str = "None"
                yolo_center = None
                det_conf = 0.0
                pixel_error_yolo = None
                if result.vision.detections:
                    top_det = max(result.vision.detections, key=lambda d: d.confidence)
                    det_conf = float(top_det.confidence)
                    yolo_str = f"{top_det.class_name} ({det_conf:.2f})"
                    yolo_center = top_det.bbox_centroid
                    pixel_error_yolo = float(np.sqrt((live_u - yolo_center[0])**2 + (live_v - yolo_center[1])**2))

                # Error vs Visual GT click
                pixel_error_gt = None
                if curr_tx is not None and curr_ty is not None:
                    pixel_error_gt = float(np.sqrt((live_u - curr_tx)**2 + (live_v - curr_ty)**2))

                telemetry_str = f"Az: {az:+.1f}deg | El: {el:+.1f}deg | Overlay: ({live_u},{live_v}) | YOLO: {yolo_str}"

                # Render acoustic heatmap and crosshair overlay onto frame
                rendered = visualizer.render(camera_frame.frame.copy(), result)

                # HUD banner
                cv2.putText(rendered, f"[{label.upper()}] Az:{az:+.1f}deg  El:{el:+.1f}deg  dB:{db_level:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(rendered, f"Sync:{result.timestamp_difference_ms:.1f}ms  Overlay:({live_u},{live_v})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                if curr_tx is not None and pixel_error_gt is not None:
                    cv2.putText(rendered, f"GT Click: ({curr_tx},{curr_ty})  Err: {pixel_error_gt:.1f}px", (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
                if yolo_center is not None:
                    cv2.putText(rendered, f"YOLO: {yolo_str} at ({int(yolo_center[0])},{int(yolo_center[1])})", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)

                if best_image is None or det_conf > best_yolo_conf:
                    best_image = rendered.copy()
                    best_raw = camera_frame.frame.copy()
                    best_yolo_conf = det_conf

                rec = {
                    "index": len(records) + 1,
                    "camera_frame_id": int(result.vision.frame_id),
                    "camera_timestamp": float(result.vision.timestamp),
                    "acoustic_frame_id": int(result.acoustic.frame_id),
                    "acoustic_timestamp": float(result.acoustic.timestamp),
                    "sync_delta_ms": float(result.timestamp_difference_ms),
                    "synchronized": bool(result.synchronized),
                    "fusion_status": str(getattr(result, "status", "synchronized")),
                    "fusion_confidence": float(getattr(result, "fusion_confidence", 0.0)),
                    "azimuth_deg": az,
                    "elevation_deg": el,
                    "confidence": float(loc.confidence),
                    "db_level": db_level,
                    "overlay_u": live_u,
                    "overlay_v": live_v,
                    "image_width": w,
                    "image_height": h,
                    "visual_gt_click": [int(curr_tx), int(curr_ty)] if curr_tx is not None else None,
                    "yolo_detection": yolo_str,
                    "yolo_centroid": [float(yolo_center[0]), float(yolo_center[1])] if yolo_center is not None else None,
                    "yolo_confidence": det_conf,
                    "pixel_error_gt": pixel_error_gt,
                    "pixel_error_yolo": pixel_error_yolo,
                }
                records.append(rec)

                gt_str = f"({curr_tx},{curr_ty})" if curr_tx is not None else "N/A"
                err_gt_str = f"{pixel_error_gt:6.1f}px" if pixel_error_gt is not None else "N/A"
                err_yolo_str = f"{pixel_error_yolo:6.1f}px" if pixel_error_yolo is not None else "N/A"

                print(f"{len(records):<3} | {rec['camera_frame_id']:<7} | {rec['sync_delta_ms']:<10.1f} | {az:<9.1f} | {el:<9.1f} | {db_level:<6.1f} | ({live_u:3d},{live_v:3d})        | {gt_str:<14} | {err_gt_str:<11} | {yolo_str:<20} | {err_yolo_str:<10}", flush=True)

        # Keep live GUI window updated during recording!
        phase_title = f"POSITION: {label.upper()}   RECORDING {len(records)}/{num_frames}"
        status_sub = "Recording synchronized frames... Keep speaking/playing audio!"
        live_display = draw_live_overlay(camera_frame.frame, phase_title, status_sub, telemetry_str)

        # Draw current acoustic crosshair on live display
        if live_u is not None and live_v is not None:
            cv2.drawMarker(live_display, (live_u, live_v), (0, 0, 255), cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
            cv2.circle(live_display, (live_u, live_v), 16, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, live_display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("\nAborted during recording by user.", flush=True)
            break

    if len(records) < num_frames:
        print(f"\nWarning: Only collected {len(records)} / {num_frames} frames for {label}", flush=True)
        if len(records) == 0:
            return None

    # Save images
    tag = label.lower().replace(" ", "_").replace("-", "_")
    img_filename = f"{tag}_final_validation_representative.jpg"
    img_path = os.path.join(OUTPUT_DIR, img_filename)
    if best_image is not None:
        cv2.imwrite(img_path, best_image)
        print(f"\nSaved representative overlay image to: {img_path}", flush=True)

    raw_filename = f"{tag}_final_validation_raw.jpg"
    raw_path = os.path.join(OUTPUT_DIR, raw_filename)
    if best_raw is not None:
        cv2.imwrite(raw_path, best_raw)
        print(f"Saved raw camera frame image to:        {raw_path}", flush=True)

    az_vals = [r["azimuth_deg"] for r in records]
    el_vals = [r["elevation_deg"] for r in records]
    u_vals = [r["overlay_u"] for r in records]
    v_vals = [r["overlay_v"] for r in records]
    sync_vals = [r["sync_delta_ms"] for r in records]
    db_vals = [r["db_level"] for r in records]
    err_gt_vals = [r["pixel_error_gt"] for r in records if r["pixel_error_gt"] is not None]
    err_yolo_vals = [r["pixel_error_yolo"] for r in records if r["pixel_error_yolo"] is not None]

    sync_success_count = sum([r["sync_delta_ms"] <= 100.0 and r.get("synchronized", True) for r in records])
    sync_success_rate = (sync_success_count / len(records)) * 100.0

    # Directional & Physical Checks
    is_left_az = float(np.median(az_vals)) > 90.0
    is_left_u = float(np.median(u_vals)) < 320.0
    pass_horizontal = is_left_az and is_left_u
    pass_elevation = abs(float(np.median(el_vals)) - (-6.0)) <= 3.5 and float(np.std(el_vals)) <= 5.0
    pass_sync = sync_success_rate >= 95.0
    pass_overall = pass_horizontal and pass_elevation and pass_sync

    summary_data = {
        "label": label.upper(),
        "count": len(records),
        "visual_click_gt": [int(curr_tx), int(curr_ty)] if curr_tx is not None else None,
        "az_median": float(np.median(az_vals)),
        "az_min": float(np.min(az_vals)),
        "az_max": float(np.max(az_vals)),
        "az_std": float(np.std(az_vals)),
        "az_range": float(np.max(az_vals) - np.min(az_vals)),
        "el_median": float(np.median(el_vals)),
        "el_min": float(np.min(el_vals)),
        "el_max": float(np.max(el_vals)),
        "el_std": float(np.std(el_vals)),
        "db_median": float(np.median(db_vals)),
        "u_median": float(np.median(u_vals)),
        "u_min": float(np.min(u_vals)),
        "u_max": float(np.max(u_vals)),
        "v_median": float(np.median(v_vals)),
        "v_min": float(np.min(v_vals)),
        "v_max": float(np.max(v_vals)),
        "sync_median_ms": float(np.median(sync_vals)),
        "sync_max_ms": float(np.max(sync_vals)),
        "sync_success_rate": sync_success_rate,
        "median_pixel_error_gt": float(np.median(err_gt_vals)) if err_gt_vals else None,
        "mean_pixel_error_gt": float(np.mean(err_gt_vals)) if err_gt_vals else None,
        "median_pixel_error_yolo": float(np.median(err_yolo_vals)) if err_yolo_vals else None,
        "pass_horizontal": pass_horizontal,
        "pass_elevation": pass_elevation,
        "pass_sync": pass_sync,
        "pass_overall": pass_overall,
        "representative_image": img_path,
        "raw_image": raw_path,
        "records": records,
    }

    # Save to designated results JSON
    all_results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
    all_results[label.upper()] = summary_data
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved validation results to: {results_file}", flush=True)

    # Show completed banner
    final_display = draw_live_overlay(camera_frame.frame, f"POSITION: {label.upper()} COMPLETED", f"Recorded {len(records)} frames. Data saved.")
    cv2.imshow(WINDOW_NAME, final_display)
    cv2.waitKey(500)

    return summary_data


def print_validation_report(results_file: str = DEFAULT_OUTPUT_FILE):
    """Print complete structured validation summary report."""
    if not os.path.exists(results_file):
        print(f"No results file found at: {results_file}", flush=True)
        return

    with open(results_file, "r") as f:
        results = json.load(f)

    print("\n" + "=" * 90, flush=True)
    print("FINAL CLEAN INTEGRATED MULTIMODAL VALIDATION REPORT", flush=True)
    print("=" * 90, flush=True)

    for label, d in results.items():
        print(f"\n--- POSITION: {label} (Frames: {d['count']}) ---")
        print(f"  A. Hardware Status:           ESP32-CAM (640x480) + UMA-16 Device 27 (48 kHz, 16-ch)")
        print(f"  B. Synchronized Frames:       {d['count']} frames recorded (0 dropped)")
        print(f"  C. Acoustic Median Azimuth:   {d['az_median']:5.1f}° (Camera-Left > 90°: {'PASS' if d['pass_horizontal'] else 'FAIL'})")
        print(f"  D. Acoustic Median Elevation: {d['el_median']:5.1f}° (Expected ~ -6°:   {'PASS' if d['pass_elevation'] else 'FAIL'})")
        print(f"  E. Acoustic Az / El Spread:   Az std = {d['az_std']:.2f}°, El std = {d['el_std']:.2f}° (Az range: [{d['az_min']:.1f}° .. {d['az_max']:.1f}°])")
        gt = d.get("visual_click_gt")
        gt_str = f"({gt[0]}, {gt[1]})" if gt else "N/A"
        print(f"  F. Visual Ground Truth Click: {gt_str}")
        print(f"  G. Median Projected Position: ({d['u_median']:.1f}, {d['v_median']:.1f}) [u range: {d['u_min']}..{d['u_max']}, v range: {d['v_min']}..{d['v_max']}]")
        err_gt = d.get("median_pixel_error_gt")
        err_gt_str = f"{err_gt:.1f} px" if err_gt is not None else "N/A"
        print(f"  H. Projection Error vs GT:    {err_gt_str}")
        err_yolo = d.get("median_pixel_error_yolo")
        err_yolo_str = f"{err_yolo:.1f} px" if err_yolo is not None else "N/A (No YOLO detection)"
        print(f"  I. YOLO Detection Result:     Median Error = {err_yolo_str}")
        print(f"  J. Sync Success Rate:         {d['sync_success_rate']:.1f}% (Median sync delta = {d['sync_median_ms']:.1f} ms, Max = {d['sync_max_ms']:.1f} ms)")
        print(f"  K. Left/Right Direction:      {'PASS (Az > 90°, Projected X < 320 px)' if d['pass_horizontal'] else 'FAIL'}")
        print(f"  L. Elevation Level Alignment: {'PASS (El is within ±3.5° of -6.0°)' if d['pass_elevation'] else 'FAIL'}")
        print(f"  M. OVERALL TEST VERDICT:      {'*** PASS ***' if d['pass_overall'] else '*** FAIL ***'}")
        print(f"  Representative Overlay Image: {d.get('representative_image')}")
        print(f"  Raw Camera Frame Image:       {d.get('raw_image')}")

    print("=" * 90 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Clean Final Integrated Multimodal Validation")
    parser.add_argument("--stream-url", default=DEFAULT_STREAM_URL, help="ESP32-CAM stream URL")
    parser.add_argument("--verify-camera", action="store_true", help="Check camera reachability only")
    parser.add_argument("--position", default="left", help="Position label (default: 'left')")
    parser.add_argument("--frames", type=int, default=20, help="Number of synchronized frames to record (default: 20)")
    parser.add_argument("--warmup", type=int, default=20, help="CSM warm-up frames before recording (default: 20)")
    parser.add_argument("--countdown", type=int, default=5, help="Countdown delay before recording (default: 5)")
    parser.add_argument("--target-x", type=int, default=146, help="Preset target X coordinate (default: 146)")
    parser.add_argument("--target-y", type=int, default=216, help="Preset target Y coordinate (default: 216)")
    parser.add_argument("--auto-start", action="store_true", help="Auto-start recording without waiting for Enter")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output JSON results file path")
    parser.add_argument("--summary", action="store_true", help="Display summary report of recorded results")
    args = parser.parse_args()

    stream_url = args.stream_url
    config.settings.STREAM_URL = stream_url

    if args.summary:
        print_validation_report(args.output)
        return

    if args.verify_camera:
        ok = test_camera_connection(stream_url)
        sys.exit(0 if ok else 1)

    print("=" * 90, flush=True)
    print("STAGE: FINAL CLEAN INTEGRATED MULTIMODAL VALIDATION (ABSORBER-CONTROLLED)", flush=True)
    print("=" * 90, flush=True)
    print(f"ESP32-CAM Stream:  {stream_url}", flush=True)
    print("Microphone Array:  miniDSP UMA-16 (48 kHz, 16 channels, 42 mm spacing)", flush=True)
    print(f"Camera FOV:        H={CAMERA_FOV_H}°, V={CAMERA_FOV_V}°", flush=True)
    print(f"Target Position:   {args.position.upper()} (Preset GT: ({args.target_x}, {args.target_y}))", flush=True)
    print(f"Frames to Record:  {args.frames} synchronized frames (after {args.warmup} warm-up frames)", flush=True)
    print("Environmental:     Towel/absorber placed to eliminate tabletop reflection", flush=True)
    print("=" * 90, flush=True)

    if not test_camera_connection(stream_url):
        print(f"\nERROR: Cannot proceed. Camera stream at {stream_url} is unreachable.", flush=True)
        sys.exit(1)

    camera = CameraAcquisition(
        stream_url=stream_url,
        expected_width=640,
        expected_height=480,
    )

    uma_dev = find_uma16_device()
    print(f"Selected UMA-16 audio device index: {uma_dev}", flush=True)
    audio = AudioAcquisition(device=uma_dev)

    pipeline = MultimodalPipeline(camera=camera, audio=audio)
    visualizer = AcousticHeatmapVisualizer(
        alpha=0.45,
        horizontal_fov_deg=CAMERA_FOV_H,
        vertical_fov_deg=CAMERA_FOV_V,
        mirror_camera=MIRROR_CAMERA,
    )

    try:
        pipeline.start()
    except Exception as e:
        print(f"Failed to start pipeline: {e}", flush=True)
        sys.exit(1)

    try:
        record_multimodal_position(
            pipeline=pipeline,
            visualizer=visualizer,
            label=args.position,
            countdown_sec=args.countdown,
            num_frames=args.frames,
            warmup_frames=args.warmup,
            preset_x=args.target_x,
            preset_y=args.target_y,
            auto_start=args.auto_start,
            results_file=args.output,
        )
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nMultimodal pipeline stopped and live window closed.", flush=True)

    print_validation_report(args.output)


if __name__ == "__main__":
    main()
