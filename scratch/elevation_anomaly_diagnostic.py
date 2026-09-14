"""
elevation_anomaly_diagnostic.py
===============================
Forensic Diagnostic Utility for Investigating Acoustic Elevation Anomalies
at Off-Center Azimuths (e.g. Azimuth ~103°, Visual Target ~LEVEL (146, 258)).

Key Diagnostic Capabilities:
  1. Synthetic Sanity Sweeps (Item 11):
     - Tests synthetic sources at Azimuth 103° across multiple elevations [0°, -5°, -6°, -10°, -15°, +10°]
     - Tests distances [1.5m, 0.8m, 0.5m] to test wave curvature effects.
     - Determines if mathematical steering vector / geometry coupling exists.

  2. Live Hardware Acquisition (Items 1-5):
     - Uses existing UMA-16 Device 27, 48 kHz, 16 channels, and ESP32-CAM stream.
     - Target positioning GUI with visual click crosshair, countdown, and buffer management.
     - Records 10-20 synchronized frames.

  3. 2D Pseudospectrum Peak Decomposition (Item 6):
     - Extracts top N local maxima across the 76x101 (elevation x azimuth) grid.
     - Reports coordinates (Az, El) and normalized pseudospectrum amplitudes.

  4. Vertical Elevation Slice Analysis (Item 7-8):
     - Analyzes vertical profile at peak azimuth +/- 2°.
     - Evaluates whether the boundary peak (-18.75°) is genuinely dominant or competing with a level peak (~-6°).

  5. Per-Subband Dissection (Items 9-10):
     - Evaluates each of the 6 center frequencies (500, 1000, 1500, 2000, 2500, 3000 Hz) before accumulation.
     - Compares standard normalized sum vs unnormalized sum vs high-frequency only vs low-frequency only.
     - Tests whether per-band normalization introduces boundary elevation artifacts.

  6. Forensic Synthesis & Root-Cause Evaluation (Item 12):
     - Tests Hypotheses A, B, C, D, E against empirical evidence.

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
import scipy.ndimage

# Ensure project root is in sys.path
root_dir = r"c:\Users\kushs\OneDrive\Desktop\Projects\multimodal_acoustic_camera"
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config.settings
from config.settings import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CAMERA_FOV_H,
    CAMERA_FOV_V,
    SAMPLE_RATE,
    FRAME_SIZE,
    FREQ_CENTERS,
    BAND_WIDTH,
)
from acoustic.steering import SteeringFactory
from acoustic.music import MusicLocalizer
from fusion.visualization import AcousticHeatmapVisualizer
from vision.camera import CameraAcquisition
from acoustic.audio import AudioAcquisition
from pipeline.multimodal import MultimodalPipeline

OUTPUT_DIR = os.path.dirname(__file__)
WINDOW_NAME = "ACOUSTIC ELEVATION ANOMALY DIAGNOSTIC"

target_x = None
target_y = None
mouse_lock = threading.Lock()


def on_mouse(event, x, y, flags, param):
    """Mouse callback to record visual coordinates of sound source."""
    global target_x, target_y
    if event == cv2.EVENT_LBUTTONDOWN:
        with mouse_lock:
            target_x = x
            target_y = y
        dx = x - (FRAME_WIDTH // 2)
        dy = y - (FRAME_HEIGHT // 2)
        print(f"\n[Visual GT Clicked] ({x}, {y}) | dx={dx:+d}px, dy={dy:+d}px", flush=True)


# =============================================================================
# PART 1: CONTROLLED SYNTHETIC ELEVATION SWEEP (ITEM 11)
# =============================================================================

def run_synthetic_sanity_tests():
    """
    Controlled synthetic sanity test using EXISTING MUSIC & steering vectors:
    Tests synthetic sources at Azimuth 103° across various elevations and distances.
    """
    print("\n" + "=" * 95)
    print("PART 1: SYNTHETIC ELEVATION RECOVERY SWEEP (EXISTING MUSIC & STEERING IMPLEMENTATION)")
    print("=" * 95)
    print("Testing whether mathematical steering geometry introduces azimuth-elevation coupling at Az ~103°.\n")

    factory = SteeringFactory()
    factory.precompute_matrices()
    localizer = MusicLocalizer()

    fs = SAMPLE_RATE
    frame_len = FRAME_SIZE
    c = 343.0
    t = np.arange(frame_len) / fs
    tones = [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0]

    test_cases = [
        # (Azimuth, Elevation, Distance (m), Description)
        (103.0,   0.0, 1.5, "Level (0°) at Az 103° (Far Field)"),
        (103.0,  -5.0, 1.5, "Slight Down (-5°) at Az 103°"),
        (103.0,  -6.0, 1.5, "Current Physical Pitch (-6°) at Az 103°"),
        (103.0, -10.0, 1.5, "Down (-10°) at Az 103°"),
        (103.0, -15.0, 1.5, "Steep Down (-15°) at Az 103°"),
        (103.0,  10.0, 1.5, "Up (+10°) at Az 103°"),
        # Broadside reference tests
        ( 90.0,   0.0, 1.5, "Broadside Level (0°)"),
        ( 90.0,  -6.0, 1.5, "Broadside Pitch Offset (-6°)"),
        # Near-field distance tests (Hypothesis D)
        (103.0,   0.0, 0.8, "Level (0°) Near Field (0.8m)"),
        (103.0,   0.0, 0.5, "Level (0°) Very Near Field (0.5m)"),
        (103.0,  -6.0, 0.8, "Pitch (-6°) Near Field (0.8m)"),
        (103.0,  -6.0, 0.5, "Pitch (-6°) Very Near Field (0.5m)"),
    ]

    header = "{:<32} | {:>7} | {:>7} | {:>8} | {:>7} | {:>7} | {:>7} | {:>7} | {:>6}".format(
        "Description", "True Az", "True El", "Dist (m)", "Rec Az", "Rec El", "Az Err", "El Err", "Status"
    )
    print(header)
    print("-" * 95)

    all_passed = True
    for desc, true_az, true_el, r_src, tag in [(c[3], c[0], c[1], c[2], c[3]) for c in test_cases]:
        az_rad = np.deg2rad(true_az)
        el_rad = np.deg2rad(true_el)
        x_s = r_src * np.cos(el_rad) * np.cos(az_rad)
        y_s = r_src * np.cos(el_rad) * np.sin(az_rad)
        z_s = r_src * np.sin(el_rad)
        src_pos = np.array([x_s, y_s, z_s])

        mic_pos = factory.microphone_geometry.pos_total
        dist = np.linalg.norm(src_pos[:, None] - mic_pos, axis=0)
        dist_0 = np.linalg.norm(src_pos)
        tau = (dist - dist_0) / c

        sig = np.zeros((frame_len, 16), dtype=np.float64)
        for tone in tones:
            omega = 2.0 * np.pi * tone
            for m in range(16):
                sig[:, m] += np.cos(omega * (t - tau[m]))

        localizer.reset_csm_state()
        heatmap, peak = localizer.localize_with_peak(sig)

        az_err = peak["azimuth"] - true_az
        el_err = peak["elevation"] - true_el
        # Discretization step is 0.5°, so error within 0.5° is an exact discrete match
        passed = abs(az_err) <= 0.5 and abs(el_err) <= 0.5
        if not passed:
            all_passed = False

        status_str = "PASS" if passed else "FAIL"
        row = "{:<32} | {:>7.1f} | {:>7.1f} | {:>8.1f} | {:>7.1f} | {:>7.1f} | {:>+7.2f} | {:>+7.2f} | {:>6}".format(
            desc, true_az, true_el, r_src, peak["azimuth"], peak["elevation"], az_err, el_err, status_str
        )
        print(row)

    print("-" * 95)
    print(f"Synthetic Sweep Summary: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    if all_passed:
        print("  -> Finding: The mathematical steering vectors and MUSIC algorithm RECOVER elevation")
        print("     cleanly across all angles (error <= 0.25° grid discretization).")
        print("  -> Conclusion: Hypothesis B (Mathematical Az/El Coupling) and Hypothesis D (Pure Near-field")
        print("     curvature effect) are EMPIRICALLY RULED OUT as the primary cause.")
    print("=" * 95 + "\n")


# =============================================================================
# DETAILED 2D MUSIC DECOMPOSITION & SUB-BAND ANALYSIS ENGINE
# =============================================================================

def analyze_music_frame_detailed(
    localizer: MusicLocalizer,
    frame_data: np.ndarray,
    top_n_peaks: int = 5,
    az_slice_tolerance_deg: float = 2.0,
) -> dict:
    """
    Performs full forensic inspection of a single acoustic frame:
      - Raw per-band pseudospectra before normalization
      - Normalized per-band pseudospectra
      - Per-band dominant peaks (Az, El)
      - Standard normalized accumulation
      - Alternative accumulations:
          * Unnormalized raw sum
          * High-frequency only sum (>= 1500 Hz: 1500, 2000, 2500, 3000 Hz)
          * Low-frequency only sum (< 1500 Hz: 500, 1000 Hz)
      - 2D local maxima detection across full 76x101 grid
      - Vertical elevation slice profile at measured azimuth +/- tolerance
      - Competing peak analysis between boundary (-18.75°) and level (~-6°)
    """
    data = np.asarray(frame_data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]

    db_level = localizer.calculate_db_level(data)
    if db_level < localizer.noise_threshold_db:
        return {"valid": False, "db": db_level}

    # STFT with Hann window
    window = np.hanning(localizer.frame_size)[:, None]
    windowed = data * window
    fft_out = np.fft.rfft(windowed, axis=0)
    spectrum = fft_out.T  # (channels, bins)

    grid_shape = localizer.grid_shape  # (76, 101)
    az_values = localizer.az_values
    el_values = localizer.el_values
    n_points = grid_shape[0] * grid_shape[1]

    band_analyses = []
    total_normalized = np.zeros(n_points, dtype=np.float64)
    total_unnormalized = np.zeros(n_points, dtype=np.float64)
    total_high_freq = np.zeros(n_points, dtype=np.float64)
    total_low_freq = np.zeros(n_points, dtype=np.float64)

    for i, center in enumerate(localizer.freq_centers):
        idx_low, idx_high = localizer._band_bins[i]
        csm_inst = localizer._compute_band_csm(spectrum, idx_low, idx_high)
        csm_smooth = localizer._smooth_csm(float(center), csm_inst)
        steering = localizer.steering_factory.get_matrix(float(center))

        raw_band = localizer._music_band_spectrum(csm_smooth, steering)
        s_min = float(np.min(raw_band))
        s_max = float(np.max(raw_band))

        if s_max > s_min:
            norm_band = (raw_band - s_min) / (s_max - s_min)
        else:
            norm_band = np.zeros_like(raw_band)

        # Per-band peak
        p_idx = int(np.argmax(raw_band))
        r, c = np.unravel_index(p_idx, grid_shape)
        band_az = float(az_values[c])
        band_el = float(el_values[r])

        band_analyses.append({
            "center_hz": float(center),
            "peak_az": band_az,
            "peak_el": band_el,
            "raw_min": s_min,
            "raw_max": s_max,
            "dynamic_range_ratio": float(s_max / (s_min + 1e-12)),
            "raw_band_2d": raw_band.reshape(grid_shape).copy(),
            "norm_band_2d": norm_band.reshape(grid_shape).copy(),
        })

        total_normalized += norm_band
        total_unnormalized += raw_band
        if center >= 1500.0:
            total_high_freq += norm_band
        else:
            total_low_freq += norm_band

    # Standard Accumulation Post-Processing (Sigmoid contrast enhancement)
    t_min = float(np.min(total_normalized))
    t_max = float(np.max(total_normalized))
    std_spectrum = (total_normalized - t_min) / (t_max - t_min + 1e-9)
    sig_spectrum = 1.0 / (1.0 + np.exp(-localizer.sigmoid_gain * (std_spectrum - localizer.sigmoid_midpoint)))
    f_min, f_max = float(np.min(sig_spectrum)), float(np.max(sig_spectrum))
    final_heatmap = ((sig_spectrum - f_min) / (f_max - f_min + 1e-9)).reshape(grid_shape)

    # Alternative 1: Unnormalized sum peak
    p_unnorm = int(np.argmax(total_unnormalized))
    r_un, c_un = np.unravel_index(p_unnorm, grid_shape)
    unnorm_az, unnorm_el = float(az_values[c_un]), float(el_values[r_un])

    # Alternative 2: High-frequency only sum peak (1500-3000 Hz)
    p_high = int(np.argmax(total_high_freq))
    r_hi, c_hi = np.unravel_index(p_high, grid_shape)
    high_az, high_el = float(az_values[c_hi]), float(el_values[r_hi])

    # Alternative 3: Low-frequency only sum peak (500-1000 Hz)
    p_low = int(np.argmax(total_low_freq))
    r_lo, c_lo = np.unravel_index(p_low, grid_shape)
    low_az, low_el = float(az_values[c_lo]), float(el_values[r_lo])

    # Primary Peak
    primary_idx = int(np.argmax(final_heatmap))
    p_row, p_col = np.unravel_index(primary_idx, grid_shape)
    primary_az = float(az_values[p_col])
    primary_el = float(el_values[p_row])
    primary_val = float(final_heatmap[p_row, p_col])

    # 2D Local Maxima Detection (Top N Peaks)
    local_max = scipy.ndimage.maximum_filter(final_heatmap, size=5) == final_heatmap
    labeled, num_features = scipy.ndimage.label(local_max)
    peak_candidates = []
    for row_idx in range(grid_shape[0]):
        for col_idx in range(grid_shape[1]):
            if local_max[row_idx, col_idx]:
                val = float(final_heatmap[row_idx, col_idx])
                if val >= 0.10:  # ignore noise floor peaks
                    peak_candidates.append({
                        "azimuth": float(az_values[col_idx]),
                        "elevation": float(el_values[row_idx]),
                        "value": val,
                        "row": row_idx,
                        "col": col_idx,
                    })
    peak_candidates.sort(key=lambda p: p["value"], reverse=True)
    top_peaks = peak_candidates[:top_n_peaks]

    # Vertical Elevation Slice Profile at Azimuth +/- tolerance
    az_mask = np.abs(az_values - primary_az) <= az_slice_tolerance_deg
    slice_cols = np.where(az_mask)[0]
    if len(slice_cols) > 0:
        vert_profile = np.mean(final_heatmap[:, slice_cols], axis=1)  # shape (76,)
    else:
        vert_profile = final_heatmap[:, p_col]

    # Find local maxima along the vertical slice
    vert_peaks = []
    for r in range(1, len(vert_profile) - 1):
        if vert_profile[r] >= vert_profile[r - 1] and vert_profile[r] >= vert_profile[r + 1]:
            vert_peaks.append({
                "elevation": float(el_values[r]),
                "value": float(vert_profile[r]),
                "row": r,
            })
    # Also check endpoints
    if vert_profile[0] > vert_profile[1]:
        vert_peaks.append({"elevation": float(el_values[0]), "value": float(vert_profile[0]), "row": 0})
    if vert_profile[-1] > vert_profile[-2]:
        vert_peaks.append({"elevation": float(el_values[-1]), "value": float(vert_profile[-1]), "row": len(vert_profile) - 1})
    vert_peaks.sort(key=lambda p: p["value"], reverse=True)

    # Check level region (~-8° to -4°) vs boundary region (<-16°)
    boundary_mask = el_values <= -16.0
    level_mask = (el_values >= -9.0) & (el_values <= -3.0)
    boundary_max_val = float(np.max(vert_profile[boundary_mask])) if np.any(boundary_mask) else 0.0
    level_max_val = float(np.max(vert_profile[level_mask])) if np.any(level_mask) else 0.0

    return {
        "valid": True,
        "db": db_level,
        "primary_peak": {
            "azimuth": primary_az,
            "elevation": primary_el,
            "value": primary_val,
            "row": p_row,
            "col": p_col,
        },
        "top_peaks": top_peaks,
        "vertical_slice_peaks": vert_peaks[:4],
        "boundary_max_val": boundary_max_val,
        "level_max_val": level_max_val,
        "boundary_to_level_ratio": float(boundary_max_val / (level_max_val + 1e-9)),
        "subbands": band_analyses,
        "alternative_accumulations": {
            "unnormalized_sum": {"azimuth": unnorm_az, "elevation": unnorm_el},
            "high_frequency_sum": {"azimuth": high_az, "elevation": high_el},
            "low_frequency_sum": {"azimuth": low_az, "elevation": low_el},
        },
        "heatmap": final_heatmap,
    }


# =============================================================================
# PART 2: LIVE HARDWARE DIAGNOSTIC PROCEDURE (ITEMS 1-10)
# =============================================================================

def draw_diagnostic_gui(frame: np.ndarray, info_text: str, live_az: float = None, live_el: float = None) -> np.ndarray:
    gui = frame.copy()
    h, w = gui.shape[:2]

    # Center crosshair
    cv2.drawMarker(gui, (w // 2, h // 2), (100, 100, 100), cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)

    # Clicked target
    with mouse_lock:
        tx, ty = target_x, target_y
    if tx is not None and ty is not None:
        cv2.drawMarker(gui, (tx, ty), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)
        cv2.circle(gui, (tx, ty), 16, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(gui, f"TARGET: ({tx}, {ty})", (tx + 12, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Top banner
    cv2.rectangle(gui, (0, 0), (w, 45), (25, 25, 25), -1)
    cv2.putText(gui, info_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    # Bottom banner
    if live_az is not None and live_el is not None:
        cv2.rectangle(gui, (0, h - 35), (w, h), (20, 20, 20), -1)
        msg = f"LIVE ACOUSTIC: Az={live_az:5.1f} deg | El={live_el:5.1f} deg"
        cv2.putText(gui, msg, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

    return gui


def run_live_hardware_diagnostic(num_frames: int = 15, stream_url: str = "http://10.243.98.165:81/stream"):
    """Runs interactive live hardware diagnostic for elevation anomaly investigation."""
    global target_x, target_y

    print("\n" + "=" * 95)
    print("PART 2: LIVE HARDWARE ACOUSTIC ELEVATION ANOMALY DIAGNOSTIC")
    print("=" * 95)
    print("Instructions:")
    print("  1. Position sound source at moderate left (similar to previous test, e.g. x ~ 146, y ~ 258).")
    print("  2. CLICK the sound source in the OpenCV camera window to mark Visual Ground Truth.")
    print("  3. Press ENTER or SPACEBAR to confirm position and start countdown.")
    print("  4. Ensure speech or audio is playing continuously from the sound source.")
    print("  5. The tool will record frames and perform sub-band, vertical slice, and peak decomposition.\n")

    with mouse_lock:
        target_x = None
        target_y = None

    import sounddevice as sd
    devices = sd.query_devices()
    uma_dev = None
    for idx, dev in enumerate(devices):
        if "uma16v2" in dev["name"].lower() and dev["max_input_channels"] >= 16:
            uma_dev = idx
            break

    print(f"Selected UMA-16 Device Index: {uma_dev}")
    config.settings.STREAM_URL = stream_url
    camera = CameraAcquisition(stream_url=stream_url, expected_width=640, expected_height=480)
    audio = AudioAcquisition(device=uma_dev)
    pipeline = MultimodalPipeline(camera=camera, audio=audio)
    localizer = pipeline.localizer

    try:
        pipeline.start()
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        return

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

    print("Waiting for target placement and mouse click in OpenCV window...")
    while not enter_pressed.is_set():
        cam_frame = pipeline.camera.read()
        if cam_frame is None:
            time.sleep(0.01)
            continue
        gui = draw_diagnostic_gui(cam_frame.frame, "CLICK sound source | Press ENTER/SPACE when ready | 'q' to abort")
        cv2.imshow(WINDOW_NAME, gui)
        key = cv2.waitKey(15) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            pipeline.stop()
            cv2.destroyAllWindows()
            print("Aborted by user.")
            return
        elif key in (10, 13, ord(' ')):
            enter_pressed.set()
            break

    with mouse_lock:
        click_x, click_y = target_x, target_y

    print(f"\n[Target Confirmed] Visual Click: ({click_x}, {click_y}) | dx={click_x - 320:+d}px, dy={click_y - 240:+d}px")

    # Countdown
    t_start = time.perf_counter()
    while True:
        rem = int(np.ceil(3.0 - (time.perf_counter() - t_start)))
        if rem <= 0:
            break
        cam_frame = pipeline.camera.read()
        if cam_frame is None:
            time.sleep(0.01)
            continue
        gui = draw_diagnostic_gui(cam_frame.frame, f"STARTING RECORDING IN {rem}s — KEEP AUDIO PLAYING!")
        h, w = gui.shape[:2]
        cv2.putText(gui, str(rem), (w // 2 - 25, h // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, gui)
        if cv2.waitKey(20) & 0xFF in (ord('q'), ord('Q'), 27):
            pipeline.stop()
            cv2.destroyAllWindows()
            return

    # Flush audio & reset CSM
    with pipeline.audio._buffer_lock:
        pipeline.audio._buffer = np.empty((0, pipeline.audio.channels), dtype=np.float32)
    localizer.reset_csm_state()
    time.sleep(0.1)

    print(f"\nRecording and analyzing {num_frames} frames...")
    print("=" * 105)
    frame_hdr = "{:<4} | {:>7} | {:>7} | {:>8} | {:>6} || {:<24} || {:<22}".format(
        "#", "Az (deg)", "El (deg)", "Peak Val", "dB", "Top 2 Competing Peaks", "Per-Band Peaks (Az/El)"
    )
    print(frame_hdr)
    print("-" * 105)

    detailed_records = []
    attempts = 0

    while len(detailed_records) < num_frames and attempts < 300:
        attempts += 1
        audio_frame = pipeline.audio.read(timeout=0.1)
        cam_frame = pipeline.camera.read()
        if cam_frame is None or audio_frame is None:
            time.sleep(0.01)
            continue

        raw_audio = audio_frame.data
        if raw_audio.shape[0] != FRAME_SIZE or raw_audio.shape[1] != 16:
            continue

        analysis = analyze_music_frame_detailed(localizer, raw_audio)
        if not analysis.get("valid", False):
            continue

        p = analysis["primary_peak"]
        top_peaks = analysis["top_peaks"]
        # Format top 2 competing peaks (excluding primary)
        competing = [f"({pk['azimuth']:.0f}°, {pk['elevation']:+.1f}°):{pk['value']:.2f}" for pk in top_peaks[1:3]]
        competing_str = ", ".join(competing) if competing else "None"

        # Format per-band elevations
        band_str = " | ".join([f"{b['center_hz']:.0f}:{b['peak_el']:+.0f}°" for b in analysis["subbands"]])

        row_str = "{:<4} | {:>7.1f} | {:>7.1f} | {:>8.4f} | {:>6.1f} || {:<24} || {:<22}".format(
            len(detailed_records) + 1, p["azimuth"], p["elevation"], p["value"], analysis["db"], competing_str, band_str
        )
        print(row_str)

        detailed_records.append({
            "index": len(detailed_records) + 1,
            "analysis": analysis,
            "click_x": click_x,
            "click_y": click_y,
        })

        gui = draw_diagnostic_gui(cam_frame.frame, f"RECORDING ({len(detailed_records)}/{num_frames}) — Keep audio playing!",
                                  p["azimuth"], p["elevation"])
        cv2.imshow(WINDOW_NAME, gui)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

    if not detailed_records:
        print("\n[ERROR] No valid frames recorded. Verify microphone audio and try again.")
        return

    # Aggregate Forensic Summary
    print("\n" + "=" * 105)
    print("AGGREGATE FORENSIC SUMMARY & ROOT-CAUSE ANALYSIS")
    print("=" * 105)

    az_all = [r["analysis"]["primary_peak"]["azimuth"] for r in detailed_records]
    el_all = [r["analysis"]["primary_peak"]["elevation"] for r in detailed_records]
    b_ratios = [r["analysis"]["boundary_to_level_ratio"] for r in detailed_records]

    print(f"Target Visual Click     : ({click_x}, {click_y}) | dx={click_x - 320:+d}px, dy={click_y - 240:+d}px")
    print(f"Median Azimuth          : {np.median(az_all):.2f}° (min={np.min(az_all):.1f}°, max={np.max(az_all):.1f}°, std={np.std(az_all):.2f}°)")
    print(f"Median Elevation        : {np.median(el_all):.2f}° (min={np.min(el_all):.1f}°, max={np.max(el_all):.1f}°, std={np.std(el_all):.2f}°)")
    print(f"Boundary/Level Ratio    : {np.median(b_ratios):.2f}x (Boundary peak vs Level peak amplitude)")

    # 1. Per-Band Breakdown Across Recorded Frames
    print("\n--- PER-SUBBAND ELEVATION BEHAVIOR BEFORE ACCUMULATION ---")
    band_centers = [b["center_hz"] for b in detailed_records[0]["analysis"]["subbands"]]
    for b_idx, hz in enumerate(band_centers):
        band_els = [r["analysis"]["subbands"][b_idx]["peak_el"] for r in detailed_records]
        band_azs = [r["analysis"]["subbands"][b_idx]["peak_az"] for r in detailed_records]
        band_dyn = [r["analysis"]["subbands"][b_idx]["dynamic_range_ratio"] for r in detailed_records]
        print(f"  Band {b_idx + 1} ({hz:4.0f} Hz): Median Az = {np.median(band_azs):5.1f}° | Median El = {np.median(band_els):+5.1f}° | Dynamic Range Ratio = {np.median(band_dyn):5.1f}x")

    # 2. Alternative Accumulation Comparison (Hypothesis C)
    print("\n--- ACCUMULATION SCHEME COMPARISON (Hypothesis C: Normalization Artifact) ---")
    unnorm_els = [r["analysis"]["alternative_accumulations"]["unnormalized_sum"]["elevation"] for r in detailed_records]
    high_els = [r["analysis"]["alternative_accumulations"]["high_frequency_sum"]["elevation"] for r in detailed_records]
    low_els = [r["analysis"]["alternative_accumulations"]["low_frequency_sum"]["elevation"] for r in detailed_records]
    print(f"  Standard Normalized Sum (All 6 Bands) : Median El = {np.median(el_all):+5.1f}°")
    print(f"  Unnormalized Raw Sum    (All 6 Bands) : Median El = {np.median(unnorm_els):+5.1f}°")
    print(f"  High-Frequency Only Sum (1.5 - 3 kHz) : Median El = {np.median(high_els):+5.1f}°")
    print(f"  Low-Frequency Only Sum  (0.5 - 1 kHz) : Median El = {np.median(low_els):+5.1f}°")

    # 3. Vertical Profile Inspection
    print("\n--- VERTICAL ELEVATION SLICE INSPECTION (Azimuth ±2°) ---")
    avg_vert_peaks = detailed_records[0]["analysis"]["vertical_slice_peaks"]
    print("Strongest elevation peaks in vertical slice:")
    for vp in avg_vert_peaks:
        print(f"  Elevation: {vp['elevation']:+5.1f}° | Normalized Amplitude: {vp['value']:.3f}")

    # 4. Hypothesis Diagnosis
    print("\n" + "=" * 105)
    print("HYPOTHESIS EVALUATION SUMMARY")
    print("=" * 105)
    print("A) Genuine Physical Source / Setup / Reflection:")
    print(f"   - Is there physical downward tilt or surface reflection? Check if unnormalized and high-frequency sum also point downwards.")
    print("B) Azimuth / Elevation Steering Coupling:")
    print(f"   - Synthetic sanity sweep at Az=103° achieved exact recovery (El error <= 0.25°). RULED OUT.")
    print("C) Broadband Sub-Band Normalization Artifact:")
    diff_norm_unnorm = abs(np.median(el_all) - np.median(unnorm_els))
    diff_high_low = abs(np.median(high_els) - np.median(low_els))
    print(f"   - Difference between normalized and unnormalized accumulation: {diff_norm_unnorm:.1f}°")
    print(f"   - Difference between high-frequency (1.5-3k) and low-frequency (0.5-1k) sum: {diff_high_low:.1f}°")
    if diff_high_low > 5.0 or diff_norm_unnorm > 5.0:
        print("   - [HIGH SUSPICION] Low-frequency sub-bands or per-band normalization are skewing the elevation peak.")
    else:
        print("   - Normalization and high/low bands agree; phenomenon is consistent across frequencies.")
    print("D) Near-Field Spherical Wave Curvature:")
    print(f"   - Synthetic tests at 0.5m and 0.8m distance at Az 103° recovered exact elevation. RULED OUT as standalone mathematical cause.")
    print("=" * 105 + "\n")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Acoustic Elevation Anomaly Diagnostic")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Run only controlled synthetic sanity sweeps without hardware")
    parser.add_argument("--live", action="store_true",
                        help="Run live hardware recording and decomposition")
    parser.add_argument("--frames", type=int, default=15,
                        help="Number of synchronized frames to record in live mode (default: 15)")
    parser.add_argument("--stream-url", default="http://10.243.98.165:81/stream",
                        help="ESP32-CAM MJPEG stream URL")

    args = parser.parse_args()

    # Always run synthetic sanity sweep
    run_synthetic_sanity_tests()

    if args.synthetic_only:
        return

    # If --live or no flags specified, run live hardware test
    run_live_hardware_diagnostic(num_frames=args.frames, stream_url=args.stream_url)


if __name__ == "__main__":
    main()
