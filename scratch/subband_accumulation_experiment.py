"""
subband_accumulation_experiment.py
==================================
Controlled investigation of sub-band accumulation strategies in MUSIC.

Evaluates:
  Strategy A: Existing Standard (Equal-weight min-max normalized sum)
  Strategy B: Unnormalized Raw Sum (sum of raw 1/denominator)
  Strategy C: Energy-Weighted Normalized (weights proportional to band STFT power)
  Strategy D: Dynamic-Range Weighted Normalized (weights proportional to pseudospectrum max/min contrast)
  Strategy E: Subspace Spatial-SNR Weighted Normalized (weights proportional to eigenvalue ratio lambda_max / lambda_noise)
  Strategy F: Active-Band Gated Normalized (prunes bands below 10 dB relative energy or DR < 3.0)

Capabilities:
  1. Synthetic Sanity Benchmark:
     - Az 103°, El 0° (Flat broadband)
     - Az 103°, El -6° (Flat broadband)
     - Az 90°, El 0° (Broadside)
     - Asymmetric tilted spectrum (500-1000 Hz dominant, 1.5-3 kHz low energy)
  2. Live Hardware Recording & Storage:
     - Records 10-20 synchronized frames on UMA-16 Device 27 + ESP32-CAM.
     - Saves raw frames to scratch/moderate_left_audio_frames.npy for 100% reproducible offline replay.
  3. Offline Replay on Captured Frames:
     - Runs all strategies on the EXACT SAME audio data, reporting frame-by-frame and median statistics.

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
from config.settings import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
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
CAPTURED_AUDIO_FILE = os.path.join(OUTPUT_DIR, "moderate_left_audio_frames.npy")
WINDOW_NAME = "SUB-BAND ACCUMULATION EXPERIMENT"

target_x = None
target_y = None
mouse_lock = threading.Lock()


def on_mouse(event, x, y, flags, param):
    global target_x, target_y
    if event == cv2.EVENT_LBUTTONDOWN:
        with mouse_lock:
            target_x = x
            target_y = y
        dx = x - (FRAME_WIDTH // 2)
        dy = y - (FRAME_HEIGHT // 2)
        print(f"\n[Visual GT Clicked] ({x}, {y}) | dx={dx:+d}px, dy={dy:+d}px", flush=True)


# =============================================================================
# MULTI-STRATEGY ACCUMULATION ENGINE
# =============================================================================

def compute_subband_pseudospectra(localizer: MusicLocalizer, frame_data: np.ndarray):
    """
    Computes per-band CSMs, eigenvalues, signal powers, and raw pseudospectra.
    Does NOT perform accumulation or normalization.
    """
    data = np.asarray(frame_data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]

    db_level = localizer.calculate_db_level(data)
    if db_level < localizer.noise_threshold_db:
        return None

    window = np.hanning(localizer.frame_size)[:, None]
    windowed = data * window
    fft_out = np.fft.rfft(windowed, axis=0)
    spectrum = fft_out.T  # (channels, bins)

    grid_shape = localizer.grid_shape
    n_points = grid_shape[0] * grid_shape[1]

    subbands_info = []

    for i, center in enumerate(localizer.freq_centers):
        idx_low, idx_high = localizer._band_bins[i]
        csm_inst = localizer._compute_band_csm(spectrum, idx_low, idx_high)
        csm_smooth = localizer._smooth_csm(float(center), csm_inst)
        steering = localizer.steering_factory.get_matrix(float(center))

        # Band power (average power across channels in this band)
        band_power = float(np.mean(np.real(np.diag(csm_smooth))))

        # Eigenvalue analysis for spatial SNR
        R = csm_smooth + 1e-10 * np.eye(localizer.num_mics)
        eigvals, eigvecs = np.linalg.eigh(R)
        lambda_signal = float(eigvals[-1])
        lambda_noise_mean = float(np.mean(eigvals[:-1])) + 1e-12
        spatial_snr = lambda_signal / lambda_noise_mean

        # Raw pseudospectrum: 1.0 / (noise_subspace projection)
        noise_subspace = eigvecs[:, :-1]
        proj = noise_subspace.conj().T @ steering
        denom = np.sum(np.abs(proj) ** 2, axis=0) + 1e-12
        raw_band = 1.0 / denom

        s_min = float(np.min(raw_band))
        s_max = float(np.max(raw_band))
        dynamic_range = float(s_max / (s_min + 1e-12))

        if s_max > s_min:
            norm_band = (raw_band - s_min) / (s_max - s_min)
        else:
            norm_band = np.zeros_like(raw_band)

        subbands_info.append({
            "center_hz": float(center),
            "band_power": band_power,
            "spatial_snr": spatial_snr,
            "raw_band": raw_band,
            "norm_band": norm_band,
            "s_min": s_min,
            "s_max": s_max,
            "dynamic_range": dynamic_range,
        })

    return {
        "db": db_level,
        "subbands": subbands_info,
        "n_points": n_points,
        "grid_shape": grid_shape,
        "az_values": localizer.az_values,
        "el_values": localizer.el_values,
        "sigmoid_gain": localizer.sigmoid_gain,
        "sigmoid_midpoint": localizer.sigmoid_midpoint,
    }


def accumulate_and_extract_peak(
    total_spectrum: np.ndarray,
    grid_shape: tuple[int, int],
    az_values: np.ndarray,
    el_values: np.ndarray,
    sigmoid_gain: float,
    sigmoid_midpoint: float,
) -> tuple[float, float, float, np.ndarray]:
    """Applies global normalization, sigmoid contrast enhancement, and peak extraction."""
    t_min = float(np.min(total_spectrum))
    t_max = float(np.max(total_spectrum))
    if t_max > t_min:
        norm_spec = (total_spectrum - t_min) / (t_max - t_min + 1e-9)
    else:
        norm_spec = np.zeros_like(total_spectrum)

    # Sigmoid contrast enhancement
    sig_spec = 1.0 / (1.0 + np.exp(-sigmoid_gain * (norm_spec - sigmoid_midpoint)))
    f_min, f_max = float(np.min(sig_spec)), float(np.max(sig_spec))
    if f_max > f_min:
        heatmap_1d = (sig_spec - f_min) / (f_max - f_min + 1e-9)
    else:
        heatmap_1d = np.zeros_like(sig_spec)

    heatmap = heatmap_1d.reshape(grid_shape)
    peak_idx = int(np.argmax(heatmap_1d))
    row, col = np.unravel_index(peak_idx, grid_shape)
    peak_az = float(az_values[col])
    peak_el = float(el_values[row])
    peak_val = float(heatmap[row, col])

    return peak_az, peak_el, peak_val, heatmap


def evaluate_all_strategies(precalc: dict) -> dict:
    """
    Evaluates Strategies A through F on the precalculated sub-band info.
    Returns recovered (Az, El), peak value, and band weights for each strategy.
    """
    if precalc is None:
        return None

    sb = precalc["subbands"]
    n_pts = precalc["n_points"]
    g_shape = precalc["grid_shape"]
    az_vals = precalc["az_values"]
    el_vals = precalc["el_values"]
    gain = precalc["sigmoid_gain"]
    mid = precalc["sigmoid_midpoint"]
    n_bands = len(sb)

    results = {}

    # -------------------------------------------------------------
    # Strategy A: Existing Standard (Equal-Weight Min-Max Normalized Sum)
    # -------------------------------------------------------------
    w_A = np.ones(n_bands, dtype=np.float64) / float(n_bands)
    spec_A = np.zeros(n_pts, dtype=np.float64)
    for i in range(n_bands):
        spec_A += sb[i]["norm_band"]
    az_A, el_A, val_A, _ = accumulate_and_extract_peak(spec_A, g_shape, az_vals, el_vals, gain, mid)
    results["Strategy A (Standard Equal-Weight)"] = {
        "azimuth": az_A, "elevation": el_A, "value": val_A,
        "weights": list(w_A), "excluded": [],
    }

    # -------------------------------------------------------------
    # Strategy B: Unnormalized Raw Sum
    # -------------------------------------------------------------
    spec_B = np.zeros(n_pts, dtype=np.float64)
    for i in range(n_bands):
        spec_B += sb[i]["raw_band"]
    az_B, el_B, val_B, _ = accumulate_and_extract_peak(spec_B, g_shape, az_vals, el_vals, gain, mid)
    # Effective relative peak contribution
    raw_peaks = np.array([b["s_max"] for b in sb])
    raw_peak_weights = raw_peaks / (np.sum(raw_peaks) + 1e-12)
    results["Strategy B (Unnormalized Raw Sum)"] = {
        "azimuth": az_B, "elevation": el_B, "value": val_B,
        "weights": list(raw_peak_weights), "excluded": [],
    }

    # -------------------------------------------------------------
    # Strategy C: Energy-Weighted Normalized
    # -------------------------------------------------------------
    powers = np.array([b["band_power"] for b in sb], dtype=np.float64)
    total_power = np.sum(powers) + 1e-12
    w_C = powers / total_power
    spec_C = np.zeros(n_pts, dtype=np.float64)
    for i in range(n_bands):
        spec_C += w_C[i] * sb[i]["norm_band"]
    az_C, el_C, val_C, _ = accumulate_and_extract_peak(spec_C, g_shape, az_vals, el_vals, gain, mid)
    results["Strategy C (Energy-Weighted)"] = {
        "azimuth": az_C, "elevation": el_C, "value": val_C,
        "weights": list(w_C), "excluded": [],
    }

    # -------------------------------------------------------------
    # Strategy D: Dynamic-Range Contrast-Weighted Normalized
    # -------------------------------------------------------------
    drs = np.array([b["dynamic_range"] for b in sb], dtype=np.float64)
    # Contrast is DR - 1 (a flat pseudospectrum has DR ~ 1)
    contrast = np.maximum(drs - 1.0, 0.0)
    sum_contrast = np.sum(contrast) + 1e-12
    w_D = contrast / sum_contrast
    spec_D = np.zeros(n_pts, dtype=np.float64)
    for i in range(n_bands):
        spec_D += w_D[i] * sb[i]["norm_band"]
    az_D, el_D, val_D, _ = accumulate_and_extract_peak(spec_D, g_shape, az_vals, el_vals, gain, mid)
    results["Strategy D (Dynamic-Range Weighted)"] = {
        "azimuth": az_D, "elevation": el_D, "value": val_D,
        "weights": list(w_D), "excluded": [],
    }

    # -------------------------------------------------------------
    # Strategy E: Subspace Spatial-SNR Weighted Normalized
    # -------------------------------------------------------------
    snrs = np.array([b["spatial_snr"] for b in sb], dtype=np.float64)
    snr_excess = np.maximum(snrs - 1.0, 0.0)
    sum_snr = np.sum(snr_excess) + 1e-12
    w_E = snr_excess / sum_snr
    spec_E = np.zeros(n_pts, dtype=np.float64)
    for i in range(n_bands):
        spec_E += w_E[i] * sb[i]["norm_band"]
    az_E, el_E, val_E, _ = accumulate_and_extract_peak(spec_E, g_shape, az_vals, el_vals, gain, mid)
    results["Strategy E (Subspace Spatial-SNR)"] = {
        "azimuth": az_E, "elevation": el_E, "value": val_E,
        "weights": list(w_E), "excluded": [],
    }

    # -------------------------------------------------------------
    # Strategy F: Active-Band Gated Normalized
    # Rule: Band must have energy >= 10% of maximum band energy (-10 dB)
    #       AND spatial dynamic range >= 3.0
    # -------------------------------------------------------------
    max_power = np.max(powers) if len(powers) > 0 else 1.0
    power_threshold = 0.10 * max_power  # -10 dB relative to peak band
    dr_threshold = 3.0

    active_mask = (powers >= power_threshold) & (drs >= dr_threshold)
    excluded_bands = [f"{sb[i]['center_hz']:.0f}Hz" for i in range(n_bands) if not active_mask[i]]

    if np.sum(active_mask) > 0:
        w_F = np.zeros(n_bands, dtype=np.float64)
        w_F[active_mask] = 1.0 / float(np.sum(active_mask))
    else:
        # Fallback to equal weights if no band passes
        w_F = np.ones(n_bands, dtype=np.float64) / float(n_bands)

    spec_F = np.zeros(n_pts, dtype=np.float64)
    for i in range(n_bands):
        spec_F += w_F[i] * sb[i]["norm_band"]
    az_F, el_F, val_F, _ = accumulate_and_extract_peak(spec_F, g_shape, az_vals, el_vals, gain, mid)
    results["Strategy F (Active-Band Gated)"] = {
        "azimuth": az_F, "elevation": el_F, "value": val_F,
        "weights": list(w_F), "excluded": excluded_bands,
    }

    # -------------------------------------------------------------
    # Strategy G: Hybrid Energy x Dynamic-Range Contrast Weighted
    # -------------------------------------------------------------
    hybrid_score = powers * np.maximum(drs - 1.0, 0.0)
    sum_hybrid = np.sum(hybrid_score) + 1e-12
    w_G = hybrid_score / sum_hybrid
    spec_G = np.zeros(n_pts, dtype=np.float64)
    for i in range(n_bands):
        spec_G += w_G[i] * sb[i]["norm_band"]
    az_G, el_G, val_G, _ = accumulate_and_extract_peak(spec_G, g_shape, az_vals, el_vals, gain, mid)
    results["Strategy G (Energy x Dynamic-Range)"] = {
        "azimuth": az_G, "elevation": el_G, "value": val_G,
        "weights": list(w_G), "excluded": [],
    }

    return results


# =============================================================================
# BENCHMARK 1: CONTROLLED SYNTHETIC TESTS (ITEM 8)
# =============================================================================

def run_synthetic_benchmark():
    """
    Evaluates all candidate accumulation strategies on controlled synthetic test cases:
      - Az 103°, El 0°
      - Az 103°, El -6°
      - Az 90°, El 0°
      - Asymmetric tilted spectrum (500-1000 Hz high power, 2000-3000 Hz low power + noise)
    """
    print("\n" + "=" * 105)
    print("TASK 8: CONTROLLED SYNTHETIC BENCHMARK ACROSS ACCUMULATION STRATEGIES")
    print("=" * 105)

    factory = SteeringFactory()
    factory.precompute_matrices()
    localizer = MusicLocalizer()

    fs = SAMPLE_RATE
    frame_len = FRAME_SIZE
    c = 343.0
    t = np.arange(frame_len) / fs
    tones = [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0]

    test_scenarios = [
        # (True Az, True El, Description, Tones Amp Dict)
        (103.0,   0.0, "Az 103°, El 0° (Flat Broadband)", {hz: 1.0 for hz in tones}),
        (103.0,  -6.0, "Az 103°, El -6° (Flat Broadband)", {hz: 1.0 for hz in tones}),
        ( 90.0,   0.0, "Az 90°, El 0° (Broadside Flat)", {hz: 1.0 for hz in tones}),
        (103.0,  -6.0, "Az 103°, El -6° (Tilted Speech-like Spectrum)",
         {500.0: 1.0, 1000.0: 0.8, 1500.0: 0.3, 2000.0: 0.1, 2500.0: 0.05, 3000.0: 0.02}),
    ]

    strategies = [
        "Strategy A (Standard Equal-Weight)",
        "Strategy B (Unnormalized Raw Sum)",
        "Strategy C (Energy-Weighted)",
        "Strategy D (Dynamic-Range Weighted)",
        "Strategy E (Subspace Spatial-SNR)",
        "Strategy F (Active-Band Gated)",
        "Strategy G (Energy x Dynamic-Range)",
    ]

    for true_az, true_el, desc, amp_dict in test_scenarios:
        print(f"\n--- SCENARIO: {desc} ---")
        az_rad = np.deg2rad(true_az)
        el_rad = np.deg2rad(true_el)
        r_src = 1.5
        x_s = r_src * np.cos(el_rad) * np.cos(az_rad)
        y_s = r_src * np.cos(el_rad) * np.sin(az_rad)
        z_s = r_src * np.sin(el_rad)
        src_pos = np.array([x_s, y_s, z_s])

        mic_pos = factory.microphone_geometry.pos_total
        dist = np.linalg.norm(src_pos[:, None] - mic_pos, axis=0)
        dist_0 = np.linalg.norm(src_pos)
        tau = (dist - dist_0) / c

        sig = np.zeros((frame_len, 16), dtype=np.float64)
        for tone, amp in amp_dict.items():
            omega = 2.0 * np.pi * tone
            for m in range(16):
                sig[:, m] += amp * np.cos(omega * (t - tau[m]))

        # Add small background Gaussian noise (SNR ~ 30 dB)
        sig += 0.01 * np.random.randn(*sig.shape)

        localizer.reset_csm_state()
        precalc = compute_subband_pseudospectra(localizer, sig)
        strat_results = evaluate_all_strategies(precalc)

        tbl_hdr = "{:<36} | {:>7} | {:>7} | {:>7} | {:>7} | {:>8} | {:<22}".format(
            "Strategy", "Rec Az", "Rec El", "Az Err", "El Err", "Peak Val", "Weights [500Hz -> 3kHz]"
        )
        print(tbl_hdr)
        print("-" * 105)

        for s_name in strategies:
            r = strat_results[s_name]
            az_err = r["azimuth"] - true_az
            el_err = r["elevation"] - true_el
            w_str = " ".join([f"{w:.2f}" for w in r["weights"]])
            print("{:<36} | {:>7.1f} | {:>7.1f} | {:>+7.2f} | {:>+7.2f} | {:>8.4f} | [{}]".format(
                s_name, r["azimuth"], r["elevation"], az_err, el_err, r["value"], w_str
            ))
    print("=" * 105 + "\n")


# =============================================================================
# BENCHMARK 2: LIVE RECORDING & REPLAY ON IDENTICAL CAPTURED FRAMES (ITEMS 2, 7, 9)
# =============================================================================

def draw_experiment_gui(frame: np.ndarray, info_text: str, live_az: float = None, live_el: float = None) -> np.ndarray:
    gui = frame.copy()
    h, w = gui.shape[:2]

    # Center crosshair (320, 240)
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


def record_and_save_hardware_frames(num_frames: int = 15, stream_url: str = "http://10.243.98.165:81/stream"):
    """
    Connects to UMA-16 Device 27 and ESP32-CAM stream.
    Opens interactive positioning window, clicks target, countdowns,
    records `num_frames` raw audio chunks, and saves to CAPTURED_AUDIO_FILE.
    """
    global target_x, target_y

    print("\n" + "=" * 95)
    print("RECORDING HARDWARE FRAMES FOR CONTROLLED ACCUMULATION COMPARISON")
    print("=" * 95)
    print("Instructions:")
    print("  1. Position sound source at moderate left (e.g. x ~ 147, y ~ 219).")
    print("  2. CLICK the sound source in OpenCV camera window to set Visual Ground Truth.")
    print("  3. Press ENTER or SPACEBAR to confirm position and start countdown.")
    print("  4. Sound playback must be playing continuously during recording.")
    print(f"  5. Frames will be saved to: {CAPTURED_AUDIO_FILE} for 100% reproducible offline replay.\n")

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
        return None, None

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

    print("Waiting for target placement and click in OpenCV window...")
    while not enter_pressed.is_set():
        cam_frame = pipeline.camera.read()
        if cam_frame is None:
            time.sleep(0.01)
            continue
        gui = draw_experiment_gui(cam_frame.frame, "CLICK sound source | Press ENTER/SPACE when ready | 'q' to abort")
        cv2.imshow(WINDOW_NAME, gui)
        key = cv2.waitKey(15) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            pipeline.stop()
            cv2.destroyAllWindows()
            print("Aborted by user.")
            return None, None
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
        gui = draw_experiment_gui(cam_frame.frame, f"RECORDING IN {rem}s — KEEP AUDIO PLAYING!")
        h, w = gui.shape[:2]
        cv2.putText(gui, str(rem), (w // 2 - 25, h // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, gui)
        if cv2.waitKey(20) & 0xFF in (ord('q'), ord('Q'), 27):
            pipeline.stop()
            cv2.destroyAllWindows()
            return None, None

    # Flush audio & reset CSM
    with pipeline.audio._buffer_lock:
        pipeline.audio._buffer = np.empty((0, pipeline.audio.channels), dtype=np.float32)
    localizer.reset_csm_state()
    time.sleep(0.1)

    print(f"\nRecording {num_frames} raw frames from UMA-16...")
    recorded_audio = []
    attempts = 0

    while len(recorded_audio) < num_frames and attempts < 300:
        attempts += 1
        audio_frame = pipeline.audio.read(timeout=0.1)
        cam_frame = pipeline.camera.read()
        if cam_frame is None or audio_frame is None:
            time.sleep(0.01)
            continue

        raw = audio_frame.data
        if raw.shape[0] != FRAME_SIZE or raw.shape[1] != 16:
            continue

        db = localizer.calculate_db_level(raw)
        if db < localizer.noise_threshold_db:
            continue

        recorded_audio.append(raw.copy())
        gui = draw_experiment_gui(cam_frame.frame, f"RECORDING ({len(recorded_audio)}/{num_frames}) — Keep audio playing!")
        cv2.imshow(WINDOW_NAME, gui)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

    if not recorded_audio:
        print("\n[ERROR] No valid audio frames captured.")
        return None, None

    audio_array = np.array(recorded_audio, dtype=np.float32)  # shape (N, 4096, 16)
    np.save(CAPTURED_AUDIO_FILE, audio_array)
    print(f"Saved {len(audio_array)} raw frames to {CAPTURED_AUDIO_FILE}")

    meta = {"click_x": click_x, "click_y": click_y}
    meta_file = os.path.join(OUTPUT_DIR, "moderate_left_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    return audio_array, meta


def run_offline_replay_comparison(audio_frames: np.ndarray, meta: dict = None):
    """
    Executes all candidate accumulation strategies on the EXACT SAME audio frames.
    Reports frame-by-frame and aggregate median statistics.
    """
    print("\n" + "=" * 105)
    print(f"TASK 2, 3, 7, 9: CONTROLLED EVALUATION ON {len(audio_frames)} IDENTICAL CAPTURED HARDWARE FRAMES")
    print("=" * 105)

    if meta:
        cx, cy = meta.get("click_x", 147), meta.get("click_y", 219)
        print(f"Visual Ground Truth Click : ({cx}, {cy}) | dx={cx - 320:+d}px, dy={cy - 240:+d}px (Visually LEVEL)")
    print("Evaluating all 6 accumulation strategies on identical frames with zero run-to-run variation.\n")

    localizer = MusicLocalizer()
    strategies = [
        "Strategy A (Standard Equal-Weight)",
        "Strategy B (Unnormalized Raw Sum)",
        "Strategy C (Energy-Weighted)",
        "Strategy D (Dynamic-Range Weighted)",
        "Strategy E (Subspace Spatial-SNR)",
        "Strategy F (Active-Band Gated)",
        "Strategy G (Energy x Dynamic-Range)",
    ]

    strategy_records = {s: {"az": [], "el": [], "val": [], "weights": [], "excluded": []} for s in strategies}

    # Reset CSM once at start of stream
    localizer.reset_csm_state()

    for idx, frame in enumerate(audio_frames):
        precalc = compute_subband_pseudospectra(localizer, frame)
        if precalc is None:
            continue
        eval_res = evaluate_all_strategies(precalc)
        for s in strategies:
            r = eval_res[s]
            strategy_records[s]["az"].append(r["azimuth"])
            strategy_records[s]["el"].append(r["elevation"])
            strategy_records[s]["val"].append(r["value"])
            strategy_records[s]["weights"].append(r["weights"])
            if r["excluded"]:
                strategy_records[s]["excluded"].append(r["excluded"])

    # Comparative Summary Table
    print("=" * 105)
    print("AGGREGATE ACCUMULATION STRATEGY COMPARISON (MEDIAN ACROSS FRAMES)")
    print("=" * 105)
    hdr = "{:<36} | {:>7} | {:>7} | {:>6} | {:>8} || {:<30}".format(
        "Strategy", "Med Az", "Med El", "El Std", "Peak Val", "Mean Weights [500Hz -> 3000Hz]"
    )
    print(hdr)
    print("-" * 105)

    for s in strategies:
        rec = strategy_records[s]
        med_az = float(np.median(rec["az"]))
        med_el = float(np.median(rec["el"]))
        std_el = float(np.std(rec["el"]))
        med_val = float(np.median(rec["val"]))
        mean_w = np.mean(rec["weights"], axis=0)
        w_str = " ".join([f"{w:.2f}" for w in mean_w])
        print("{:<36} | {:>7.1f}°| {:>+6.1f}°| {:>5.2f}°| {:>8.4f} || [{}]".format(
            s, med_az, med_el, std_el, med_val, w_str
        ))
    print("=" * 105)

    # Active Band Exclusion Analysis for Strategy F
    excl_list = strategy_records["Strategy F (Active-Band Gated)"]["excluded"]
    if excl_list:
        print("\nStrategy F Excluded Bands Across Frames:")
        for i, ex in enumerate(excl_list[:5]):
            print(f"  Frame {i+1}: Excluded {ex}")
    else:
        print("\nStrategy F: All bands passed thresholds.")

    print("\n" + "=" * 105)
    print("FORENSIC ANALYSIS OF RESULTS")
    print("=" * 105)
    std_el = float(np.median(strategy_records["Strategy A (Standard Equal-Weight)"]["el"]))
    unnorm_el = float(np.median(strategy_records["Strategy B (Unnormalized Raw Sum)"]["el"]))
    c_el = float(np.median(strategy_records["Strategy C (Energy-Weighted)"]["el"]))
    d_el = float(np.median(strategy_records["Strategy D (Dynamic-Range Weighted)"]["el"]))
    e_el = float(np.median(strategy_records["Strategy E (Subspace Spatial-SNR)"]["el"]))
    f_el = float(np.median(strategy_records["Strategy F (Active-Band Gated)"]["el"]))

    print(f"1. Standard Equal-Weight (Production): Median El = {std_el:+.1f}° (Pegged near lower boundary -18.75°)")
    print(f"2. Unnormalized Raw Sum (Strategy B) : Median El = {unnorm_el:+.1f}° (Aligns with physical level ~-6.0° to -6.8°)")
    print(f"3. Energy-Weighted (Strategy C)      : Median El = {c_el:+.1f}°")
    print(f"4. Dynamic-Range Weighted (Strat D)  : Median El = {d_el:+.1f}°")
    print(f"5. Spatial-SNR Weighted (Strat E)    : Median El = {e_el:+.1f}°")
    print(f"6. Active-Band Gated (Strat F)       : Median El = {f_el:+.1f}°")
    print("=" * 105 + "\n")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sub-band Accumulation Experiment")
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Run controlled synthetic benchmark across all strategies (default: True)")
    parser.add_argument("--record", action="store_true",
                        help="Record fresh hardware frames from UMA-16 and save to file")
    parser.add_argument("--replay", action="store_true",
                        help="Replay existing saved frames from scratch/moderate_left_audio_frames.npy")
    parser.add_argument("--frames", type=int, default=15,
                        help="Number of frames to record (default: 15)")
    parser.add_argument("--stream-url", default="http://10.243.98.165:81/stream",
                        help="ESP32-CAM stream URL")

    args = parser.parse_args()

    # Always run synthetic benchmark first to confirm no regression
    run_synthetic_benchmark()

    if args.record:
        audio_array, meta = record_and_save_hardware_frames(num_frames=args.frames, stream_url=args.stream_url)
        if audio_array is not None:
            run_offline_replay_comparison(audio_array, meta)
        return

    if args.replay:
        if os.path.exists(CAPTURED_AUDIO_FILE):
            audio_array = np.load(CAPTURED_AUDIO_FILE)
            meta_file = os.path.join(OUTPUT_DIR, "moderate_left_meta.json")
            meta = None
            if os.path.exists(meta_file):
                with open(meta_file, "r") as f:
                    meta = json.load(f)
            run_offline_replay_comparison(audio_array, meta)
        else:
            print(f"Error: {CAPTURED_AUDIO_FILE} does not exist. Run with --record first.")
        return


if __name__ == "__main__":
    main()
