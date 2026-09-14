"""
csm_settling_diagnostic.py
==========================
Forensic Investigation of Recording Duration, Audio Continuity,
and CSM Exponential Moving Average (EMA) Settling Dynamics in MUSIC.

Key Investigations:
  1. Audio Frame Acquisition Architecture & Continuity:
     - Measures sample rate (48 kHz), frame size (4096), hop size (2048).
     - Logs frame-by-frame timing: wall-clock timestamp, delta t (ms), cumulative time,
       audio stream time, and buffer overflow status.
     - Confirms whether frames are strictly consecutive (50% overlap, 42.67 ms hop).

  2. Mathematical EMA CSM Settling Analysis:
     - Documents EMA formula: CSM_smooth = (1 - alpha)*CSM_prev + alpha*CSM_inst (alpha = 0.3).
     - Tracks effective number of snapshots and memory decay time constant tau = 1/alpha = 3.33 frames.
     - Computes decay of initial transient: (1 - alpha)^(k-1).

  3. Long Controlled Recording (--duration 8):
     - Records 5 to 10 seconds of continuous audio (e.g. 8s = 187 frames).
     - Saves raw captured audio to: scratch/moderate_left_long_audio_frames.npy.

  4. Time-Slice Localization Tracking:
     - Evaluates localization across progressive time horizons:
       * Frame 1 (Instantaneous snapshot)
       * Frames 1–3 (0.0s – 0.17s)
       * Frames 1–5 (0.0s – 0.25s)
       * Frames 1–10 (0.0s – 0.47s)
       * Frames 1–20 (0.0s – 0.90s)
       * Frames 1–50 (0.0s – 2.18s)
       * Frames 1–100 (0.0s – 4.31s)
       * Final Steady-State Frames (e.g. frames 50–180)
     - Reports for each: median Az, median El, El std, per-band energy, and per-band Az/El.

  5. Warm-up Protocol Comparison:
     - Protocol A: Reset CSM and process from Frame 1
     - Protocol B: 1-Second Warm-up (allow EMA to stabilize for 25 frames, then evaluate)
     - Protocol C: Steady-State Only (evaluate exclusively on final frames)

  6. Multi-Strategy Evaluation Over Time:
     - Tracks Standard Equal-Weight (A) vs Unnormalized (B) vs Energy-Weighted (C).

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

root_dir = r"c:\Users\kushs\OneDrive\Desktop\Projects\multimodal_acoustic_camera"
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config.settings
from config.settings import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    SAMPLE_RATE,
    FRAME_SIZE,
    HOP_SIZE,
    FREQ_CENTERS,
    CSM_ALPHA,
    NOISE_THRESHOLD_DB,
)
from acoustic.steering import SteeringFactory
from acoustic.music import MusicLocalizer
from vision.camera import CameraAcquisition
from acoustic.audio import AudioAcquisition
from pipeline.multimodal import MultimodalPipeline

OUTPUT_DIR = os.path.dirname(__file__)
LONG_AUDIO_FILE = os.path.join(OUTPUT_DIR, "moderate_left_long_audio_frames.npy")
TIMING_LOG_FILE = os.path.join(OUTPUT_DIR, "csm_settling_timing.json")
WINDOW_NAME = "CSM SETTLING DIAGNOSTIC"

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
# MATHEMATICAL EMA SETTLING THEORY
# =============================================================================

def print_ema_theory():
    """Prints mathematical analysis of the CSM EMA filter."""
    alpha = CSM_ALPHA
    decay_base = 1.0 - alpha
    tau_frames = 1.0 / alpha
    tau_sec = tau_frames * (HOP_SIZE / float(SAMPLE_RATE))
    n_eff = (2.0 - alpha) / alpha

    print("\n" + "=" * 95)
    print("TASK 5: MATHEMATICAL ANALYSIS OF CSM EXPONENTIAL MOVING AVERAGE (EMA)")
    print("=" * 95)
    print(f"Sample Rate           : {SAMPLE_RATE} Hz")
    print(f"Analysis Frame Size   : {FRAME_SIZE} samples ({FRAME_SIZE / SAMPLE_RATE * 1000.0:.2f} ms)")
    print(f"Hop Size              : {HOP_SIZE} samples ({HOP_SIZE / SAMPLE_RATE * 1000.0:.2f} ms)")
    print(f"Frame Overlap         : {(1.0 - HOP_SIZE / FRAME_SIZE) * 100.0:.1f}%")
    print(f"EMA Smoothing Alpha   : {alpha:.2f} (config.settings.CSM_ALPHA)")
    print(f"Memory Time Constant  : tau = {tau_frames:.2f} frames ({tau_sec * 1000.0:.1f} ms)")
    print(f"Effective Averaging   : N_eff = {n_eff:.2f} frames")
    print("\nTransient Decay of Frame 1 Snapshot (weight = (1 - alpha)^(k-1)):")
    print("  Frame  1 ( 85 ms): weight = 100.0%  (Pure instantaneous snapshot, 0% averaging)")
    print(f"  Frame  3 (171 ms): weight = {(decay_base ** 2) * 100.0:5.1f}%")
    print(f"  Frame  5 (256 ms): weight = {(decay_base ** 4) * 100.0:5.1f}%")
    print(f"  Frame 10 (469 ms): weight = {(decay_base ** 9) * 100.0:5.1f}%  (< 5% initial transient)")
    print(f"  Frame 14 (640 ms): weight = {(decay_base ** 13) * 100.0:5.1f}%  (< 1% initial transient, near steady-state)")
    print(f"  Frame 25 (1.11 s): weight = {(decay_base ** 24) * 100.0:5.2f}%  (Fully settled steady-state)")
    print("-" * 95)
    print("Key Finding:")
    print("  A capture of only 15 frames spans exactly: (4096 + 14 * 2048) / 48000 = 0.683 seconds.")
    print("  During frames 1–5 (the first 250 ms), the CSM has had fewer than 5 averages and is")
    print("  dominated by the initial snapshot, room onset reflections, and sound source ramp-up.")
    print("  To observe steady-state MUSIC performance, a recording of >= 4–8 seconds is required.")
    print("=" * 95 + "\n")


# =============================================================================
# SYNTHETIC SETTLING SIMULATION
# =============================================================================

def run_synthetic_settling_simulation():
    """
    Simulates continuous audio of 5 seconds with speech-like energy distribution
    to observe how MUSIC Az/El and per-band energy evolve frame-by-frame from reset.
    """
    print("=" * 95)
    print("SYNTHETIC SETTLING BENCHMARK: 5 SECONDS CONTINUOUS AUDIO (Az 103°, El -6°)")
    print("=" * 95)

    factory = SteeringFactory()
    factory.precompute_matrices()
    localizer = MusicLocalizer()

    fs = SAMPLE_RATE
    frame_len = FRAME_SIZE
    hop_len = HOP_SIZE
    c = 343.0
    tones = [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0]

    # 4 seconds of audio = ~90 frames
    duration_sec = 4.0
    total_samples = int(duration_sec * fs)
    t_full = np.arange(total_samples) / fs

    r_src = 1.5
    az_rad = np.deg2rad(103.0)
    el_rad = np.deg2rad(-6.0)
    x_s = r_src * np.cos(el_rad) * np.cos(az_rad)
    y_s = r_src * np.cos(el_rad) * np.sin(az_rad)
    z_s = r_src * np.sin(el_rad)
    src_pos = np.array([x_s, y_s, z_s])

    mic_pos = factory.microphone_geometry.pos_total
    dist = np.linalg.norm(src_pos[:, None] - mic_pos, axis=0)
    dist_0 = np.linalg.norm(src_pos)
    tau = (dist - dist_0) / c

    # Tilted speech-like amplitudes
    amps = {500.0: 1.0, 1000.0: 0.6, 1500.0: 0.25, 2000.0: 0.10, 2500.0: 0.05, 3000.0: 0.02}
    sig_full = np.zeros((total_samples, 16), dtype=np.float64)
    for tone, amp in amps.items():
        omega = 2.0 * np.pi * tone
        for m in range(16):
            sig_full[:, m] += amp * np.cos(omega * (t_full - tau[m]))

    # Add realistic noise floor (SNR ~ 22 dB)
    sig_full += 0.08 * np.random.randn(*sig_full.shape)

    localizer.reset_csm_state()
    n_frames = (total_samples - frame_len) // hop_len

    checkpoints = [1, 3, 5, 10, 15, 25, 50, n_frames]
    checkpoints = [c for c in checkpoints if c <= n_frames]

    print("{:>5} | {:>10} | {:>7} | {:>7} | {:>8} | {:>7} | {:<25}".format(
        "Frame", "Stream (s)", "Rec Az", "Rec El", "Peak Val", "dB", "Sub-band Powers [500Hz -> 3k]"
    ))
    print("-" * 95)

    for i in range(n_frames):
        start_idx = i * hop_len
        frame = sig_full[start_idx : start_idx + frame_len, :]
        heatmap, peak = localizer.localize_with_peak(frame)
        audio_t = (start_idx + frame_len) / fs

        if (i + 1) in checkpoints:
            powers = []
            for b_idx in range(6):
                csm = localizer._smoothed_csms.get(float(tones[b_idx]))
                pwr = float(np.mean(np.real(np.diag(csm)))) if csm is not None else 0.0
                powers.append(pwr)
            p_tot = sum(powers) + 1e-12
            pw_str = " ".join([f"{p/p_tot:.2f}" for p in powers])
            print(f"{i+1:5d} | {audio_t:9.3f}s | {peak['azimuth']:6.1f}°| {peak['elevation']:+6.1f}°| {peak['value']:8.4f} | {peak['db']:6.1f} | [{pw_str}]")

    print("-" * 95 + "\n")


# =============================================================================
# TIMING DIAGNOSTIC & LONG RECORDING
# =============================================================================

def record_long_hardware_capture(duration_sec: float = 8.0, stream_url: str = "http://10.243.98.165:81/stream"):
    """
    Records a long continuous hardware capture (default 8s) with precise timing diagnostics:
      - Measures delta t between frames (expected 42.67 ms)
      - Checks for PortAudio buffer overflows / dropped frames
      - Saves raw frames to scratch/moderate_left_long_audio_frames.npy
    """
    global target_x, target_y

    target_frames = int(np.ceil((duration_sec * SAMPLE_RATE - FRAME_SIZE) / float(HOP_SIZE))) + 1

    print("\n" + "=" * 95)
    print(f"TASK 3, 4, 6: LONG CONTINUOUS RECORDING ({duration_sec:.1f}s = {target_frames} FRAMES)")
    print("=" * 95)
    print(f"Target Duration  : {duration_sec:.1f} seconds")
    print(f"Target Frames    : {target_frames} frames (4096 samples/frame, 2048 samples hop = 42.67 ms/frame)")
    print("Instructions:")
    print("  1. Position phone at moderate left (e.g. x ~ 147, y ~ 219).")
    print("  2. CLICK the sound source in OpenCV window to record visual ground truth.")
    print("  3. Press ENTER or SPACEBAR to confirm position and start countdown.")
    print("  4. Play continuous audio (speech or music) continuously for the full 8 seconds.")
    print(f"  5. Frames and timing will be saved to: {LONG_AUDIO_FILE}\n")

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
        gui = cam_frame.frame.copy()
        cv2.drawMarker(gui, (320, 240), (100, 100, 100), cv2.MARKER_CROSS, 20, 1)
        with mouse_lock:
            tx, ty = target_x, target_y
        if tx is not None and ty is not None:
            cv2.drawMarker(gui, (tx, ty), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 22, 2)
            cv2.putText(gui, f"TARGET: ({tx}, {ty})", (tx + 12, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(gui, (0, 0), (640, 45), (25, 25, 25), -1)
        cv2.putText(gui, f"LONG RECORDING ({duration_sec}s) | Click source | Press ENTER when ready", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
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
        gui = cam_frame.frame.copy()
        cv2.rectangle(gui, (0, 0), (640, 45), (25, 25, 25), -1)
        cv2.putText(gui, f"RECORDING IN {rem}s — PLAY AUDIO CONTINUOUSLY!", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(gui, str(rem), (295, 275), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 5)
        cv2.imshow(WINDOW_NAME, gui)
        if cv2.waitKey(20) & 0xFF in (ord('q'), ord('Q'), 27):
            pipeline.stop()
            cv2.destroyAllWindows()
            return None, None

    # Flush buffer & reset CSM
    with pipeline.audio._buffer_lock:
        pipeline.audio._buffer = np.empty((0, pipeline.audio.channels), dtype=np.float32)
    localizer.reset_csm_state()
    time.sleep(0.05)

    print(f"\nRecording {target_frames} consecutive frames (~{duration_sec:.1f}s)...")
    print(f"{'#':<4} | {'Wall Time (s)':<13} | {'Delta t (ms)':<12} | {'Cumulative (s)':<14} | {'Audio Time (s)':<14} | {'Status':<10}")
    print("-" * 80)

    recorded_audio = []
    timing_records = []
    t_record_start = time.perf_counter()
    t_prev = t_record_start
    overflow_start = pipeline.audio._overflow_count

    while len(recorded_audio) < target_frames:
        audio_frame = pipeline.audio.read(timeout=0.1)
        t_now = time.perf_counter()
        cam_frame = pipeline.camera.read()

        if audio_frame is None:
            time.sleep(0.005)
            continue

        raw = audio_frame.data
        if raw.shape[0] != FRAME_SIZE or raw.shape[1] != 16:
            continue

        dt_ms = (t_now - t_prev) * 1000.0
        cum_sec = t_now - t_record_start
        audio_time_sec = (FRAME_SIZE + len(recorded_audio) * HOP_SIZE) / float(SAMPLE_RATE)
        t_prev = t_now

        recorded_audio.append(raw.copy())

        # Check gap / jitter (nominal is 42.67 ms)
        jitter_flag = "OK"
        if dt_ms > 90.0:
            jitter_flag = "GAP / DELAY"

        timing_records.append({
            "frame_idx": len(recorded_audio),
            "timestamp": t_now,
            "delta_t_ms": dt_ms,
            "cumulative_sec": cum_sec,
            "audio_time_sec": audio_time_sec,
            "jitter": jitter_flag,
        })

        if len(recorded_audio) <= 10 or len(recorded_audio) % 25 == 0 or len(recorded_audio) == target_frames:
            print(f"{len(recorded_audio):<4} | {t_now:<13.3f} | {dt_ms:<12.2f} | {cum_sec:<14.3f} | {audio_time_sec:<14.3f} | {jitter_flag:<10}")

        if cam_frame is not None:
            gui = cam_frame.frame.copy()
            cv2.rectangle(gui, (0, 0), (640, 40), (25, 25, 25), -1)
            cv2.putText(gui, f"RECORDING ({len(recorded_audio)}/{target_frames}) | {cum_sec:.1f}s / {duration_sec:.1f}s", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, gui)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break

    pipeline.stop()
    cv2.destroyAllWindows()

    total_wall_sec = time.perf_counter() - t_record_start
    overflow_total = pipeline.audio._overflow_count - overflow_start

    print("-" * 80)
    print(f"Total Frames Acquired : {len(recorded_audio)} frames")
    print(f"Total Audio Samples   : {len(recorded_audio) * HOP_SIZE + (FRAME_SIZE - HOP_SIZE)} samples")
    print(f"Audio Stream Duration : {len(recorded_audio) * (HOP_SIZE / float(SAMPLE_RATE)):.3f} seconds")
    print(f"Total Wall-Clock Time : {total_wall_sec:.3f} seconds")
    print(f"Mean Delta t per Frame: {np.mean([t['delta_t_ms'] for t in timing_records[1:]]):.2f} ms (Nominal: 42.67 ms)")
    print(f"Buffer Overflows      : {overflow_total}")
    print(f"Audio Continuity      : {'STRICTLY CONSECUTIVE (Zero dropped frames)' if overflow_total == 0 else 'OVERFLOW DETECTED'}")

    audio_array = np.array(recorded_audio, dtype=np.float32)
    np.save(LONG_AUDIO_FILE, audio_array)
    print(f"Saved raw frames array to: {LONG_AUDIO_FILE}")

    meta = {
        "click_x": click_x,
        "click_y": click_y,
        "duration_sec": duration_sec,
        "total_frames": len(recorded_audio),
        "wall_clock_sec": total_wall_sec,
        "audio_stream_sec": len(recorded_audio) * (HOP_SIZE / float(SAMPLE_RATE)),
        "overflow_count": overflow_total,
    }
    with open(TIMING_LOG_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    return audio_array, meta


# =============================================================================
# SETTLING ANALYSIS OVER TIME SLICES (TASKS 8, 9, 10)
# =============================================================================

def analyze_csm_settling_timeline(audio_frames: np.ndarray, meta: dict = None):
    """
    Executes deep settling analysis across progressive time windows:
      - First frame
      - First 3 frames
      - First 5 frames
      - First 10 frames
      - First 20 frames
      - First 50 frames
      - First 100 frames
      - Final steady-state frames (e.g. frames 50 to end)

    Compares:
      - Standard Strategy A vs Unnormalized Strategy B vs Energy-Weighted Strategy C
      - Condition A: Continuous from reset
      - Condition B: 1-second warm-up (skipping first 25 frames)
      - Condition C: Steady-state only
    """
    total_frames = len(audio_frames)
    print("\n" + "=" * 105)
    print(f"TASKS 8, 9, 10: SETTLING ANALYSIS OVER TIME ({total_frames} FRAMES CAPTURED)")
    print("=" * 105)

    if meta:
        print(f"Visual Click GT: ({meta.get('click_x', 'N/A')}, {meta.get('click_y', 'N/A')}) | Visually LEVEL")
        print(f"Wall Clock Time: {meta.get('wall_clock_sec', 0):.2f}s | Audio Stream Duration: {meta.get('audio_stream_sec', 0):.2f}s\n")

    localizer = MusicLocalizer()
    localizer.reset_csm_state()

    # Step-by-step frame processing
    frame_results = []
    tones = [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0]

    for idx, frame in enumerate(audio_frames):
        # Extract per-band properties BEFORE accumulation
        window = np.hanning(FRAME_SIZE)[:, None]
        fft_out = np.fft.rfft(frame * window, axis=0)
        spectrum = fft_out.T

        band_powers = []
        band_raw_pseudos = []
        band_norm_pseudos = []
        band_peaks = []

        for b_idx, center in enumerate(localizer.freq_centers):
            idx_low, idx_high = localizer._band_bins[b_idx]
            csm_inst = localizer._compute_band_csm(spectrum, idx_low, idx_high)
            csm_smooth = localizer._smooth_csm(float(center), csm_inst)
            steering = localizer.steering_factory.get_matrix(float(center))

            pwr = float(np.mean(np.real(np.diag(csm_smooth))))
            band_powers.append(pwr)

            raw_band = localizer._music_band_spectrum(csm_smooth, steering)
            s_min, s_max = float(np.min(raw_band)), float(np.max(raw_band))
            norm_band = (raw_band - s_min) / (s_max - s_min + 1e-12) if s_max > s_min else np.zeros_like(raw_band)

            # Band peak
            bp_idx = int(np.argmax(raw_band))
            r_b, c_b = np.unravel_index(bp_idx, localizer.grid_shape)
            band_peaks.append({
                "az": float(localizer.az_values[c_b]),
                "el": float(localizer.el_values[r_b]),
                "dr": float(s_max / (s_min + 1e-12)),
            })

            band_raw_pseudos.append(raw_band)
            band_norm_pseudos.append(norm_band)

        # Accumulation Strategy A: Standard Equal-Weight Normalized
        spec_A = np.sum(band_norm_pseudos, axis=0)
        p_idx_A = np.unravel_index(np.argmax(spec_A), localizer.grid_shape)
        az_A, el_A = float(localizer.az_values[p_idx_A[1]]), float(localizer.el_values[p_idx_A[0]])

        # Accumulation Strategy B: Unnormalized Raw Sum
        spec_B = np.sum(band_raw_pseudos, axis=0)
        p_idx_B = np.unravel_index(np.argmax(spec_B), localizer.grid_shape)
        az_B, el_B = float(localizer.az_values[p_idx_B[1]]), float(localizer.el_values[p_idx_B[0]])

        # Accumulation Strategy C: Energy-Weighted Normalized
        w_C = np.array(band_powers) / (np.sum(band_powers) + 1e-12)
        spec_C = np.sum([w * nb for w, nb in zip(w_C, band_norm_pseudos)], axis=0)
        p_idx_C = np.unravel_index(np.argmax(spec_C), localizer.grid_shape)
        az_C, el_C = float(localizer.az_values[p_idx_C[1]]), float(localizer.el_values[p_idx_C[0]])

        # Accumulation Strategy D: Dynamic Range Weighted
        drs = np.array([bp["dr"] for bp in band_peaks])
        w_D = np.maximum(drs - 1.0, 0.0) / (np.sum(np.maximum(drs - 1.0, 0.0)) + 1e-12)
        spec_D = np.sum([w * nb for w, nb in zip(w_D, band_norm_pseudos)], axis=0)
        p_idx_D = np.unravel_index(np.argmax(spec_D), localizer.grid_shape)
        az_D, el_D = float(localizer.az_values[p_idx_D[1]]), float(localizer.el_values[p_idx_D[0]])

        audio_t = (FRAME_SIZE + idx * HOP_SIZE) / float(SAMPLE_RATE)
        frame_results.append({
            "frame": idx + 1,
            "audio_t": audio_t,
            "az_A": az_A, "el_A": el_A,
            "az_B": az_B, "el_B": el_B,
            "az_C": az_C, "el_C": el_C,
            "az_D": az_D, "el_D": el_D,
            "powers": band_powers,
            "band_peaks": band_peaks,
        })

    # PROGRESSIVE TIME WINDOWS REPORT
    print("--- PROGRESSIVE TIME-WINDOW SETTLING COMPARISON ---")
    windows = [
        ("Frame 1 (Instantaneous)", 0, 1),
        ("First 3 Frames (0.0s - 0.17s)", 0, min(3, total_frames)),
        ("First 5 Frames (0.0s - 0.25s)", 0, min(5, total_frames)),
        ("First 10 Frames (0.0s - 0.47s)", 0, min(10, total_frames)),
        ("First 15 Frames (0.0s - 0.68s)", 0, min(15, total_frames)),
        ("First 25 Frames (0.0s - 1.11s)", 0, min(25, total_frames)),
        ("First 50 Frames (0.0s - 2.18s)", 0, min(50, total_frames)),
        ("All Frames (Full Capture)", 0, total_frames),
        ("Steady-State Only (Frames 25 -> End)", min(25, total_frames - 1), total_frames),
        ("Final 50 Frames (Late Steady-State)", max(0, total_frames - 50), total_frames),
    ]

    header = "{:<36} | {:>9} | {:>9} | {:>9} | {:>9} | {:>7}".format(
        "Window / Condition", "Strat A El", "Strat B El", "Strat C El", "Strat D El", "El Std (A)"
    )
    print(header)
    print("-" * 95)

    for desc, start, end in windows:
        sub = frame_results[start:end]
        if not sub:
            continue
        el_A_med = np.median([f["el_A"] for f in sub])
        el_B_med = np.median([f["el_B"] for f in sub])
        el_C_med = np.median([f["el_C"] for f in sub])
        el_D_med = np.median([f["el_D"] for f in sub])
        el_A_std = np.std([f["el_A"] for f in sub])
        print("{:<36} | {:>+8.1f}° | {:>+8.1f}° | {:>+8.1f}° | {:>+8.1f}° | {:>6.2f}°".format(
            desc, el_A_med, el_B_med, el_C_med, el_D_med, el_A_std
        ))
    print("=" * 105)

    # PER-BAND EVOLUTION OVER TIME
    print("\n--- PER-BAND ELEVATION EVOLUTION OVER TIME (Initial vs Steady-State) ---")
    f_early = frame_results[min(4, len(frame_results)-1)]  # Frame 5
    f_late = frame_results[-1]                             # Final frame
    p_tot_early = sum(f_early["powers"]) + 1e-12
    p_tot_late = sum(f_late["powers"]) + 1e-12

    print("{:<8} | {:<25} || {:<25}".format("Band (Hz)", "Frame 5 (0.25s)", "Final Steady-State Frame"))
    print("{:<8} | {:>7} | {:>7} | {:>6} || {:>7} | {:>7} | {:>6}".format(
        "", "Peak Az", "Peak El", "Energy%", "Peak Az", "Peak El", "Energy%"
    ))
    print("-" * 75)
    for b_idx in range(6):
        hz = tones[b_idx]
        bp_e = f_early["band_peaks"][b_idx]
        bp_l = f_late["band_peaks"][b_idx]
        pe = (f_early["powers"][b_idx] / p_tot_early) * 100.0
        pl = (f_late["powers"][b_idx] / p_tot_late) * 100.0
        print(f"{hz:8.0f} | {bp_e['az']:6.1f}°| {bp_e['el']:+6.1f}°| {pe:5.1f}% || {bp_l['az']:6.1f}°| {bp_l['el']:+6.1f}°| {pl:5.1f}%")

    print("=" * 105 + "\n")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CSM Settling and Recording Duration Diagnostic")
    parser.add_argument("--theory", action="store_true", default=True,
                        help="Print mathematical EMA settling theory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run synthetic settling benchmark over 5 seconds")
    parser.add_argument("--record", action="store_true",
                        help="Record long hardware capture (default: 8.0s)")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="Recording duration in seconds (default: 8.0)")
    parser.add_argument("--replay", action="store_true",
                        help="Replay and analyze existing saved long capture")
    parser.add_argument("--stream-url", default="http://10.243.98.165:81/stream",
                        help="ESP32-CAM stream URL")

    args = parser.parse_args()

    # Always print EMA theory
    print_ema_theory()

    if args.synthetic:
        run_synthetic_settling_simulation()
        return

    if args.record:
        audio_array, meta = record_long_hardware_capture(duration_sec=args.duration, stream_url=args.stream_url)
        if audio_array is not None:
            analyze_csm_settling_timeline(audio_array, meta)
        return

    if args.replay:
        if os.path.exists(LONG_AUDIO_FILE):
            audio_array = np.load(LONG_AUDIO_FILE)
            meta = None
            if os.path.exists(TIMING_LOG_FILE):
                with open(TIMING_LOG_FILE, "r") as f:
                    meta = json.load(f)
            analyze_csm_settling_timeline(audio_array, meta)
        else:
            print(f"File {LONG_AUDIO_FILE} does not exist. Run with --record first.")
        return

    # If no record/replay requested, run synthetic settling simulation
    run_synthetic_settling_simulation()


if __name__ == "__main__":
    main()
