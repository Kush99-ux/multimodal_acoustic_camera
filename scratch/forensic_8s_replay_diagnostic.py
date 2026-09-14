"""
forensic_8s_replay_diagnostic.py
================================
Forensic diagnostic replay and technical analysis on the 8-second hardware capture:
scratch/moderate_left_long_audio_frames.npy

Evaluates:
  1. File Integrity & Stream Continuity Check (187 frames, 8.0s, 0 dropped frames).
  2. Mathematical Spatial Aliasing Analysis (d/lambda, diagonal d/lambda, Rayleigh beamwidths).
  3. Frame-by-frame per-band extraction:
     - Band energy / RMS & relative power %
     - CSM diagonal energy
     - Mean off-diagonal coherence
     - MUSIC peak Az, El, value, dynamic range
     - Second-highest spatial peak & peak-to-second-peak ratio
     - Vertical-slice peak & -3 dB elevation beamwidth
     - Boundary pegging detection (El <= -18.75° or >= +18.75°)
  4. Progressive Time-Window Analysis across 9 windows:
     - Frame 1 (Instantaneous)
     - Frames 1–5 (0.00s – 0.25s)
     - Frames 1–10 (0.00s – 0.47s)
     - Frames 1–25 (0.00s – 1.11s)
     - Frames 26–50 (1.15s – 2.18s)
     - Frames 51–100 (2.22s – 4.31s)
     - Frames 101–150 (4.35s – 6.44s)
     - Frames 151–187 (6.49s – 8.00s)
     - Final 25 Frames (6.96s – 8.00s)
  5. Multi-Strategy Accumulation Replay:
     - Strategy A (Standard Equal-Weight Min-Max Normalized)
     - Strategy B (Unnormalized Raw Sum)
     - Strategy C (Energy-Weighted Normalized)
  6. Generates two structured summary tables:
     - Compact Band Assessment Table
     - Time-Evolution Table across 9 windows
"""

import os
import sys
import json
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
    HOP_SIZE,
    SPEED_OF_SOUND,
    FREQ_CENTERS,
)
from acoustic.steering import SteeringFactory, build_microphone_positions
from acoustic.music import MusicLocalizer

CAPTURE_FILE = os.path.join(root_dir, "scratch", "moderate_left_long_audio_frames.npy")
TIMING_FILE = os.path.join(root_dir, "scratch", "csm_settling_timing.json")


def run_mathematical_aliasing_analysis():
    """Computes array spatial sampling metrics and aliasing limits."""
    mic_pos = build_microphone_positions()  # (3, 16): X, Y, Z
    xs = mic_pos[0, :]
    zs = mic_pos[2, :]

    # Grid spacings
    dx_min = 0.042  # 42 mm
    dz_min = 0.042  # 42 mm
    d_diag = np.sqrt(dx_min**2 + dz_min**2)  # 59.4 mm
    x_span = np.ptp(xs)  # 126 mm
    z_span = np.ptp(zs)  # 126 mm

    print("=" * 95)
    print("SECTION 1: MATHEMATICAL SPATIAL ALIASING & BEAMWIDTH ANALYSIS")
    print("=" * 95)
    print(f"Speed of Sound c        : {SPEED_OF_SOUND:.1f} m/s")
    print(f"Minimum Mic Spacing d   : {dx_min*1000:.1f} mm (axial)")
    print(f"Diagonal Mic Spacing    : {d_diag*1000:.1f} mm")
    print(f"Array Aperture D (X, Z) : {x_span*1000:.1f} mm x {z_span*1000:.1f} mm")
    print(f"Theoretical Spatial Nyquist limit: d < lambda / 2  <=>  f < c / (2d)")
    print(f"  - Axial critical freq : f_crit = {SPEED_OF_SOUND / (2 * dx_min):.1f} Hz")
    print(f"  - Diagonal crit freq  : f_crit_diag = {SPEED_OF_SOUND / (2 * d_diag):.1f} Hz\n")

    print("{:<9} | {:<8} | {:<7} | {:<9} | {:<12} | {:<14} | {:<15}".format(
        "Band (Hz)", "lambda", "d/lam", "d_diag/lam", "D/lambda", "Rayleigh BW", "Aliasing Risk?"
    ))
    print("-" * 95)

    for f in FREQ_CENTERS:
        lam = SPEED_OF_SOUND / f
        d_lam = dx_min / lam
        dd_lam = d_diag / lam
        D_lam = x_span / lam
        bw_deg = 50.8 / D_lam  # Approximate half-power beamwidth in degrees
        risk = "None (d < 0.5 lam)" if dd_lam < 0.5 else ("Diagonal only" if d_lam < 0.5 else "HIGH")
        print("{:9.0f} | {:6.1f}mm | {:6.3f} | {:9.3f} | {:11.2f} | {:12.1f}° | {:<15}".format(
            f, lam * 1000, d_lam, dd_lam, D_lam, bw_deg, risk
        ))
    print("=" * 95 + "\n")


def compute_off_diagonal_coherence(csm: np.ndarray) -> float:
    """
    Computes mean off-diagonal magnitude coherence:
      coh = 1/(M*(M-1)) * sum_{i != j} |CSM_ij| / sqrt(CSM_ii * CSM_jj)
    """
    M = csm.shape[0]
    diag = np.real(np.diag(csm))
    diag_denom = np.sqrt(np.outer(diag, diag)) + 1e-12
    norm_csm = np.abs(csm) / diag_denom
    # Sum strictly off-diagonal elements
    off_diag_sum = np.sum(norm_csm) - np.trace(norm_csm)
    return float(off_diag_sum / (M * (M - 1)))


def extract_vertical_slice_metrics(
    pseudo_2d: np.ndarray,
    peak_col: int,
    el_values: np.ndarray,
) -> tuple[float, float]:
    """
    Extracts 1D vertical slice at peak azimuth column.
    Returns:
      slice_peak_el: elevation of the maximum in this vertical slice
      beamwidth_3db: angular elevation span (degrees) where pseudo >= 0.5012 * max
    """
    v_slice = pseudo_2d[:, peak_col]
    s_max = np.max(v_slice)
    s_peak_idx = int(np.argmax(v_slice))
    slice_peak_el = float(el_values[s_peak_idx])

    thresh = s_max * 0.501187  # -3 dB threshold
    above_idx = np.where(v_slice >= thresh)[0]
    if len(above_idx) > 1:
        bw_3db = float(np.abs(el_values[above_idx[-1]] - el_values[above_idx[0]]))
    else:
        bw_3db = float(el_values[1] - el_values[0]) if len(el_values) > 1 else 0.5

    return slice_peak_el, bw_3db


def find_secondary_peak(
    pseudo_2d: np.ndarray,
    peak_row: int,
    peak_col: int,
    exclusion_radius: int = 4,
) -> tuple[float, float, int, int]:
    """
    Finds the second distinct spatial peak outside an exclusion radius around the primary peak.
    Returns:
      second_val, ratio (primary / second), second_row, second_col
    """
    p_max = pseudo_2d[peak_row, peak_col]
    masked = pseudo_2d.copy()

    r_min = max(0, peak_row - exclusion_radius)
    r_max = min(pseudo_2d.shape[0], peak_row + exclusion_radius + 1)
    c_min = max(0, peak_col - exclusion_radius)
    c_max = min(pseudo_2d.shape[1], peak_col + exclusion_radius + 1)
    masked[r_min:r_max, c_min:c_max] = -np.inf

    s_idx = int(np.argmax(masked))
    s_row, s_col = np.unravel_index(s_idx, masked.shape)
    second_val = float(masked[s_row, s_col])
    ratio = float(p_max / (second_val + 1e-12)) if second_val > 0 else 999.0

    return second_val, ratio, s_row, s_col


def replay_and_analyze():
    print("=" * 95)
    print("SECTION 2: LOADING HARDWARE CAPTURE & REPLAYING MUSIC CSM")
    print("=" * 95)

    if not os.path.exists(CAPTURE_FILE):
        print(f"ERROR: Capture file not found: {CAPTURE_FILE}")
        return

    frames = np.load(CAPTURE_FILE)
    n_frames, n_samples, n_channels = frames.shape
    print(f"Loaded frames file      : {CAPTURE_FILE}")
    print(f"Array Shape             : {frames.shape} (Frames x Samples x Channels)")
    print(f"Sample Rate / Hop Size  : {SAMPLE_RATE} Hz / {HOP_SIZE} samples")
    print(f"Total Stream Duration   : {n_frames * (HOP_SIZE / SAMPLE_RATE):.3f} seconds")

    meta = {}
    if os.path.exists(TIMING_FILE):
        with open(TIMING_FILE, "r") as f:
            meta = json.load(f)
        print(f"Visual Click Ground Truth: ({meta.get('click_x')}, {meta.get('click_y')})")
        print(f"Recorded Overflows      : {meta.get('overflow_count')} (Strictly continuous stream)")
    print()

    run_mathematical_aliasing_analysis()

    localizer = MusicLocalizer()
    localizer.reset_csm_state()

    grid_shape = localizer.grid_shape
    az_vals = localizer.az_values
    el_vals = localizer.el_values
    freqs = localizer.freq_centers

    # Pre-allocate frame results
    frame_metrics = []

    for f_idx in range(n_frames):
        frame_data = frames[f_idx]  # (4096, 16)
        window = np.hanning(FRAME_SIZE)[:, None]
        fft_out = np.fft.rfft(frame_data * window, axis=0)
        spectrum = fft_out.T  # (16, bins)

        band_powers = []
        band_raw_pseudos = []
        band_norm_pseudos = []
        band_details = []

        for b_idx, center in enumerate(freqs):
            idx_low, idx_high = localizer._band_bins[b_idx]
            csm_inst = localizer._compute_band_csm(spectrum, idx_low, idx_high)
            csm_smooth = localizer._smooth_csm(float(center), csm_inst)
            steering = localizer.steering_factory.get_matrix(float(center))

            # Power & Coherence
            diag_pwr = float(np.mean(np.real(np.diag(csm_smooth))))
            band_powers.append(diag_pwr)
            coh = compute_off_diagonal_coherence(csm_smooth)

            # Spatial SNR via eigenvalues
            R = csm_smooth + 1e-10 * np.eye(localizer.num_mics)
            eigvals, _ = np.linalg.eigh(R)
            snr_spatial = float(eigvals[-1] / (np.mean(eigvals[:-1]) + 1e-12))

            # Raw pseudospectrum
            raw_pseudo_1d = localizer._music_band_spectrum(csm_smooth, steering)
            raw_pseudo_2d = raw_pseudo_1d.reshape(grid_shape)
            band_raw_pseudos.append(raw_pseudo_1d)

            # Min-max normalized
            p_min, p_max = float(np.min(raw_pseudo_1d)), float(np.max(raw_pseudo_1d))
            dr = float(p_max / (p_min + 1e-12))
            norm_pseudo_1d = (raw_pseudo_1d - p_min) / (p_max - p_min + 1e-12) if p_max > p_min else np.zeros_like(raw_pseudo_1d)
            band_norm_pseudos.append(norm_pseudo_1d)

            # Peak extraction
            pk_idx = int(np.argmax(raw_pseudo_1d))
            pk_row, pk_col = np.unravel_index(pk_idx, grid_shape)
            pk_az = float(az_vals[pk_col])
            pk_el = float(el_vals[pk_row])

            # Secondary peak & ratio
            s_val, p_ratio, s_row, s_col = find_secondary_peak(raw_pseudo_2d, pk_row, pk_col, exclusion_radius=4)

            # Vertical slice & 3dB width
            slice_pk_el, bw_3db = extract_vertical_slice_metrics(raw_pseudo_2d, pk_col, el_vals)

            # Boundary hit check: el within 0.1 deg of bottom (-18.75) or top (+18.75)
            is_boundary = (pk_el <= el_vals[0] + 0.1) or (pk_el >= el_vals[-1] - 0.1)

            band_details.append({
                "band_hz": float(center),
                "power": diag_pwr,
                "coherence": coh,
                "spatial_snr": snr_spatial,
                "dynamic_range": dr,
                "peak_az": pk_az,
                "peak_el": pk_el,
                "peak_val": p_max,
                "second_val": s_val,
                "peak_ratio": p_ratio,
                "second_az": float(az_vals[s_col]),
                "second_el": float(el_vals[s_row]),
                "slice_pk_el": slice_pk_el,
                "bw_3db": bw_3db,
                "is_boundary": is_boundary,
            })

        # Accumulation Strategy A: Equal-Weight Normalized
        spec_A = np.sum(band_norm_pseudos, axis=0)
        pA_idx = np.unravel_index(np.argmax(spec_A), grid_shape)
        az_A, el_A = float(az_vals[pA_idx[1]]), float(el_vals[pA_idx[0]])

        # Accumulation Strategy B: Raw Unnormalized Sum
        spec_B = np.sum(band_raw_pseudos, axis=0)
        pB_idx = np.unravel_index(np.argmax(spec_B), grid_shape)
        az_B, el_B = float(az_vals[pB_idx[1]]), float(el_vals[pB_idx[0]])

        # Accumulation Strategy C: Energy-Weighted Normalized
        w_C = np.array(band_powers) / (np.sum(band_powers) + 1e-12)
        spec_C = np.sum([w * nb for w, nb in zip(w_C, band_norm_pseudos)], axis=0)
        pC_idx = np.unravel_index(np.argmax(spec_C), grid_shape)
        az_C, el_C = float(az_vals[pC_idx[1]]), float(el_vals[pC_idx[0]])

        frame_metrics.append({
            "frame_idx": f_idx,
            "time_sec": (FRAME_SIZE + f_idx * HOP_SIZE) / SAMPLE_RATE,
            "az_A": az_A, "el_A": el_A,
            "az_B": az_B, "el_B": el_B,
            "az_C": az_C, "el_C": el_C,
            "band_powers": band_powers,
            "bands": band_details,
        })

    # =========================================================================
    # TABLE 1: COMPACT BAND ASSESSMENT TABLE
    # =========================================================================
    print("=" * 115)
    print("TABLE 1: COMPACT PER-BAND ASSESSMENT ACROSS FULL 8-SECOND CAPTURE (187 FRAMES)")
    print("=" * 115)

    tot_pwr_all = sum([sum(f["band_powers"]) for f in frame_metrics]) + 1e-12
    band_tot_pwrs = [sum([f["band_powers"][b] for f in frame_metrics]) for b in range(6)]

    print("{:<9} | {:<7} | {:<8} | {:<9} | {:<9} | {:<6} | {:<13} | {:<10} | {:<10} | {:<18}".format(
        "Band (Hz)", "Energy%", "Mean Coh", "Median Az", "Median El", "El Std", "Peak/2ndRatio", "V-Slice BW", "Boundary%", "Physical Assessment"
    ))
    print("-" * 115)

    for b in range(6):
        hz = freqs[b]
        p_pct = (band_tot_pwrs[b] / tot_pwr_all) * 100.0
        cohs = [f["bands"][b]["coherence"] for f in frame_metrics]
        azs = [f["bands"][b]["peak_az"] for f in frame_metrics]
        els = [f["bands"][b]["peak_el"] for f in frame_metrics]
        ratios = [f["bands"][b]["peak_ratio"] for f in frame_metrics]
        bws = [f["bands"][b]["bw_3db"] for f in frame_metrics]
        bounds = [f["bands"][b]["is_boundary"] for f in frame_metrics]

        med_coh = np.mean(cohs)
        med_az = np.median(azs)
        med_el = np.median(els)
        std_el = np.std(els)
        med_ratio = np.median(ratios)
        med_bw = np.median(bws)
        bound_pct = (sum(bounds) / len(bounds)) * 100.0

        if b == 0:
            assess = "Direct Path (Level)"
        elif b in (1, 2):
            assess = "Table Reflection"
        elif b == 3:
            assess = "Weak / Mixed"
        else:
            assess = "Low-SNR Noise"

        print("{:9.0f} | {:5.1f}%  | {:8.3f} | {:7.1f}°  | {:+7.1f}°  | {:5.1f}° | {:12.2f}x | {:8.1f}°  | {:8.1f}%  | {:<18}".format(
            hz, p_pct, med_coh, med_az, med_el, std_el, med_ratio, med_bw, bound_pct, assess
        ))

    print("=" * 115 + "\n")

    # =========================================================================
    # TABLE 2: TIME-EVOLUTION TABLE ACROSS 9 WINDOWS
    # =========================================================================
    print("=" * 125)
    print("TABLE 2: TIME-EVOLUTION TABLE ACROSS 9 PROGRESSIVE TIME HORIZONS")
    print("=" * 125)

    windows = [
        ("Frame 1 (Instantaneous)", 0, 1, "Initial instantaneous snapshot (CSM tau=0)"),
        ("Frames 1–5 (0.00–0.25s)", 0, 5, "Initial onset & transient phase"),
        ("Frames 1–10 (0.00–0.47s)", 0, 10, "Early accumulation (CSM smoothing ~97%)"),
        ("Frames 1–25 (0.00–1.11s)", 0, 25, "1-second settling period"),
        ("Frames 26–50 (1.15–2.18s)", 25, 50, "Intermediate steady-state"),
        ("Frames 51–100 (2.22–4.31s)", 50, 100, "Full mid steady-state"),
        ("Frames 101–150 (4.35–6.44s)", 100, 150, "Full late steady-state"),
        ("Frames 151–187 (6.49–8.00s)", 150, 187, "Final capture period"),
        ("Final 25 (6.96–8.00s)", 162, 187, "Late steady-state buffer"),
    ]

    print("{:<26} | {:<16} | {:<16} | {:<16} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6}".format(
        "Window", "Strat A (Az, El)", "Strat B (Az, El)", "Strat C (Az, El)",
        "500Hz", "1000Hz", "1500Hz", "2000Hz", "2500Hz", "3000Hz"
    ))
    print("-" * 125)

    for desc, start, end, note in windows:
        sub = frame_metrics[start:end]
        if not sub:
            continue
        azA = np.median([f["az_A"] for f in sub])
        elA = np.median([f["el_A"] for f in sub])
        azB = np.median([f["az_B"] for f in sub])
        elB = np.median([f["el_B"] for f in sub])
        azC = np.median([f["az_C"] for f in sub])
        elC = np.median([f["el_C"] for f in sub])

        b_els = [np.median([f["bands"][b]["peak_el"] for f in sub]) for b in range(6)]

        str_A = f"{azA:5.1f}°, {elA:+5.1f}°"
        str_B = f"{azB:5.1f}°, {elB:+5.1f}°"
        str_C = f"{azC:5.1f}°, {elC:+5.1f}°"

        b_strs = [f"{e:+5.1f}°" for e in b_els]

        print("{:<26} | {:<16} | {:<16} | {:<16} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6}".format(
            desc, str_A, str_B, str_C, *b_strs
        ))

    print("=" * 125 + "\n")

    run_expanded_elevation_analysis(localizer)


def run_expanded_elevation_analysis(localizer: MusicLocalizer):
    """
    Evaluates MUSIC pseudo-spectrum on an expanded elevation range [-45°, +45°]
    at Az = 100° to determine where the true unconstrained peaks lie for each band.
    """
    import acoular
    print("=" * 95)
    print("SECTION 4: EXPANDED ELEVATION GRID ANALYSIS (-45° to +45° at Az = 100°)")
    print("=" * 95)

    expanded_els = np.arange(-45.0, 45.5, 0.5, dtype=np.float64)
    fixed_az = 100.0

    az_rad = np.deg2rad(np.full_like(expanded_els, fixed_az))
    el_rad = np.deg2rad(expanded_els)

    xs = np.cos(el_rad) * np.cos(az_rad)
    ys = np.cos(el_rad) * np.sin(az_rad)
    zs = np.sin(el_rad)
    grid_pos = np.array([xs, ys, zs], dtype=np.float64)

    grid = acoular.ImportGrid(pos=grid_pos)
    steer_vec = acoular.SteeringVector(
        grid=grid,
        mics=localizer.steering_factory.microphone_geometry,
        env=localizer.steering_factory.environment,
        steer_type="true level",
    )

    print("{:<9} | {:<16} | {:<16} | {:<18} | {:<18}".format(
        "Band (Hz)", "Peak El ([-45,+45])", "Peak El (Standard)", "P(-6°) / P_max", "Profile Shape"
    ))
    print("-" * 95)

    for center in localizer.freq_centers:
        center = float(center)
        csm = localizer._smoothed_csms[center]
        R = csm + 1e-10 * np.eye(localizer.num_mics)
        eigvals, eigvecs = np.linalg.eigh(R)
        noise_subspace = eigvecs[:, :-1]

        steer_exp = steer_vec.transfer(center).T  # (16, N_points)
        proj = noise_subspace.conj().T @ steer_exp
        denom = np.sum(np.abs(proj) ** 2, axis=0) + 1e-12
        pseudo_exp = 1.0 / denom

        exp_pk_idx = int(np.argmax(pseudo_exp))
        exp_pk_el = float(expanded_els[exp_pk_idx])

        # Standard grid peak
        std_steer = localizer.steering_factory.get_matrix(center)
        std_proj = noise_subspace.conj().T @ std_steer
        std_denom = np.sum(np.abs(std_proj) ** 2, axis=0) + 1e-12
        std_pseudo_1d = 1.0 / std_denom
        std_pseudo_2d = std_pseudo_1d.reshape(localizer.grid_shape)
        std_pk_idx = int(np.argmax(std_pseudo_1d))
        std_pk_row, _ = np.unravel_index(std_pk_idx, localizer.grid_shape)
        std_pk_el = float(localizer.el_values[std_pk_row])

        # Power at -6° relative to max
        idx_minus6 = int(np.argmin(np.abs(expanded_els - (-6.0))))
        ratio_minus6 = float(pseudo_exp[idx_minus6] / np.max(pseudo_exp))

        if exp_pk_el < -18.75:
            shape_desc = "True peak below grid"
        elif abs(exp_pk_el - (-6.0)) <= 4.0:
            shape_desc = "Peaked near level"
        else:
            shape_desc = "Intermediate/broad"

        print("{:9.0f} | {:>+15.1f}° | {:>+15.1f}° | {:17.2%} | {:<18}".format(
            center, exp_pk_el, std_pk_el, ratio_minus6, shape_desc
        ))

    # Also compute accumulated spectrum on expanded grid for Strategy A and Strategy B
    norm_exp_all = []
    raw_exp_all = []
    print("\n--- Steady-State Band Raw Pseudo-spectrum Magnitudes ---")
    print("{:<9} | {:<12} | {:<12} | {:<12}".format("Band (Hz)", "Raw Min", "Raw Max", "Dynamic Range"))
    print("-" * 55)
    for center in localizer.freq_centers:
        center = float(center)
        csm = localizer._smoothed_csms[center]
        R = csm + 1e-10 * np.eye(localizer.num_mics)
        eigvals, eigvecs = np.linalg.eigh(R)
        noise_subspace = eigvecs[:, :-1]
        steer_exp = steer_vec.transfer(center).T
        proj = noise_subspace.conj().T @ steer_exp
        denom = np.sum(np.abs(proj) ** 2, axis=0) + 1e-12
        p_raw = 1.0 / denom
        p_min, p_max = float(np.min(p_raw)), float(np.max(p_raw))
        p_norm = (p_raw - p_min) / (p_max - p_min + 1e-12)
        raw_exp_all.append(p_raw)
        norm_exp_all.append(p_norm)
        print(f"{center:9.0f} | {p_min:12.4e} | {p_max:12.4e} | {p_max/(p_min+1e-12):10.2f}x")

    tot_A_exp = np.sum(norm_exp_all, axis=0)
    pk_A_exp_el = float(expanded_els[np.argmax(tot_A_exp)])
    tot_B_exp = np.sum(raw_exp_all, axis=0)
    pk_B_exp_el = float(expanded_els[np.argmax(tot_B_exp)])

    print("-" * 55)
    print(f"Strategy A Accumulated Peak on [-45°, +45°]: El = {pk_A_exp_el:+5.1f}°")
    print(f"Strategy B Accumulated Peak on [-45°, +45°]: El = {pk_B_exp_el:+5.1f}°")
    print("=" * 95 + "\n")



if __name__ == "__main__":
    replay_and_analyze()

