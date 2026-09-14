"""
Real-Time Acoustic Camera — Week 1 Milestone
=============================================
Uses a 16-channel microphone array + MUSIC algorithm to localise sound sources
in azimuth/elevation, then overlays a heatmap on a live camera feed.

Keyboard controls (active while the OpenCV window is focused):
  1 — Low frequency band    (300–1000 Hz)
  2 — Speech band           (500–3000 Hz)  ← default
  3 — High frequency band   (2000–4000 Hz)
  q — Quit

Key tunable parameters are grouped at the top for easy experimentation.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Prevent OpenBLAS from spawning extra threads

import sounddevice as sd
import matplotlib.pyplot as plt
import cv2
import queue
import sys
import acoular
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time

# ---------------------------------------------------------------------------
# Hardware configuration
# ---------------------------------------------------------------------------
DEVICE_INDEX = 55       # PortAudio device index for the mic array
FS           = 48000    # Sample rate (Hz)
FRAME_SIZE   = 4096     # Analysis window length (samples)
HOP_SIZE     = 2048     # Samples consumed per callback (50 % overlap)
NFFT         = FRAME_SIZE

# ---------------------------------------------------------------------------
# Frequency band configuration — edit these to change the analysis range.
# At runtime these are treated as mutable state; the constants below define
# the DEFAULT values only.  Use keyboard keys 1/2/3 to switch bands live.
# ---------------------------------------------------------------------------
FREQ_MIN   = 500    # Lower bound of the analysis band (Hz)
FREQ_MAX   = 3000   # Upper bound of the analysis band (Hz)
BAND_WIDTH = 500    # Width of each sub-band used for MUSIC (Hz)

# Predefined frequency presets for interactive switching (key -> (min, max, label))
FREQ_PRESETS = {
    ord('1'): (300,  1000, 'Low (300-1000 Hz)'),
    ord('2'): (500,  3000, 'Speech (500-3000 Hz)'),
    ord('3'): (2000, 4000, 'High (2000-4000 Hz)'),
}

# Derive sub-band centre frequencies automatically from the range above
FREQ_CENTERS = np.arange(FREQ_MIN, FREQ_MAX + 1, BAND_WIDTH)

# ---------------------------------------------------------------------------
# Signal-processing settings
# ---------------------------------------------------------------------------
NOISE_THRESHOLD_DB = 15   # Minimum dB level before MUSIC is run
CSM_ALPHA          = 0.3  # EMA weight for cross-spectral matrix smoothing
                           # (higher = faster update, more noise)
ATTENUATION_POWER  = 3.0  # Sigmoid sharpness for heatmap contrast

# ---------------------------------------------------------------------------
# Camera / display settings
# ---------------------------------------------------------------------------
CAMERA_INDEX  = 0
CAMERA_FOV_H  = 50                         # Horizontal field-of-view (deg)
CAMERA_FOV_V  = CAMERA_FOV_H / (4 / 3)    # Vertical FOV assuming 4:3 aspect
DISPLAY_SCALE = 1.0                        # Resize output window (1.0 = full)
OVERLAY_ALPHA = 0.5                        # Heatmap transparency over camera

# ---------------------------------------------------------------------------
# Microphone array geometry
# ---------------------------------------------------------------------------
# Raw positions as recorded (x in col-0, z in col-1, y-unused in col-2).
# Rows are then reordered to match acoular's (x, y, z) convention.
mic_raw = np.array([
    [ 0.021, -0.063, 0.0], [ 0.063, -0.063, 0.0],
    [ 0.021, -0.021, 0.0], [ 0.063, -0.021, 0.0],
    [ 0.021,  0.021, 0.0], [ 0.063,  0.021, 0.0],
    [ 0.021,  0.063, 0.0], [ 0.063,  0.063, 0.0],
    [-0.063,  0.063, 0.0], [-0.021,  0.063, 0.0],
    [-0.063,  0.021, 0.0], [-0.021,  0.021, 0.0],
    [-0.063, -0.021, 0.0], [-0.021, -0.021, 0.0],
    [-0.063, -0.063, 0.0], [-0.021, -0.063, 0.0],
]).T  # shape: (3, 16)

# Remap axes: raw col-0 -> x, raw col-2 -> y (elevation), raw col-1 -> z
mic_pos       = np.zeros_like(mic_raw)
mic_pos[0, :] = mic_raw[0, :]   # x
mic_pos[1, :] = mic_raw[2, :]   # y  (was z in raw layout)
mic_pos[2, :] = mic_raw[1, :]   # z  (was y in raw layout)

mg = acoular.MicGeom()
mg.pos_total = mic_pos
N_CHANNELS = mic_pos.shape[1]   # 16

# ---------------------------------------------------------------------------
# Scanning grid — azimuth/elevation covering the camera FOV
# ---------------------------------------------------------------------------
az_center = 90   # Broadside direction (deg)
el_center = 0

az_min_grid = az_center - CAMERA_FOV_H / 2
az_max_grid = az_center + CAMERA_FOV_H / 2
el_min_grid = el_center - CAMERA_FOV_V / 2
el_max_grid = el_center + CAMERA_FOV_V / 2

step_deg = 0.5   # Angular resolution of the grid (deg)
az_vals  = np.arange(az_min_grid, az_max_grid + step_deg, step_deg)
el_vals  = np.arange(el_min_grid, el_max_grid + step_deg, step_deg)
az_grid_plot, el_grid_plot = np.meshgrid(az_vals, el_vals)

# Convert (az, el) angles to unit-sphere Cartesian for acoular
az_rad = np.deg2rad(az_grid_plot.flatten())
el_rad = np.deg2rad(el_grid_plot.flatten())
x_sph  = np.cos(el_rad) * np.cos(az_rad)
y_sph  = np.cos(el_rad) * np.sin(az_rad)
z_sph  = np.sin(el_rad)

grid = acoular.ImportGrid(pos=np.array([x_sph, y_sph, z_sph]))

# Build acoular steering vector (true-level normalisation).
# The SteeringVector object is reused across frequency changes because
# the spatial grid and mic geometry never change — only the frequencies do.
st = acoular.SteeringVector(grid=grid, mics=mg, steer_type='true level')


def build_steering_matrices(freq_centers):
    """
    Pre-compute a steering matrix for every sub-band centre frequency.

    This is called once at startup and again whenever the user switches
    frequency presets.  Pre-computing outside the real-time loop avoids
    the expensive st.transfer() call on every frame.

    Returns a dict mapping centre frequency (Hz) -> steering matrix ndarray.
    """
    return {freq: st.transfer(freq).T for freq in freq_centers}


def build_csm_history(freq_centers):
    """
    Initialise (or reset) the cross-spectral matrix history dictionary.

    Each entry starts as None and is populated on the first audio frame.
    Resetting is required when changing frequency bands because the old
    smoothed CSMs belong to different frequencies and would corrupt the
    new band's eigen-decomposition if carried over.

    Returns a dict mapping centre frequency (Hz) -> None.
    """
    return {f: None for f in freq_centers}


def estimate_distance(spectrum_flat, current_db):
    """
    Approximate source distance from signal strength and beam sharpness.

    This is NOT a physically exact measurement.  It combines two heuristics:
      - Signal attenuation: louder signals (higher dB) suggest a closer source.
      - Spatial concentration: a sharp MUSIC peak (high peak-to-mean ratio)
        indicates a well-defined source, also consistent with closer proximity.

    Both cues are normalised to [0, 1] and blended, then inverted so that
    a stronger / sharper signal maps to a shorter estimated distance.
    Result is clamped to a practical range of 0.5 – 5.0 metres.

    Returns estimated distance in metres (float).
    """
    peak_value = np.max(spectrum_flat)
    mean_value = np.mean(spectrum_flat)

    # Beam sharpness: ratio of peak to mean — high when source is localised
    sharpness = peak_value / (mean_value + 1e-6)

    # Normalise dB to [0, 1] over a 30–70 dB operating range
    db_norm = np.clip((current_db - 30) / 40, 0, 1)

    # Normalise sharpness — empirically, values above 10 indicate clear sources
    sharp_norm = np.clip(sharpness / 10, 0, 1)

    # Weighted combination: stronger weight on dB (60%), beam sharpness (40%)
    combined = 0.6 * db_norm + 0.4 * sharp_norm

    # Invert: high combined score → small distance
    distance_est = 1.0 / (combined + 1e-6)

    return float(np.clip(distance_est, 0.5, 5.0))


# ---------------------------------------------------------------------------
# Initial steering matrices and CSM history (built from default band)
# ---------------------------------------------------------------------------
steering_matrices = build_steering_matrices(FREQ_CENTERS)
csm_history       = build_csm_history(FREQ_CENTERS)

# Human-readable label for the active preset (shown in HUD)
active_preset_label = 'Speech (500-3000 Hz)'

# ---------------------------------------------------------------------------
# Audio pipeline setup
# ---------------------------------------------------------------------------
audio_queue  = queue.Queue()                                          # Inter-thread buffer
audio_buffer = np.zeros((FRAME_SIZE, N_CHANNELS), dtype=np.float32)  # Sliding window
window       = np.hanning(FRAME_SIZE)[:, np.newaxis]                  # Analysis window

# ---------------------------------------------------------------------------
# Camera initialisation
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    sys.exit("ERROR: Cannot open camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

ret, frame = cap.read()
if not ret:
    sys.exit("ERROR: Cannot read from camera.")

CAMERA_HEIGHT, CAMERA_WIDTH = frame.shape[:2]

# ---------------------------------------------------------------------------
# Matplotlib heatmap canvas (rendered off-screen via Agg backend)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(CAMERA_WIDTH / 100, CAMERA_HEIGHT / 100), dpi=100
)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(az_min_grid, az_max_grid)
ax.set_ylim(el_min_grid, el_max_grid)
ax.axis('off')

heatmap_data = np.zeros_like(az_grid_plot, dtype=float)
img = ax.imshow(
    heatmap_data,
    origin='lower', aspect='auto',
    extent=[az_min_grid, az_max_grid, el_min_grid, el_max_grid],
    cmap='inferno', vmin=0, vmax=1
)
canvas = FigureCanvasAgg(fig)

# ---------------------------------------------------------------------------
# State variables
# ---------------------------------------------------------------------------
current_spectrum = None
current_db       = 0
peak_az          = az_center
peak_el          = 0
distance_est     = None   # Estimated source distance in metres (None = no source)
fps              = 0.0
last_time        = time.time()
frame_counter    = 0


# ---------------------------------------------------------------------------
# Audio callback — called by sounddevice on a background thread
# ---------------------------------------------------------------------------
def callback(indata, frames, time_info, status):
    """Push incoming audio blocks into the thread-safe queue."""
    audio_queue.put(indata.copy())


# ---------------------------------------------------------------------------
# MUSIC algorithm
# ---------------------------------------------------------------------------
def music(signal_buffer):
    """
    Apply the MUSIC (MUltiple SIgnal Classification) algorithm to localise
    sound sources across all configured frequency sub-bands.

    Steps per sub-band:
      1. STFT  — window and FFT the buffered signal.
      2. CSM   — build the cross-spectral matrix from FFT bins in the band,
                 then smooth with an EMA to reduce frame-to-frame noise.
      3. Eigen — decompose the CSM; the minor eigenvectors span the noise
                 subspace orthogonal to the signal subspace.
      4. MUSIC — project each steering vector onto the noise subspace;
                 peaks (low projection = high reciprocal) indicate source
                 directions.

    Returns a normalised 1-D power spectrum over all grid points.
    """
    global csm_history

    # --- Step 1: Short-Time Fourier Transform --------------------------------
    windowed = signal_buffer * window                     # Apply Hann window
    fft_out  = np.fft.rfft(windowed, n=NFFT, axis=0)     # (freq_bins, channels)
    freqs    = np.fft.rfftfreq(NFFT, 1 / FS)

    total_spectrum = np.zeros(steering_matrices[FREQ_CENTERS[0]].shape[1])

    for center in FREQ_CENTERS:
        # Identify FFT bins that fall within this sub-band
        f_min    = center - (BAND_WIDTH / 2)
        f_max    = center + (BAND_WIDTH / 2)
        idx_low  = np.argmin(np.abs(freqs - f_min))
        idx_high = np.argmin(np.abs(freqs - f_max))
        if idx_low == idx_high:
            idx_high += 1          # Guarantee at least one bin

        X_band = fft_out[idx_low:idx_high, :]  # (bins_in_band, channels)
        if X_band.shape[0] == 0:
            continue

        # --- Step 2: Cross-Spectral Matrix (CSM) -----------------------------
        # Sum outer products across bins, then normalise by bin count
        csm_instant  = np.einsum('fi,fj->ij', X_band, X_band.conj())
        csm_instant /= X_band.shape[0]

        # Exponential moving average smoothing for temporal stability
        if csm_history[center] is None:
            csm_history[center] = csm_instant
        else:
            csm_history[center] = (
                (1 - CSM_ALPHA) * csm_history[center] + CSM_ALPHA * csm_instant
            )

        # Regularise to avoid singular matrix during eigen-decomposition
        R = csm_history[center] + 1e-10 * np.eye(N_CHANNELS)

        # --- Step 3: Eigen-decomposition — extract noise subspace -----------
        # eigh returns eigenvalues in ascending order; the last eigenvector
        # corresponds to the dominant signal.  All others form the noise subspace.
        eigvals, eigvecs = np.linalg.eigh(R)
        noise_subspace   = eigvecs[:, :-1]   # (channels, channels-1)

        # --- Step 4: MUSIC pseudo-spectrum -----------------------------------
        # For each steering vector a(θ), compute how much it lies in the noise
        # subspace.  A true source direction is orthogonal to the noise subspace,
        # giving a near-zero projection and hence a large reciprocal.
        A             = steering_matrices[center]          # (channels, grid_pts)
        projection    = noise_subspace.conj().T @ A        # (channels-1, grid_pts)
        spectrum_band = 1.0 / (
            np.sum(np.abs(projection) ** 2, axis=0) + 1e-12
        )

        # Min-max normalise so all bands contribute equally
        s_min, s_max = np.min(spectrum_band), np.max(spectrum_band)
        if s_max > s_min:
            spectrum_band = (spectrum_band - s_min) / (s_max - s_min)

        total_spectrum += spectrum_band

    return total_spectrum


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
stream = sd.InputStream(
    device=DEVICE_INDEX, channels=N_CHANNELS, samplerate=FS,
    blocksize=HOP_SIZE, callback=callback
)

with stream:
    while True:
        # --- FPS measurement -------------------------------------------------
        frame_counter += 1
        now     = time.time()
        elapsed = now - last_time
        if elapsed >= 1.0:
            fps           = frame_counter / elapsed
            frame_counter = 0
            last_time     = now

        # --- Drain audio queue into the sliding buffer -----------------------
        # New audio blocks are appended at the end; old samples are discarded
        # from the front, maintaining a FRAME_SIZE-length rolling window.
        try:
            while not audio_queue.empty():
                indata = audio_queue.get_nowait()
                audio_buffer[:-HOP_SIZE] = audio_buffer[HOP_SIZE:]
                audio_buffer[-HOP_SIZE:] = indata
        except queue.Empty:
            pass

        # --- Compute RMS level (dB) ------------------------------------------
        rms        = np.sqrt(np.mean(audio_buffer ** 2))
        current_db = 20 * np.log10(rms + 1e-9) + 90   # Offset to positive range

        # --- Grab camera frame -----------------------------------------------
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)   # Mirror so left/right matches natural view

        # --- Run MUSIC only when signal is above the noise floor -------------
        if current_db >= NOISE_THRESHOLD_DB:
            try:
                spectrum_flat = music(audio_buffer)

                # Normalise to [0, 1]
                spectrum_flat = (
                    (spectrum_flat - np.min(spectrum_flat))
                    / (np.max(spectrum_flat) - np.min(spectrum_flat) + 1e-9)
                )

                # Sigmoid contrast enhancement — sharpens the peak display
                spectrum_flat = 1 / (
                    1 + np.exp(-ATTENUATION_POWER * (spectrum_flat - 0.5))
                )
                spectrum_flat = (
                    (spectrum_flat - np.min(spectrum_flat))
                    / (np.max(spectrum_flat) - np.min(spectrum_flat) + 1e-9)
                )

                spectrum_2d      = spectrum_flat.reshape(az_grid_plot.shape)
                current_spectrum = spectrum_2d

                # Locate the peak direction on the grid
                peak_idx = np.argmax(spectrum_flat)
                peak_az  = az_grid_plot.flatten()[peak_idx]
                peak_el  = el_grid_plot.flatten()[peak_idx]

                # --- Approximate distance estimation -------------------------
                # Uses signal strength (dB) and MUSIC beam sharpness as proxies
                # for distance.  Not a physical measurement — see estimate_distance().
                distance_est = estimate_distance(spectrum_flat, current_db)

            except Exception:
                current_spectrum = None
                distance_est     = None
        else:
            # Below noise floor — clear spectrum, CSM history, and distance
            current_spectrum = None
            distance_est     = None
            for f in FREQ_CENTERS:
                csm_history[f] = None

        # --- Render heatmap overlay ------------------------------------------
        if current_spectrum is not None:
            img.set_data(current_spectrum)
            canvas.draw()

            # Convert Matplotlib RGBA buffer -> OpenCV BGR
            buf         = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            buf         = buf.reshape(canvas.get_width_height()[::-1] + (4,))
            heatmap_img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
            heatmap_img = cv2.resize(heatmap_img, (CAMERA_WIDTH, CAMERA_HEIGHT))
            overlay     = cv2.addWeighted(
                frame, 1 - OVERLAY_ALPHA, heatmap_img, OVERLAY_ALPHA, 0
            )
        else:
            overlay = frame

        # --- Draw peak-direction crosshair -----------------------------------
        if current_db > NOISE_THRESHOLD_DB:
            x_norm = (peak_az - az_min_grid) / (az_max_grid - az_min_grid)
            y_norm = (peak_el - el_min_grid) / (el_max_grid - el_min_grid)
            cx     = int(x_norm * CAMERA_WIDTH)
            cy     = int((1 - y_norm) * CAMERA_HEIGHT)   # Invert y for image coords
            cv2.drawMarker(overlay, (cx, cy), (0, 0, 255),
                           cv2.MARKER_CROSS, 40, 3)

        # --- HUD text overlays -----------------------------------------------
        # SPL readout
        cv2.putText(overlay, f'{current_db:.2f} dB',
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # Frames per second
        cv2.putText(overlay, f'{fps:.1f} FPS',
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Approximate distance — labelled "~" to make clear it is an estimate
        if distance_est is not None:
            cv2.putText(overlay, f'~{distance_est:.2f} m (est.)',
                        (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 165, 0), 2, cv2.LINE_AA)

        # Active frequency range
        freq_label = f'Freq: {FREQ_MIN}-{FREQ_MAX} Hz  |  {active_preset_label}'
        cv2.putText(overlay, freq_label,
                    (30, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 220, 0), 2, cv2.LINE_AA)

        # Key-hint reminder at bottom-left
        cv2.putText(overlay, '1:Low  2:Speech  3:High  q:Quit',
                    (30, CAMERA_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # --- Optional display downscale --------------------------------------
        if DISPLAY_SCALE != 1.0:
            display_w = int(CAMERA_WIDTH  * DISPLAY_SCALE)
            display_h = int(CAMERA_HEIGHT * DISPLAY_SCALE)
            overlay   = cv2.resize(overlay, (display_w, display_h))

        cv2.imshow('Acoustic Camera', overlay)

        # --- Keyboard handling -----------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key in FREQ_PRESETS:
            # Switch frequency band — must rebuild steering matrices (wavelength-
            # dependent) and reset CSM history (old band data corrupts new band).
            new_min, new_max, new_label = FREQ_PRESETS[key]

            FREQ_MIN            = new_min
            FREQ_MAX            = new_max
            FREQ_CENTERS        = np.arange(FREQ_MIN, FREQ_MAX + 1, BAND_WIDTH)
            active_preset_label = new_label

            steering_matrices = build_steering_matrices(FREQ_CENTERS)
            csm_history       = build_csm_history(FREQ_CENTERS)
            current_spectrum  = None
            distance_est      = None

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
plt.close(fig)