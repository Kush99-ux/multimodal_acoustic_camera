"""
test_localization_position_sweep.py
====================================

Controlled position-sweep experiment for the miniDSP UMA-16 v2.

Purpose
-------
Determine whether MUSIC:

1. Tracks known physical source movements correctly.
2. Has a constant coordinate offset.
3. Has a nonlinear / direction-dependent localization error.

Experimental source
-------------------
A continuous 1 kHz tone played from a phone speaker.

The speaker is positioned approximately 70 mm above the microphone
array.

Five known source positions are tested:

        Y = +30 mm
             |
             |
X = -30 ---- 0 ---- +30 mm
             |
             |
        Y = -30 mm

All measurements use:

    Z = 70 mm
    Frequency = 1 kHz
    Grid resolution = 5 mm

This test does NOT use:

    - ESP32-CAM
    - YOLO
    - heatmap visualization
    - audio/video fusion

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

import acoustic.steering as steering_module

from acoustic.acquisition import AudioAcquisition
from acoustic.music import MusicLocalizer
from config.settings import (
    SAMPLE_RATE,
    FRAME_SIZE,
    HOP_SIZE,
)


# =====================================================================
# EXPERIMENT CONFIGURATION
# =====================================================================

TARGET_FREQUENCY = 1000.0

SOURCE_Z = 0.070

GRID_X_MIN = -0.20
GRID_X_MAX = +0.20

GRID_Y_MIN = -0.20
GRID_Y_MAX = +0.20

GRID_INCREMENT = 0.005


# Number of independent source measurements at each position.
MEASUREMENTS_PER_POSITION = 5

# Number of consecutive MUSIC frames used to form one measurement.
FRAMES_PER_MEASUREMENT = 5

WARMUP_SECONDS = 2.0

FRAME_INTERVAL = 0.05


# =====================================================================
# KNOWN PHYSICAL SOURCE POSITIONS
# =====================================================================
#
# Coordinates are in metres.
#
# IMPORTANT:
#
# X:
#   negative = left
#   positive = right
#
# Y:
#   negative = bottom/front side of the board in the current geometry
#   positive = top/back side of the board in the current geometry
#
# Z:
#   positive = above the microphone plane
#
# =====================================================================

POSITIONS: Dict[str, Tuple[float, float, float]] = {
    "X_NEG": (-0.030, 0.000, SOURCE_Z),
    "CENTER": (0.000, 0.000, SOURCE_Z),
    "X_POS": (+0.030, 0.000, SOURCE_Z),
    "Y_NEG": (0.000, -0.030, SOURCE_Z),
    "Y_POS": (0.000, +0.030, SOURCE_Z),
}


# =====================================================================
# CONFIGURATION OVERRIDE
# =====================================================================

def configure_experiment() -> None:
    """
    Override the normal project localization settings for this
    controlled near-field experiment.

    config/settings.py is NOT modified.
    """

    steering_module.GRID_X_MIN = GRID_X_MIN
    steering_module.GRID_X_MAX = GRID_X_MAX

    steering_module.GRID_Y_MIN = GRID_Y_MIN
    steering_module.GRID_Y_MAX = GRID_Y_MAX

    steering_module.GRID_Z = SOURCE_Z

    steering_module.GRID_INCREMENT = GRID_INCREMENT

    steering_module.FREQ_MIN = TARGET_FREQUENCY
    steering_module.FREQ_MAX = TARGET_FREQUENCY
    steering_module.FREQ_STEP = 100.0


# =====================================================================
# GRID AXES
# =====================================================================

def get_grid_axes() -> Tuple[np.ndarray, np.ndarray]:

    x_values = np.arange(
        GRID_X_MIN,
        GRID_X_MAX + GRID_INCREMENT * 0.5,
        GRID_INCREMENT,
        dtype=np.float64,
    )

    y_values = np.arange(
        GRID_Y_MIN,
        GRID_Y_MAX + GRID_INCREMENT * 0.5,
        GRID_INCREMENT,
        dtype=np.float64,
    )

    return x_values, y_values


# =====================================================================
# POSITION ESTIMATION
# =====================================================================

def estimate_position(
    heatmap: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> Tuple[float, float, float, int, int]:

    peak_index = int(np.argmax(heatmap))

    row, column = np.unravel_index(
        peak_index,
        heatmap.shape,
    )

    estimated_x = float(x_values[column])
    estimated_y = float(y_values[row])

    peak = float(heatmap[row, column])

    return (
        estimated_x,
        estimated_y,
        peak,
        int(row),
        int(column),
    )


# =====================================================================
# ERROR CALCULATION
# =====================================================================

def calculate_error(
    estimated_x: float,
    estimated_y: float,
    actual_x: float,
    actual_y: float,
) -> float:

    dx = estimated_x - actual_x
    dy = estimated_y - actual_y

    return float(np.sqrt(dx * dx + dy * dy))


# =====================================================================
# SINGLE POSITION EXPERIMENT
# =====================================================================

def measure_position(
    localizer: MusicLocalizer,
    audio: AudioAcquisition,
    x_values: np.ndarray,
    y_values: np.ndarray,
    actual_x: float,
    actual_y: float,
) -> Tuple[float, float, float, float, int, int]:
    """
    Perform one independent measurement at the current physical
    source position.

    Several consecutive MUSIC heatmaps are averaged before finding
    the final peak.

    Returns
    -------
    estimated_x
    estimated_y
    peak
    error
    row
    column
    """

    # Start each measurement without historical CSM information.
    localizer.previous_csm = None

    heatmaps: List[np.ndarray] = []

    for _ in range(FRAMES_PER_MEASUREMENT):

        frame = audio.read()

        if frame.ndim != 2:
            raise RuntimeError(
                f"Unexpected audio frame dimensions: {frame.shape}"
            )

        if frame.shape[0] != FRAME_SIZE:
            raise RuntimeError(
                f"Unexpected frame size: {frame.shape}"
            )

        if frame.shape[1] != 16:
            raise RuntimeError(
                f"Unexpected channel count: {frame.shape}"
            )

        heatmap = localizer.localize(frame)

        heatmaps.append(heatmap)

        time.sleep(FRAME_INTERVAL)

    # Average the consecutive MUSIC results.
    averaged_heatmap = np.mean(
        np.stack(heatmaps, axis=0),
        axis=0,
    )

    # Normalize only after temporal averaging.
    averaged_heatmap -= averaged_heatmap.min()

    maximum = averaged_heatmap.max()

    if maximum > 0:
        averaged_heatmap /= maximum

    (
        estimated_x,
        estimated_y,
        peak,
        row,
        column,
    ) = estimate_position(
        averaged_heatmap,
        x_values,
        y_values,
    )

    error = calculate_error(
        estimated_x,
        estimated_y,
        actual_x,
        actual_y,
    )

    return (
        estimated_x,
        estimated_y,
        peak,
        error,
        row,
        column,
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print()
    print("=" * 76)
    print("UMA-16 MUSIC LOCALIZATION POSITION SWEEP")
    print("=" * 76)

    print()
    print("Purpose:")
    print("  Determine whether MUSIC follows controlled physical")
    print("  movement of a 1 kHz acoustic source.")

    print()
    print("This test does NOT use:")
    print("  - ESP32-CAM")
    print("  - YOLO")
    print("  - Heatmap visualization")
    print("  - Audio/video fusion")

    print()
    print("=" * 76)

    # ---------------------------------------------------------------
    # Configure controlled experiment.
    # ---------------------------------------------------------------

    configure_experiment()

    x_values, y_values = get_grid_axes()

    # ---------------------------------------------------------------
    # Experiment information.
    # ---------------------------------------------------------------

    print()
    print("[Source]")
    print(f"  Frequency       : {TARGET_FREQUENCY:.1f} Hz")
    print(f"  Height Z        : {SOURCE_Z * 1000:.0f} mm")

    print()
    print("[Search grid]")
    print(
        f"  X range         : "
        f"{GRID_X_MIN:+.3f} → {GRID_X_MAX:+.3f} m"
    )

    print(
        f"  Y range         : "
        f"{GRID_Y_MIN:+.3f} → {GRID_Y_MAX:+.3f} m"
    )

    print(
        f"  Z plane         : "
        f"{SOURCE_Z:+.3f} m"
    )

    print(
        f"  Resolution      : "
        f"{GRID_INCREMENT * 1000:.0f} mm"
    )

    print()
    print("[Audio]")
    print(f"  Sample rate     : {SAMPLE_RATE} Hz")
    print(f"  Frame size      : {FRAME_SIZE}")
    print(f"  Hop size        : {HOP_SIZE}")

    print()
    print("[Measurements]")
    print(
        f"  Positions       : {len(POSITIONS)}"
    )

    print(
        f"  Measurements / position : "
        f"{MEASUREMENTS_PER_POSITION}"
    )

    print(
        f"  MUSIC frames / measurement : "
        f"{FRAMES_PER_MEASUREMENT}"
    )

    # ---------------------------------------------------------------
    # Initialize MUSIC.
    # ---------------------------------------------------------------

    print()
    print("[System] Initializing MUSIC localizer...")

    localizer = MusicLocalizer(
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        smoothing=0.3,
    )

    print(
        f"[System] Grid shape    : "
        f"{localizer.grid_shape}"
    )

    print(
        f"[System] Frequencies   : "
        f"{len(localizer.frequencies)}"
    )

    print(
        f"[System] Frame size    : "
        f"{localizer.frame_size}"
    )

    print(
        f"[System] Sample rate   : "
        f"{localizer.sample_rate} Hz"
    )

    expected_shape = (
        len(y_values),
        len(x_values),
    )

    if localizer.grid_shape != expected_shape:
        raise RuntimeError(
            "Unexpected MUSIC grid shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {localizer.grid_shape}"
        )

    # ---------------------------------------------------------------
    # Audio.
    # ---------------------------------------------------------------

    audio = AudioAcquisition()

    # Dictionary storing all estimates.
    all_results: Dict[
        str,
        List[Tuple[float, float, float]]
    ] = {}

    try:

        audio.start()

        print()
        print(
            f"[Audio] Warming up for "
            f"{WARMUP_SECONDS:.1f} seconds..."
        )

        start_time = time.perf_counter()

        while (
            time.perf_counter() - start_time
            < WARMUP_SECONDS
        ):
            audio.read()
            time.sleep(0.05)

        # -----------------------------------------------------------
        # Explain coordinate system.
        # -----------------------------------------------------------

        print()
        print("=" * 76)
        print("COORDINATE SYSTEM")
        print("=" * 76)

        print()
        print("Looking at the microphone board as in your photographs:")

        print()
        print("                 Y = +")
        print("                   ↑")
        print("                   |")
        print("                   |")
        print("          X = - ← CENTER → X = +")
        print("                   |")
        print("                   |")
        print("                   ↓")
        print("                 Y = -")

        print()
        print(
            "Positive X = toward the right side of the board."
        )

        print(
            "Positive Y = toward the top side of the board."
        )

        print(
            "Z = +70 mm = above the microphone plane."
        )

        # -----------------------------------------------------------
        # Position loop.
        # -----------------------------------------------------------

        for position_name, (
            actual_x,
            actual_y,
            actual_z,
        ) in POSITIONS.items():

            print()
            print()
            print("=" * 76)
            print(
                f"NEXT POSITION: {position_name}"
            )
            print("=" * 76)

            print()
            print("Place the PHONE SPEAKER at:")

            print(
                f"  X = {actual_x * 1000:+.0f} mm"
            )

            print(
                f"  Y = {actual_y * 1000:+.0f} mm"
            )

            print(
                f"  Z = {actual_z * 1000:+.0f} mm"
            )

            print()
            print("Important:")
            print("  1. Move the actual PHONE SPEAKER opening,")
            print("     not the center of the phone.")
            print("  2. Keep the phone parallel to the microphone board.")
            print("  3. Keep the speaker height at approximately 70 mm.")
            print("  4. Keep the 1 kHz tone continuously playing.")
            print("  5. Do not move the UMA-16.")
            print("  6. Keep the phone volume unchanged.")

            input(
                "\nPress ENTER when the source is positioned correctly..."
            )

            print()
            print(
                f"[Measurement] {position_name}"
            )

            position_results = []

            for measurement_index in range(
                1,
                MEASUREMENTS_PER_POSITION + 1,
            ):

                (
                    estimated_x,
                    estimated_y,
                    peak,
                    error,
                    row,
                    column,
                ) = measure_position(
                    localizer,
                    audio,
                    x_values,
                    y_values,
                    actual_x,
                    actual_y,
                )

                position_results.append(
                    (
                        estimated_x,
                        estimated_y,
                        error,
                    )
                )

                print(
                    f"  "
                    f"{measurement_index:02d}/"
                    f"{MEASUREMENTS_PER_POSITION:02d}"
                    f"  → "
                    f"X={estimated_x * 1000:+.0f} mm, "
                    f"Y={estimated_y * 1000:+.0f} mm"
                    f"  "
                    f"Error={error * 1000:.1f} mm"
                    f"  "
                    f"Grid=({row},{column})"
                )

            all_results[position_name] = position_results

    finally:

        audio.stop()

    # =================================================================
    # POSITION SUMMARY
    # =================================================================

    print()
    print()
    print("=" * 76)
    print("POSITION SWEEP RESULTS")
    print("=" * 76)

    print()

    print(
        "Position     Actual X     Actual Y     "
        "Mean X       Mean Y       Error"
    )

    print(
        "             (mm)         (mm)         "
        "(mm)         (mm)         (mm)"
    )

    print("-" * 76)

    summary = {}

    for position_name, results in all_results.items():

        data = np.asarray(
            results,
            dtype=np.float64,
        )

        mean_x = float(np.mean(data[:, 0]))
        mean_y = float(np.mean(data[:, 1]))
        mean_error = float(
            np.mean(data[:, 2])
        )

        actual_x = POSITIONS[position_name][0]
        actual_y = POSITIONS[position_name][1]

        summary[position_name] = (
            actual_x,
            actual_y,
            mean_x,
            mean_y,
            mean_error,
        )

        print(
            f"{position_name:<10} "
            f"{actual_x * 1000:+8.0f}     "
            f"{actual_y * 1000:+8.0f}     "
            f"{mean_x * 1000:+8.0f}     "
            f"{mean_y * 1000:+8.0f}     "
            f"{mean_error * 1000:8.1f}"
        )

    # =================================================================
    # X-AXIS ANALYSIS
    # =================================================================

    print()
    print("=" * 76)
    print("X-AXIS TRACKING ANALYSIS")
    print("=" * 76)

    x_actual = []
    x_estimated = []

    for name in ("X_NEG", "CENTER", "X_POS"):

        actual_x, _, mean_x, _, _ = summary[name]

        x_actual.append(actual_x)
        x_estimated.append(mean_x)

        print(
            f"{name:<8} : "
            f"Actual X = {actual_x * 1000:+.1f} mm   "
            f"MUSIC X = {mean_x * 1000:+.1f} mm   "
            f"Bias = {(mean_x - actual_x) * 1000:+.1f} mm"
        )

    x_actual_array = np.asarray(
        x_actual,
        dtype=np.float64,
    )

    x_estimated_array = np.asarray(
        x_estimated,
        dtype=np.float64,
    )

    # Linear fit:
    #
    # estimated = slope * actual + intercept
    #
    # A slope near 1 means MUSIC tracks movement correctly.
    # An intercept indicates a constant coordinate bias.

    if len(x_actual_array) >= 2:

        slope_x, intercept_x = np.polyfit(
            x_actual_array,
            x_estimated_array,
            1,
        )

        print()
        print(
            "X-axis linear relationship:"
        )

        print(
            f"  MUSIC_X ≈ "
            f"{slope_x:.3f} * Actual_X "
            f"{intercept_x * 1000:+.1f} mm"
        )

        print()
        print(
            f"  Slope     : {slope_x:.3f}"
        )

        print(
            f"  Intercept : "
            f"{intercept_x * 1000:+.1f} mm"
        )

    # =================================================================
    # Y-AXIS ANALYSIS
    # =================================================================

    print()
    print("=" * 76)
    print("Y-AXIS TRACKING ANALYSIS")
    print("=" * 76)

    y_actual = []
    y_estimated = []

    for name in ("Y_NEG", "CENTER", "Y_POS"):

        _, actual_y, _, mean_y, _ = summary[name]

        y_actual.append(actual_y)
        y_estimated.append(mean_y)

        print(
            f"{name:<8} : "
            f"Actual Y = {actual_y * 1000:+.1f} mm   "
            f"MUSIC Y = {mean_y * 1000:+.1f} mm   "
            f"Bias = {(mean_y - actual_y) * 1000:+.1f} mm"
        )

    y_actual_array = np.asarray(
        y_actual,
        dtype=np.float64,
    )

    y_estimated_array = np.asarray(
        y_estimated,
        dtype=np.float64,
    )

    if len(y_actual_array) >= 2:

        slope_y, intercept_y = np.polyfit(
            y_actual_array,
            y_estimated_array,
            1,
        )

        print()
        print(
            "Y-axis linear relationship:"
        )

        print(
            f"  MUSIC_Y ≈ "
            f"{slope_y:.3f} * Actual_Y "
            f"{intercept_y * 1000:+.1f} mm"
        )

        print()
        print(
            f"  Slope     : {slope_y:.3f}"
        )

        print(
            f"  Intercept : "
            f"{intercept_y * 1000:+.1f} mm"
        )

    # =================================================================
    # INTERPRETATION
    # =================================================================

    print()
    print("=" * 76)
    print("INTERPRETATION GUIDE")
    print("=" * 76)

    print()
    print(
        "If X slope is approximately 1.0 and the intercept is"
    )
    print(
        "approximately constant:"
    )

    print(
        "  → MUSIC is tracking X correctly."
    )

    print(
        "  → A fixed X coordinate offset is likely."
    )

    print()
    print(
        "If X slope is far from 1.0:"
    )

    print(
        "  → MUSIC is not correctly translating physical"
    )

    print(
        "    source movement into estimated movement."
    )

    print()
    print(
        "If Y behaves similarly:"
    )

    print(
        "  → The issue may be a global coordinate / phase bias."
    )

    print()
    print(
        "If X works but Y does not:"
    )

    print(
        "  → We likely have an axis-specific geometry/orientation"
    )

    print(
        "    or phase-model problem."
    )

    print()
    print(
        "DO NOT MODIFY geometry.py, steering.py, or music.py"
    )

    print(
        "until these results have been analyzed."
    )

    print()
    print("=" * 76)
    print("POSITION SWEEP COMPLETE")
    print("=" * 76)
    print()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()