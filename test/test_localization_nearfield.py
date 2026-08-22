"""
test_localization_nearfield.py
================================

Controlled near-field MUSIC localization experiment for the
miniDSP UMA-16 v2 microphone array.

Experimental setup
------------------
A 1 kHz tone is played from a phone speaker positioned approximately
7 cm above the center of the UMA-16 microphone array.

Known source position:

    X = 0.00 m
    Y = 0.00 m
    Z = 0.07 m

The experiment searches for the source on a fixed near-field plane:

    Z = 0.07 m

Search region:

    X = -0.20 ... +0.20 m
    Y = -0.20 ... +0.20 m

Spatial resolution:

    5 mm

This test intentionally does NOT use:

    - ESP32-CAM
    - YOLO
    - heatmap visualization
    - audio/video fusion
    - camera calibration

Purpose
-------
This is a controlled numerical validation of the acoustic localization
pipeline after correcting the physical UMA-16 microphone geometry.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from acoustic.acquisition import AudioAcquisition
from acoustic.music import MusicLocalizer

# Import the steering module itself because this experiment intentionally
# overrides its grid/frequency configuration before SteeringFactory is created.
import acoustic.steering as steering_module

from config.settings import (
    SAMPLE_RATE,
    FRAME_SIZE,
    HOP_SIZE,
)


# =====================================================================
# CONTROLLED EXPERIMENT PARAMETERS
# =====================================================================

# ---------------------------------------------------------------------
# Known physical source position
# ---------------------------------------------------------------------

SOURCE_X = 0.00
SOURCE_Y = 0.00

# Measured approximately 7 cm above the microphone plane.
SOURCE_Z = 0.07


# ---------------------------------------------------------------------
# Near-field search grid
# ---------------------------------------------------------------------

SEARCH_X_MIN = -0.20
SEARCH_X_MAX = +0.20

SEARCH_Y_MIN = -0.20
SEARCH_Y_MAX = +0.20

SEARCH_Z = SOURCE_Z

GRID_INCREMENT = 0.005       # 5 mm


# ---------------------------------------------------------------------
# Acoustic source
# ---------------------------------------------------------------------

TARGET_FREQUENCY = 1000.0    # Hz


# ---------------------------------------------------------------------
# Measurement configuration
# ---------------------------------------------------------------------

NUM_MEASUREMENTS = 10

# Number of audio/MUSIC frames used to stabilize each measurement.
FRAMES_PER_MEASUREMENT = 5

# Initial stream warm-up.
WARMUP_SECONDS = 2.0

# Small delay between MUSIC frames.
FRAME_INTERVAL = 0.05


# =====================================================================
# GRID / FREQUENCY OVERRIDE
# =====================================================================

def configure_nearfield_experiment() -> None:
    """
    Override SteeringFactory's configuration for this controlled test.

    This does NOT modify config/settings.py.

    The normal project configuration remains suitable for the main
    application. These values are only applied to this experiment.
    """

    steering_module.GRID_X_MIN = SEARCH_X_MIN
    steering_module.GRID_X_MAX = SEARCH_X_MAX

    steering_module.GRID_Y_MIN = SEARCH_Y_MIN
    steering_module.GRID_Y_MAX = SEARCH_Y_MAX

    steering_module.GRID_Z = SEARCH_Z

    steering_module.GRID_INCREMENT = GRID_INCREMENT

    # The source is a controlled 1 kHz tone.
    #
    # Use only the 1 kHz steering frequency for this experiment.
    # The actual FFT bin will be approximately 996.09 Hz for a
    # 4096-point FFT at 48 kHz.
    steering_module.FREQ_MIN = TARGET_FREQUENCY
    steering_module.FREQ_MAX = TARGET_FREQUENCY
    steering_module.FREQ_STEP = 100.0


# =====================================================================
# GRID INFORMATION
# =====================================================================

def get_grid_axes() -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the X/Y coordinate arrays corresponding to the
    configured localization grid.
    """

    x_values = np.arange(
        SEARCH_X_MIN,
        SEARCH_X_MAX + GRID_INCREMENT * 0.5,
        GRID_INCREMENT,
        dtype=np.float64,
    )

    y_values = np.arange(
        SEARCH_Y_MIN,
        SEARCH_Y_MAX + GRID_INCREMENT * 0.5,
        GRID_INCREMENT,
        dtype=np.float64,
    )

    return x_values, y_values


# =====================================================================
# POSITION EXTRACTION
# =====================================================================

def estimate_position(
    heatmap: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> Tuple[float, float, float, int, int]:
    """
    Find the maximum MUSIC response and convert its grid index
    into physical X/Y coordinates.

    Returns
    -------
    x : float
        Estimated X coordinate.

    y : float
        Estimated Y coordinate.

    peak : float
        Normalized MUSIC peak.

    row : int
        Grid row.

    column : int
        Grid column.
    """

    peak_index = int(np.argmax(heatmap))

    row, column = np.unravel_index(
        peak_index,
        heatmap.shape,
    )

    x = float(x_values[column])
    y = float(y_values[row])

    peak = float(heatmap[row, column])

    return x, y, peak, row, column


# =====================================================================
# POSITION ERROR
# =====================================================================

def calculate_position_error(
    x: float,
    y: float,
) -> float:
    """
    Calculate horizontal Euclidean localization error relative
    to the known source position.
    """

    dx = x - SOURCE_X
    dy = y - SOURCE_Y

    return float(np.sqrt(dx * dx + dy * dy))


# =====================================================================
# MAIN EXPERIMENT
# =====================================================================

def main() -> None:

    print()
    print("=" * 72)
    print("UMA-16 NEAR-FIELD MUSIC LOCALIZATION EXPERIMENT")
    print("=" * 72)

    print()
    print("Purpose:")
    print("  Validate numerical MUSIC localization using a controlled")
    print("  1 kHz source positioned approximately 7 cm above the array.")

    print()
    print("This test does NOT use:")
    print("  - ESP32 camera")
    print("  - YOLO")
    print("  - Heatmap visualization")
    print("  - Audio/video fusion")

    print()
    print("=" * 72)

    # ---------------------------------------------------------------
    # Configure controlled experiment BEFORE creating localizer.
    # ---------------------------------------------------------------

    configure_nearfield_experiment()

    # ---------------------------------------------------------------
    # Display experimental geometry.
    # ---------------------------------------------------------------

    print()
    print("[Experiment]")
    print(f"  Known source X : {SOURCE_X:+.3f} m")
    print(f"  Known source Y : {SOURCE_Y:+.3f} m")
    print(f"  Known source Z : {SOURCE_Z:+.3f} m")

    print()
    print("[Search grid]")
    print(f"  X range        : {SEARCH_X_MIN:+.3f} → {SEARCH_X_MAX:+.3f} m")
    print(f"  Y range        : {SEARCH_Y_MIN:+.3f} → {SEARCH_Y_MAX:+.3f} m")
    print(f"  Z plane        : {SEARCH_Z:+.3f} m")
    print(f"  Increment      : {GRID_INCREMENT:.3f} m")

    print()
    print("[Source]")
    print(f"  Target tone    : {TARGET_FREQUENCY:.1f} Hz")

    print()
    print("[Audio]")
    print(f"  Sample rate    : {SAMPLE_RATE} Hz")
    print(f"  Frame size     : {FRAME_SIZE}")
    print(f"  Hop size       : {HOP_SIZE}")

    # ---------------------------------------------------------------
    # Build coordinate axes.
    # ---------------------------------------------------------------

    x_values, y_values = get_grid_axes()

    expected_grid_rows = len(y_values)
    expected_grid_columns = len(x_values)

    print()
    print("[Grid]")
    print(
        f"  Shape          : "
        f"({expected_grid_rows}, {expected_grid_columns})"
    )
    print(f"  X points       : {expected_grid_columns}")
    print(f"  Y points       : {expected_grid_rows}")

    # ---------------------------------------------------------------
    # Initialize MUSIC.
    # ---------------------------------------------------------------

    print()
    print("[System] Initializing near-field MUSIC localizer...")

    localizer = MusicLocalizer(
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        smoothing=0.3,
    )

    print()
    print(
        f"[System] Grid shape : "
        f"{localizer.grid_shape}"
    )

    print(
        f"[System] Frequencies : "
        f"{len(localizer.frequencies)}"
    )

    print(
        f"[System] Frame size : "
        f"{localizer.frame_size}"
    )

    print(
        f"[System] Sample rate : "
        f"{localizer.sample_rate} Hz"
    )

    # ---------------------------------------------------------------
    # Verify grid shape.
    # ---------------------------------------------------------------

    expected_shape = (
        expected_grid_rows,
        expected_grid_columns,
    )

    if localizer.grid_shape != expected_shape:
        raise RuntimeError(
            "Unexpected MUSIC grid shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {localizer.grid_shape}"
        )

    # ---------------------------------------------------------------
    # Initialize audio.
    # ---------------------------------------------------------------

    audio = AudioAcquisition()

    results: List[Tuple[float, float, float]] = []

    try:

        audio.start()

        # -----------------------------------------------------------
        # Warm-up
        # -----------------------------------------------------------

        print()
        print(
            f"[Audio] Warming up for "
            f"{WARMUP_SECONDS:.1f} seconds..."
        )

        warmup_start = time.perf_counter()

        while (
            time.perf_counter() - warmup_start
            < WARMUP_SECONDS
        ):
            audio.read()
            time.sleep(0.05)

        print()
        print("=" * 72)
        print("STARTING NEAR-FIELD LOCALIZATION")
        print("=" * 72)

        print()
        print("Keep the phone speaker:")
        print("  - stationary")
        print("  - approximately 7 cm above the array")
        print("  - approximately centered over the array")
        print()
        print(
            "Expected source position:"
            f" X={SOURCE_X:+.3f} m,"
            f" Y={SOURCE_Y:+.3f} m,"
            f" Z={SOURCE_Z:+.3f} m"
        )

        print()
        print("=" * 72)

        # -----------------------------------------------------------
        # Measurements
        # -----------------------------------------------------------

        for measurement in range(
            1,
            NUM_MEASUREMENTS + 1,
        ):

            print(
                f"Measurement {measurement:02d}/"
                f"{NUM_MEASUREMENTS:02d}"
            )

            # Reset MUSIC temporal CSM smoothing so every measurement
            # starts independently.
            localizer.previous_csm = None

            measurement_heatmap = None

            # -------------------------------------------------------
            # Collect several consecutive MUSIC estimates.
            # -------------------------------------------------------

            for _ in range(FRAMES_PER_MEASUREMENT):

                frame = audio.read()

                if frame.shape != (
                    FRAME_SIZE,
                    16,
                ):
                    raise RuntimeError(
                        "Unexpected audio frame shape: "
                        f"{frame.shape}"
                    )

                heatmap = localizer.localize(frame)

                measurement_heatmap = heatmap

                time.sleep(FRAME_INTERVAL)

            if measurement_heatmap is None:
                raise RuntimeError(
                    "No MUSIC result was produced."
                )

            # -------------------------------------------------------
            # Find peak position.
            # -------------------------------------------------------

            (
                estimated_x,
                estimated_y,
                peak,
                row,
                column,
            ) = estimate_position(
                measurement_heatmap,
                x_values,
                y_values,
            )

            error = calculate_position_error(
                estimated_x,
                estimated_y,
            )

            results.append(
                (
                    estimated_x,
                    estimated_y,
                    error,
                )
            )

            print(
                f"  Position : "
                f"X = {estimated_x:+.3f} m   "
                f"Y = {estimated_y:+.3f} m   "
                f"Z = {SEARCH_Z:+.3f} m"
            )

            print(
                f"  Error    : "
                f"{error * 1000:.1f} mm"
            )

            print(
                f"  Peak     : "
                f"{peak:.4f}"
            )

            print(
                f"  Grid     : "
                f"row = {row:03d}, "
                f"column = {column:03d}"
            )

            print("-" * 72)

    finally:

        audio.stop()

    # =================================================================
    # STATISTICS
    # =================================================================

    if not results:
        print()
        print("No measurements were collected.")
        return

    positions = np.asarray(
        results,
        dtype=np.float64,
    )

    x_positions = positions[:, 0]
    y_positions = positions[:, 1]
    errors = positions[:, 2]

    mean_x = float(np.mean(x_positions))
    mean_y = float(np.mean(y_positions))

    std_x = float(np.std(x_positions))
    std_y = float(np.std(y_positions))

    mean_error = float(np.mean(errors))
    minimum_error = float(np.min(errors))
    maximum_error = float(np.max(errors))

    mean_position_error = float(
        np.sqrt(
            (mean_x - SOURCE_X) ** 2
            + (mean_y - SOURCE_Y) ** 2
        )
    )

    # =================================================================
    # FINAL REPORT
    # =================================================================

    print()
    print("=" * 72)
    print("NEAR-FIELD LOCALIZATION STATISTICS")
    print("=" * 72)

    print()
    print("Known source position")
    print(
        f"  X : {SOURCE_X:+.3f} m"
    )
    print(
        f"  Y : {SOURCE_Y:+.3f} m"
    )
    print(
        f"  Z : {SOURCE_Z:+.3f} m"
    )

    print()
    print("Mean estimated position")
    print(
        f"  X : {mean_x:+.3f} m"
    )
    print(
        f"  Y : {mean_y:+.3f} m"
    )
    print(
        f"  Z : {SEARCH_Z:+.3f} m"
    )

    print()
    print("Position standard deviation")
    print(
        f"  X : {std_x:.3f} m"
        f"  ({std_x * 1000:.1f} mm)"
    )
    print(
        f"  Y : {std_y:.3f} m"
        f"  ({std_y * 1000:.1f} mm)"
    )

    print()
    print("Mean-position localization error")
    print(
        f"  XY error : "
        f"{mean_position_error:.3f} m"
        f" ({mean_position_error * 1000:.1f} mm)"
    )

    print()
    print("Individual measurement error")
    print(
        f"  Mean : {mean_error:.3f} m"
        f" ({mean_error * 1000:.1f} mm)"
    )
    print(
        f"  Min  : {minimum_error:.3f} m"
        f" ({minimum_error * 1000:.1f} mm)"
    )
    print(
        f"  Max  : {maximum_error:.3f} m"
        f" ({maximum_error * 1000:.1f} mm)"
    )

    print()
    print("Position range")
    print(
        f"  X : "
        f"{np.min(x_positions):+.3f}"
        f" → "
        f"{np.max(x_positions):+.3f} m"
    )
    print(
        f"  Y : "
        f"{np.min(y_positions):+.3f}"
        f" → "
        f"{np.max(y_positions):+.3f} m"
    )

    print()
    print("=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)

    print()
    print("Interpretation:")
    print(
        "  The source was intentionally placed near "
        "(X,Y) = (0,0)."
    )
    print(
        "  Compare the mean position and the measurement "
        "spread against this known location."
    )
    print()
    print(
        "  Do NOT modify geometry.py or MUSIC yet based on "
        "this experiment alone."
    )
    print(
        "  This run establishes the near-field baseline first."
    )

    print()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()