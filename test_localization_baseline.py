"""
test_localization_baseline.py
=============================

Controlled numerical baseline test for the MUSIC acoustic localization
pipeline.

Purpose
-------
This test validates the acoustic localization subsystem independently
from the camera, YOLO segmentation, visualization, and fusion systems.

Pipeline
--------
UMA-16 microphone array
        |
        v
AudioAcquisition
        |
        v
4096-sample multichannel frame
        |
        v
MusicLocalizer
        |
        v
MUSIC localization heatmap
        |
        v
Peak extraction
        |
        v
Estimated X / Y / Z position

The purpose of this experiment is NOT to produce a polished heatmap.

Instead, it answers the more important question:

    "Where does the current MUSIC implementation believe
     the dominant acoustic source is located?"

Author
------
Kush Sahu

Project
-------
Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import time

import numpy as np

from acoustic.acquisition import AudioAcquisition
from acoustic.music import MusicLocalizer


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Number of localization iterations to perform.
NUM_ITERATIONS = 10

# Time allowed for the acquisition buffer to fill before the first
# localization result is calculated.
WARMUP_SECONDS = 2.0

# Delay between consecutive localization iterations.
#
# This prevents the test from continuously hammering the CPU and
# makes the printed results easier to read.
ITERATION_DELAY_SECONDS = 0.1


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def print_header() -> None:
    """Print the experiment header."""

    print()
    print("=" * 64)
    print("MUSIC ACOUSTIC LOCALIZATION BASELINE")
    print("=" * 64)
    print()
    print("Purpose:")
    print("  Measure the numerical position estimated by MUSIC.")
    print()
    print("This test does NOT use:")
    print("  - ESP32 camera")
    print("  - YOLO")
    print("  - Heatmap visualization")
    print("  - Audio/video fusion")
    print()
    print("=" * 64)
    print()


def print_result(
    iteration: int,
    peak: dict[str, float | int],
) -> None:
    """
    Print one MUSIC localization result.

    Parameters
    ----------
    iteration : int
        Current experiment iteration.

    peak : dict
        Peak position returned by MusicLocalizer.
    """

    x = float(peak["x"])
    y = float(peak["y"])
    z = float(peak["z"])
    value = float(peak["value"])

    row = int(peak["row"])
    column = int(peak["column"])

    print(
        f"Iteration {iteration:02d}"
    )

    print(
        f"  Position : "
        f"X = {x:+.3f} m   "
        f"Y = {y:+.3f} m   "
        f"Z = {z:+.3f} m"
    )

    print(
        f"  Response : {value:.4f}"
    )

    print(
        f"  Grid     : "
        f"row = {row:03d}, "
        f"column = {column:03d}"
    )

    print("-" * 64)


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    results: list[dict[str, float | int]],
) -> None:
    """
    Calculate and print statistics from the localization results.

    Parameters
    ----------
    results : list[dict]
        Collection of MUSIC peak results.
    """

    if not results:
        print("No localization results were collected.")
        return

    x_values = np.asarray(
        [float(result["x"]) for result in results],
        dtype=np.float64,
    )

    y_values = np.asarray(
        [float(result["y"]) for result in results],
        dtype=np.float64,
    )

    z_values = np.asarray(
        [float(result["z"]) for result in results],
        dtype=np.float64,
    )

    response_values = np.asarray(
        [float(result["value"]) for result in results],
        dtype=np.float64,
    )

    print()
    print("=" * 64)
    print("LOCALIZATION STATISTICS")
    print("=" * 64)

    print()
    print("Mean estimated position")
    print(
        f"  X : {np.mean(x_values):+.3f} m"
    )
    print(
        f"  Y : {np.mean(y_values):+.3f} m"
    )
    print(
        f"  Z : {np.mean(z_values):+.3f} m"
    )

    print()
    print("Position standard deviation")
    print(
        f"  X : {np.std(x_values):.3f} m"
    )
    print(
        f"  Y : {np.std(y_values):.3f} m"
    )
    print(
        f"  Z : {np.std(z_values):.3f} m"
    )

    print()
    print("Position range")

    print(
        f"  X : "
        f"{np.min(x_values):+.3f} → "
        f"{np.max(x_values):+.3f} m"
    )

    print(
        f"  Y : "
        f"{np.min(y_values):+.3f} → "
        f"{np.max(y_values):+.3f} m"
    )

    print(
        f"  Z : "
        f"{np.min(z_values):+.3f} → "
        f"{np.max(z_values):+.3f} m"
    )

    print()
    print("MUSIC response")
    print(
        f"  Mean : {np.mean(response_values):.4f}"
    )
    print(
        f"  Min  : {np.min(response_values):.4f}"
    )
    print(
        f"  Max  : {np.max(response_values):.4f}"
    )

    print()
    print("=" * 64)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main() -> None:
    """
    Run the controlled MUSIC localization baseline experiment.
    """

    print_header()

    # --------------------------------------------------------
    # Initialize localization engine
    # --------------------------------------------------------

    print("[System] Initializing MUSIC localizer...")

    localizer = MusicLocalizer()

    print()
    print(
        f"[System] Grid shape : {localizer.grid_shape}"
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

    print()

    # --------------------------------------------------------
    # Start UMA-16 acquisition
    # --------------------------------------------------------

    audio = AudioAcquisition()

    results: list[
        dict[str, float | int]
    ] = []

    try:
        audio.start()

        # ----------------------------------------------------
        # Allow acquisition buffer to fill
        # ----------------------------------------------------

        print()
        print(
            f"[Audio] Warming up for "
            f"{WARMUP_SECONDS:.1f} seconds..."
        )

        time.sleep(WARMUP_SECONDS)

        print()
        print(
            "Starting localization experiment..."
        )

        print(
            "Keep the acoustic source approximately "
            "stationary during this measurement."
        )

        print()
        print("=" * 64)

        # ----------------------------------------------------
        # Localization loop
        # ----------------------------------------------------

        for iteration in range(
            1,
            NUM_ITERATIONS + 1,
        ):

            # Get latest sliding audio frame.
            frame = audio.read()

            # ------------------------------------------------
            # Basic frame validation
            # ------------------------------------------------

            if frame.shape != (
                localizer.frame_size,
                16,
            ):
                print(
                    f"[Warning] Unexpected frame shape: "
                    f"{frame.shape}"
                )

                continue

            # ------------------------------------------------
            # Run MUSIC + peak extraction
            # ------------------------------------------------

            heatmap, peak = (
                localizer.localize_with_peak(
                    frame
                )
            )

            # Keep the heatmap available for future debugging,
            # while deliberately not displaying it here.
            #
            # This also confirms that the numerical peak is being
            # extracted from the same heatmap returned by MUSIC.
            _ = heatmap

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            results.append(peak)

            # ------------------------------------------------
            # Print result
            # ------------------------------------------------

            print_result(
                iteration,
                peak,
            )

            time.sleep(
                ITERATION_DELAY_SECONDS
            )

    except KeyboardInterrupt:

        print()
        print(
            "[Experiment] Interrupted by user."
        )

    finally:

        # ----------------------------------------------------
        # Always stop audio cleanly
        # ----------------------------------------------------

        audio.stop()

    # ========================================================
    # Final statistics
    # ========================================================

    calculate_statistics(
        results
    )

    print()
    print("Experiment complete.")
    print()


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()