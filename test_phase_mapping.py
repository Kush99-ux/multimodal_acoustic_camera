"""
test_phase_mapping.py
=====================

UMA-16 1 kHz phase-mapping experiment.

Purpose
-------
This experiment investigates the relationship between the 16 USB audio
channels of the miniDSP UMA-16 v2.

The current acoustic localization system assumes:

    Audio CH01 -> Physical MIC01
    Audio CH02 -> Physical MIC02
    ...
    Audio CH16 -> Physical MIC16

This assumption has not yet been experimentally validated.

A controlled 1 kHz sine-wave source is used to measure the complex
frequency-domain response of every audio channel.

For each channel, this test calculates:

    - RMS level
    - 1 kHz magnitude
    - absolute FFT phase
    - phase relative to a reference channel
    - phase in degrees

The result is printed in the current assumed 4x4 physical geometry
ordering so that channel ordering can be investigated.

IMPORTANT
---------
This experiment does NOT modify the microphone geometry.

It also does NOT modify the MUSIC algorithm.

It is purely a measurement/diagnostic experiment.

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
from config.settings import (
    SAMPLE_RATE,
    FRAME_SIZE,
    NUM_CHANNELS,
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Frequency of the controlled test signal.
TEST_FREQUENCY = 1000.0

# Number of samples used for each phase measurement.
#
# We use the same frame size as the acoustic localization system.
ANALYSIS_FRAME_SIZE = FRAME_SIZE

# Number of measurements to average.
#
# Averaging multiple frames makes the phase estimate less sensitive
# to random noise.
NUM_MEASUREMENTS = 20

# Seconds to allow the audio buffer to fill before measurement.
WARMUP_SECONDS = 2.0

# Delay between measurements.
MEASUREMENT_DELAY_SECONDS = 0.05

# Channel used as the phase reference.
#
# CH01 is used because it corresponds to the first microphone in
# the current assumed geometry.
REFERENCE_CHANNEL = 0


# ============================================================
# PHASE UTILITIES
# ============================================================

def wrap_phase_degrees(phase: float) -> float:
    """
    Wrap a phase angle into the range [-180, +180) degrees.

    Parameters
    ----------
    phase : float
        Phase angle in degrees.

    Returns
    -------
    float
        Wrapped phase angle.
    """

    return (phase + 180.0) % 360.0 - 180.0


def nearest_fft_bin(
    frequency: float,
    sample_rate: int,
    frame_size: int,
) -> int:
    """
    Find the FFT bin closest to a requested frequency.

    Parameters
    ----------
    frequency : float
        Frequency in Hz.

    sample_rate : int
        Audio sample rate in Hz.

    frame_size : int
        Number of samples in the FFT frame.

    Returns
    -------
    int
        Closest FFT bin index.
    """

    frequencies = np.fft.rfftfreq(
        frame_size,
        1.0 / sample_rate,
    )

    return int(
        np.argmin(
            np.abs(frequencies - frequency)
        )
    )


# ============================================================
# SIGNAL ANALYSIS
# ============================================================

def analyze_frame(
    frame: np.ndarray,
    fft_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate magnitude and phase at the test frequency.

    Parameters
    ----------
    frame : numpy.ndarray
        Audio frame with shape:

            (FRAME_SIZE, NUM_CHANNELS)

    fft_bin : int
        FFT bin corresponding to the test frequency.

    Returns
    -------
    tuple
        magnitude_db, phase_degrees

        Both arrays have length NUM_CHANNELS.
    """

    # --------------------------------------------------------
    # Remove per-channel DC offset.
    # --------------------------------------------------------

    frame = frame - np.mean(
        frame,
        axis=0,
        keepdims=True,
    )

    # --------------------------------------------------------
    # Apply a Hann window.
    #
    # This reduces spectral leakage around the 1 kHz tone.
    # --------------------------------------------------------

    window = np.hanning(
        frame.shape[0]
    )

    windowed_frame = (
        frame * window[:, None]
    )

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    spectrum = np.fft.rfft(
        windowed_frame,
        axis=0,
    )

    # --------------------------------------------------------
    # Extract complex FFT coefficient at 1 kHz.
    # --------------------------------------------------------

    complex_values = spectrum[
        fft_bin
    ]

    # --------------------------------------------------------
    # Magnitude
    # --------------------------------------------------------

    magnitude = np.abs(
        complex_values
    )

    # Convert magnitude to dB.

    magnitude_db = (
        20.0
        * np.log10(
            magnitude + 1e-12
        )
    )

    # --------------------------------------------------------
    # Phase
    # --------------------------------------------------------

    phase_degrees = np.degrees(
        np.angle(
            complex_values
        )
    )

    return (
        magnitude_db.astype(np.float64),
        phase_degrees.astype(np.float64),
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main() -> None:
    """
    Run the UMA-16 1 kHz phase-mapping experiment.
    """

    print()
    print("=" * 72)
    print("UMA-16 1 kHz PHASE MAPPING EXPERIMENT")
    print("=" * 72)

    print()
    print("Purpose:")
    print("  Measure the 1 kHz magnitude and phase of every")
    print("  UMA-16 USB audio channel.")

    print()
    print("Current assumption:")
    print("  CH01 -> MIC01")
    print("  CH02 -> MIC02")
    print("  ...")
    print("  CH16 -> MIC16")

    print()
    print("This test does NOT modify:")
    print("  - microphone geometry")
    print("  - channel mapping")
    print("  - MUSIC")
    print("  - steering vectors")

    print()
    print("=" * 72)

    # --------------------------------------------------------
    # Display experiment parameters
    # --------------------------------------------------------

    fft_bin = nearest_fft_bin(
        TEST_FREQUENCY,
        SAMPLE_RATE,
        ANALYSIS_FRAME_SIZE,
    )

    actual_fft_frequency = (
        fft_bin
        * SAMPLE_RATE
        / ANALYSIS_FRAME_SIZE
    )

    print()
    print("[Experiment]")
    print(
        f"  Requested frequency : "
        f"{TEST_FREQUENCY:.1f} Hz"
    )
    print(
        f"  FFT bin             : "
        f"{fft_bin}"
    )
    print(
        f"  Actual FFT frequency: "
        f"{actual_fft_frequency:.3f} Hz"
    )
    print(
        f"  Sample rate         : "
        f"{SAMPLE_RATE} Hz"
    )
    print(
        f"  Frame size          : "
        f"{ANALYSIS_FRAME_SIZE}"
    )
    print(
        f"  Measurements        : "
        f"{NUM_MEASUREMENTS}"
    )
    print(
        f"  Reference channel   : "
        f"CH{REFERENCE_CHANNEL + 1:02d}"
    )

    # --------------------------------------------------------
    # Start audio
    # --------------------------------------------------------

    audio = AudioAcquisition()

    measurements_magnitude = []
    measurements_phase = []

    try:

        audio.start()

        print()
        print(
            f"[Audio] Warming up for "
            f"{WARMUP_SECONDS:.1f} seconds..."
        )

        time.sleep(
            WARMUP_SECONDS
        )

        print()
        print(
            "[Measurement] Capturing 1 kHz phase data..."
        )

        print(
            "Keep the 1 kHz source stationary "
            "during the measurement."
        )

        print()

        # ----------------------------------------------------
        # Collect measurements
        # ----------------------------------------------------

        for measurement_index in range(
            NUM_MEASUREMENTS
        ):

            frame = audio.read()

            # ------------------------------------------------
            # Validate frame
            # ------------------------------------------------

            expected_shape = (
                ANALYSIS_FRAME_SIZE,
                NUM_CHANNELS,
            )

            if frame.shape != expected_shape:

                print(
                    f"[Warning] Unexpected frame shape: "
                    f"{frame.shape}"
                )

                continue

            # ------------------------------------------------
            # Analyze 1 kHz component
            # ------------------------------------------------

            magnitude_db, phase_degrees = (
                analyze_frame(
                    frame,
                    fft_bin,
                )
            )

            measurements_magnitude.append(
                magnitude_db
            )

            measurements_phase.append(
                phase_degrees
            )

            print(
                f"  Measurement "
                f"{measurement_index + 1:02d}/"
                f"{NUM_MEASUREMENTS}"
            )

            time.sleep(
                MEASUREMENT_DELAY_SECONDS
            )

    except KeyboardInterrupt:

        print()
        print(
            "[Experiment] Interrupted by user."
        )

    finally:

        audio.stop()

    # --------------------------------------------------------
    # Validate collected data
    # --------------------------------------------------------

    if not measurements_magnitude:

        print()
        print(
            "[Error] No valid measurements were collected."
        )

        return

    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    magnitude_matrix = np.asarray(
        measurements_magnitude,
        dtype=np.float64,
    )

    phase_matrix = np.asarray(
        measurements_phase,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Average magnitude
    # --------------------------------------------------------

    mean_magnitude_db = np.mean(
        magnitude_matrix,
        axis=0,
    )

    # --------------------------------------------------------
    # Phase averaging
    #
    # DO NOT use a simple arithmetic mean for phase because
    # angles wrap around at ±180 degrees.
    #
    # Instead average the corresponding unit phasors.
    # --------------------------------------------------------

    phase_radians = np.radians(
        phase_matrix
    )

    mean_complex_phase = np.mean(
        np.exp(
            1j * phase_radians
        ),
        axis=0,
    )

    mean_phase_degrees = np.degrees(
        np.angle(
            mean_complex_phase
        )
    )

    # --------------------------------------------------------
    # Relative phase to reference channel
    # --------------------------------------------------------

    reference_phase = (
        mean_phase_degrees[
            REFERENCE_CHANNEL
        ]
    )

    relative_phase_degrees = np.array(
        [
            wrap_phase_degrees(
                phase - reference_phase
            )
            for phase in mean_phase_degrees
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Phase stability
    #
    # Calculate circular concentration as a simple indication
    # of how stable the phase measurement was across frames.
    #
    # 1.0 = very stable
    # 0.0 = highly unstable
    # --------------------------------------------------------

    phase_stability = np.abs(
        np.mean(
            np.exp(
                1j * phase_radians
            ),
            axis=0,
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 72)
    print("UMA-16 1 kHz PHASE RESULTS")
    print("=" * 72)

    print()
    print(
        "Reference channel: "
        f"CH{REFERENCE_CHANNEL + 1:02d}"
    )

    print()
    print(
        "CH      Magnitude      Absolute       Relative      Stability"
    )
    print(
        "        (dB)           Phase (°)       Phase (°)"
    )
    print("-" * 72)

    for channel in range(
        NUM_CHANNELS
    ):

        print(
            f"CH{channel + 1:02d}   "
            f"{mean_magnitude_db[channel]:10.2f}   "
            f"{mean_phase_degrees[channel]:12.2f}   "
            f"{relative_phase_degrees[channel]:12.2f}   "
            f"{phase_stability[channel]:9.3f}"
        )

    print("-" * 72)

    # ========================================================
    # 4 × 4 PHYSICAL ORDER
    # ========================================================

    print()
    print("=" * 72)
    print("4 × 4 CHANNEL PHASE MAP")
    print("=" * 72)

    print()
    print(
        "Values shown are relative phase to CH01 in degrees."
    )

    print()
    print("Physical geometry assumption:")
    print()
    print(
        "CH01   CH02   CH03   CH04"
    )
    print(
        "CH05   CH06   CH07   CH08"
    )
    print(
        "CH09   CH10   CH11   CH12"
    )
    print(
        "CH13   CH14   CH15   CH16"
    )

    print()

    for row in range(4):

        values = []

        for column in range(4):

            channel = (
                row * 4
                + column
            )

            values.append(
                f"{relative_phase_degrees[channel]:+7.1f}"
            )

        print(
            "  ".join(values)
        )

    # ========================================================
    # CHANNEL ORDERING SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("CHANNEL PHASE ORDERING")
    print("=" * 72)

    # Sort channels according to relative phase.
    phase_order = np.argsort(
        relative_phase_degrees
    )

    print()
    print(
        "Channels sorted by relative phase:"
    )

    for rank, channel in enumerate(
        phase_order,
        start=1,
    ):

        print(
            f"{rank:02d}. "
            f"CH{channel + 1:02d}  "
            f"{relative_phase_degrees[channel]:+8.2f}°"
        )

    # ========================================================
    # FINAL NOTES
    # ========================================================

    print()
    print("=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The measured phase values do NOT automatically establish "
        "a corrected microphone/channel mapping."
    )

    print(
        "They are diagnostic measurements that will be compared "
        "against the physical microphone geometry."
    )

    print()
    print(
        "Do not modify geometry.py or MUSIC based only on this "
        "single measurement."
    )

    print()


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()