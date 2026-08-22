"""
test_phase_mapping_v2.py
========================

UMA-16 1 kHz cross-channel phase/coherence experiment.

Purpose
-------
Investigate the relationship between the 16 USB audio channels of the
miniDSP UMA-16 v2 without modifying the production acoustic pipeline.

This experiment measures each channel relative to a reference channel
using:

    1. Cross-spectrum
    2. Relative phase
    3. Estimated time delay
    4. Magnitude-squared coherence
    5. Phase stability across repeated measurements

The experiment is specifically intended to investigate whether the
measured channel relationships are trustworthy enough to help validate
the physical microphone/channel ordering used by the MUSIC system.

IMPORTANT
---------
This script is diagnostic only.

It does NOT modify:

    - acoustic.geometry
    - acoustic.music
    - acoustic.steering
    - channel mapping
    - configuration

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

TEST_FREQUENCY = 1000.0

REFERENCE_CHANNEL = 0

NUM_MEASUREMENTS = 30

WARMUP_SECONDS = 2.0

MEASUREMENT_DELAY_SECONDS = 0.05

# Frequency region used around the target tone.
#
# We inspect several FFT bins around 1 kHz rather than trusting
# only one bin.
FREQUENCY_BAND_HZ = 35.0

# Coherence threshold used only for interpretation.
#
# This does NOT modify or reject measurements.
COHERENCE_THRESHOLD = 0.8


# ============================================================
# FFT UTILITIES
# ============================================================

def get_fft_frequencies() -> np.ndarray:
    """
    Return the positive-frequency FFT bins for the configured
    acquisition parameters.
    """

    return np.fft.rfftfreq(
        FRAME_SIZE,
        1.0 / SAMPLE_RATE,
    )


def get_frequency_bins() -> np.ndarray:
    """
    Return FFT bins surrounding the target frequency.
    """

    frequencies = get_fft_frequencies()

    mask = (
        np.abs(
            frequencies - TEST_FREQUENCY
        )
        <= FREQUENCY_BAND_HZ
    )

    return np.where(mask)[0]


# ============================================================
# SIGNAL PREPROCESSING
# ============================================================

def preprocess_frame(
    frame: np.ndarray,
) -> np.ndarray:
    """
    Remove per-channel DC and apply a Hann window.

    Parameters
    ----------
    frame : numpy.ndarray
        Audio frame of shape (FRAME_SIZE, NUM_CHANNELS).

    Returns
    -------
    numpy.ndarray
        Windowed audio frame.
    """

    frame = frame.astype(
        np.float64,
        copy=False,
    )

    # Remove DC independently from every channel.
    frame = (
        frame
        - np.mean(
            frame,
            axis=0,
            keepdims=True,
        )
    )

    window = np.hanning(
        frame.shape[0]
    )

    return frame * window[:, None]


# ============================================================
# CROSS-SPECTRAL ANALYSIS
# ============================================================

def analyze_frame(
    frame: np.ndarray,
    frequency_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze all channels relative to the reference channel.

    Parameters
    ----------
    frame : numpy.ndarray
        Audio frame.

    frequency_bins : numpy.ndarray
        FFT bins surrounding the target frequency.

    Returns
    -------
    tuple
        relative_phase_degrees
        coherence
        magnitude_db
    """

    processed = preprocess_frame(
        frame
    )

    spectrum = np.fft.rfft(
        processed,
        axis=0,
    )

    reference = spectrum[
        frequency_bins,
        REFERENCE_CHANNEL,
    ]

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    phase_values = np.zeros(
        NUM_CHANNELS,
        dtype=np.float64,
    )

    coherence_values = np.zeros(
        NUM_CHANNELS,
        dtype=np.float64,
    )

    magnitude_values = np.zeros(
        NUM_CHANNELS,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Analyze each channel
    # --------------------------------------------------------

    for channel in range(
        NUM_CHANNELS
    ):

        channel_spectrum = spectrum[
            frequency_bins,
            channel,
        ]

        # ----------------------------------------------------
        # Cross-spectrum
        #
        # S_xy = X * conj(Y)
        #
        # Here the reference is X and the channel is Y.
        # ----------------------------------------------------

        cross_spectrum = (
            channel_spectrum
            * np.conjugate(reference)
        )

        # ----------------------------------------------------
        # Coherence estimate
        #
        # For this narrow-band diagnostic we use:
        #
        # |mean(X * conj(Y))|²
        # ---------------------------------
        # mean(|X|²) mean(|Y|²)
        # ----------------------------------------------------

        cross_average = np.mean(
            cross_spectrum
        )

        reference_power = np.mean(
            np.abs(reference) ** 2
        )

        channel_power = np.mean(
            np.abs(channel_spectrum) ** 2
        )

        denominator = (
            reference_power
            * channel_power
        )

        if denominator > 1e-20:

            coherence = (
                np.abs(
                    cross_average
                ) ** 2
                / denominator
            )

        else:

            coherence = 0.0

        coherence = float(
            np.clip(
                coherence,
                0.0,
                1.0,
            )
        )

        # ----------------------------------------------------
        # Relative phase
        #
        # The phase of the cross-spectrum represents the phase
        # relationship between the channel and the reference.
        # ----------------------------------------------------

        relative_phase = np.angle(
            cross_average,
            deg=True,
        )

        relative_phase = (
            relative_phase
            + 180.0
        ) % 360.0 - 180.0

        # ----------------------------------------------------
        # Channel magnitude
        # ----------------------------------------------------

        magnitude = np.mean(
            np.abs(
                channel_spectrum
            )
        )

        magnitude_db = (
            20.0
            * np.log10(
                magnitude + 1e-12
            )
        )

        phase_values[channel] = (
            relative_phase
        )

        coherence_values[channel] = (
            coherence
        )

        magnitude_values[channel] = (
            magnitude_db
        )

    return (
        phase_values,
        coherence_values,
        magnitude_values,
    )


# ============================================================
# PHASE → TIME DELAY
# ============================================================

def phase_to_delay(
    phase_degrees: np.ndarray,
) -> np.ndarray:
    """
    Convert relative phase at 1 kHz into equivalent time delay.

    This conversion is inherently modulo one period.

    At 1 kHz:

        period = 1 ms

    Therefore the returned delay is wrapped into approximately
    +/- 0.5 ms.

    Parameters
    ----------
    phase_degrees : numpy.ndarray
        Relative phase in degrees.

    Returns
    -------
    numpy.ndarray
        Wrapped relative delay in seconds.
    """

    period = (
        1.0
        / TEST_FREQUENCY
    )

    delay = (
        phase_degrees
        / 360.0
        * period
    )

    # Wrap to +/- half a period.
    delay = (
        delay
        + period / 2.0
    ) % period - period / 2.0

    return delay


# ============================================================
# STATISTICAL UTILITIES
# ============================================================

def circular_mean(
    phase_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculate circular mean phase for each channel.
    """

    radians = np.radians(
        phase_matrix
    )

    mean_vector = np.mean(
        np.exp(
            1j * radians
        ),
        axis=0,
    )

    return np.degrees(
        np.angle(
            mean_vector
        )
    )


def circular_stability(
    phase_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculate phase concentration.

    1.0 -> highly stable phase
    0.0 -> highly unstable phase
    """

    radians = np.radians(
        phase_matrix
    )

    return np.abs(
        np.mean(
            np.exp(
                1j * radians
            ),
            axis=0,
        )
    )


# ============================================================
# OUTPUT
# ============================================================

def print_results(
    phase: np.ndarray,
    delay: np.ndarray,
    coherence: np.ndarray,
    magnitude: np.ndarray,
    stability: np.ndarray,
) -> None:
    """
    Print the final channel analysis.
    """

    print()
    print("=" * 82)
    print("UMA-16 1 kHz CROSS-CHANNEL RESULTS")
    print("=" * 82)

    print()
    print(
        f"Reference channel : CH{REFERENCE_CHANNEL + 1:02d}"
    )

    print(
        f"Target frequency  : {TEST_FREQUENCY:.1f} Hz"
    )

    print()

    print(
        "CH      Magnitude      Rel. Phase      Delay        "
        "Coherence      Stability"
    )

    print(
        "        (dB)              (deg)         (us)"
    )

    print("-" * 82)

    for channel in range(
        NUM_CHANNELS
    ):

        delay_us = (
            delay[channel]
            * 1e6
        )

        print(
            f"CH{channel + 1:02d}   "
            f"{magnitude[channel]:10.2f}   "
            f"{phase[channel]:+12.2f}   "
            f"{delay_us:+10.2f}   "
            f"{coherence[channel]:10.3f}   "
            f"{stability[channel]:10.3f}"
        )

    print("-" * 82)


def print_phase_map(
    phase: np.ndarray,
) -> None:
    """
    Print the relative phase in the assumed 4x4 layout.
    """

    print()
    print("=" * 82)
    print("4 × 4 RELATIVE PHASE MAP")
    print("=" * 82)

    print()
    print(
        "Current assumed physical ordering:"
    )

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
                f"{phase[channel]:+7.1f}"
            )

        print(
            "  ".join(values)
        )


def print_diagnostic_summary(
    coherence: np.ndarray,
    stability: np.ndarray,
) -> None:
    """
    Print a diagnostic interpretation of the measurements.
    """

    print()
    print("=" * 82)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 82)

    print()

    # Exclude the reference channel when calculating summaries.
    channels = [
        i
        for i in range(NUM_CHANNELS)
        if i != REFERENCE_CHANNEL
    ]

    mean_coherence = float(
        np.mean(
            coherence[channels]
        )
    )

    mean_stability = float(
        np.mean(
            stability[channels]
        )
    )

    print(
        f"Mean cross-channel coherence : "
        f"{mean_coherence:.3f}"
    )

    print(
        f"Mean phase stability         : "
        f"{mean_stability:.3f}"
    )

    print()

    high_coherence = [
        i + 1
        for i in channels
        if coherence[i]
        >= COHERENCE_THRESHOLD
    ]

    low_coherence = [
        i + 1
        for i in channels
        if coherence[i]
        < COHERENCE_THRESHOLD
    ]

    print(
        f"Channels with coherence >= "
        f"{COHERENCE_THRESHOLD:.2f}:"
    )

    if high_coherence:
        print(
            "  "
            + ", ".join(
                f"CH{i:02d}"
                for i in high_coherence
            )
        )
    else:
        print("  None")

    print()

    print(
        f"Channels with coherence < "
        f"{COHERENCE_THRESHOLD:.2f}:"
    )

    if low_coherence:
        print(
            "  "
            + ", ".join(
                f"CH{i:02d}"
                for i in low_coherence
            )
        )
    else:
        print("  None")

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "High coherence means the measured phase relationship "
        "is more trustworthy."
    )

    print(
        "Low coherence means the phase measurement should not "
        "be used to infer microphone ordering."
    )

    print(
        "This experiment does not automatically produce a "
        "corrected channel map."
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the Phase Mapping v2 experiment.
    """

    print()
    print("=" * 82)
    print("UMA-16 PHASE MAPPING v2")
    print("=" * 82)

    print()
    print(
        "Controlled 1 kHz cross-channel diagnostic"
    )

    print()
    print(
        "Keep the 1 kHz source stationary."
    )

    print(
        "Keep the UMA-16 stationary."
    )

    print(
        "Do not change the microphone geometry."
    )

    print()

    frequency_bins = (
        get_frequency_bins()
    )

    print(
        f"Frequency bins analyzed : "
        f"{len(frequency_bins)}"
    )

    print(
        f"Frequency range          : "
        f"{get_fft_frequencies()[frequency_bins[0]]:.2f}"
        f" - "
        f"{get_fft_frequencies()[frequency_bins[-1]]:.2f} Hz"
    )

    print()

    audio = AudioAcquisition()

    phase_measurements = []
    coherence_measurements = []
    magnitude_measurements = []

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
            "[Measurement] Starting..."
        )

        for measurement in range(
            NUM_MEASUREMENTS
        ):

            frame = audio.read()

            expected_shape = (
                FRAME_SIZE,
                NUM_CHANNELS,
            )

            if frame.shape != expected_shape:

                print(
                    f"[Warning] Unexpected frame shape: "
                    f"{frame.shape}"
                )

                continue

            (
                phase,
                coherence,
                magnitude,
            ) = analyze_frame(
                frame,
                frequency_bins,
            )

            phase_measurements.append(
                phase
            )

            coherence_measurements.append(
                coherence
            )

            magnitude_measurements.append(
                magnitude
            )

            print(
                f"  Measurement "
                f"{measurement + 1:02d}/"
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

    if not phase_measurements:

        print()
        print(
            "[Error] No measurements collected."
        )

        return

    # --------------------------------------------------------
    # Convert measurements to arrays
    # --------------------------------------------------------

    phase_matrix = np.asarray(
        phase_measurements,
        dtype=np.float64,
    )

    coherence_matrix = np.asarray(
        coherence_measurements,
        dtype=np.float64,
    )

    magnitude_matrix = np.asarray(
        magnitude_measurements,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Average results
    # --------------------------------------------------------

    mean_phase = circular_mean(
        phase_matrix
    )

    mean_coherence = np.mean(
        coherence_matrix,
        axis=0,
    )

    mean_magnitude = np.mean(
        magnitude_matrix,
        axis=0,
    )

    phase_stability = circular_stability(
        phase_matrix
    )

    # --------------------------------------------------------
    # Convert phase to delay
    # --------------------------------------------------------

    mean_delay = phase_to_delay(
        mean_phase
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_results(
        mean_phase,
        mean_delay,
        mean_coherence,
        mean_magnitude,
        phase_stability,
    )

    print_phase_map(
        mean_phase
    )

    print_diagnostic_summary(
        mean_coherence,
        phase_stability,
    )

    print()
    print("=" * 82)
    print("PHASE MAPPING v2 COMPLETE")
    print("=" * 82)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()