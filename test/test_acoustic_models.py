"""
test_acoustic_models.py
=======================

Milestone 15 unit test.

Tests the project-level acoustic data model without requiring:

- UMA-16 hardware
- sounddevice
- MUSIC execution
- Acoular
- ESP32-CAM
- YOLO
- synchronization
- fusion

The purpose is to verify that the acoustic subsystem has a
clean interface for the future synchronization and fusion layers.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import math

from acoustic.models import (
    AcousticFrame,
    AcousticLocalization,
    create_acoustic_frame,
    create_acoustic_localization,
)


# ============================================================
# Test configuration
# ============================================================

TEST_X = 0.025
TEST_Y = -0.010
TEST_Z = 0.070

TEST_RESPONSE = 0.92
TEST_FREQUENCY = 1000.0
TEST_CONFIDENCE = 0.85

TEST_TIMESTAMP = 12345.678
TEST_FRAME_ID = 17


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 72)
    print("MILESTONE 15 - ACOUSTIC DATA MODEL TEST")
    print("=" * 72)

    # --------------------------------------------------------
    # AcousticLocalization
    # --------------------------------------------------------

    localization = AcousticLocalization(
        x=TEST_X,
        y=TEST_Y,
        z=TEST_Z,
        response=TEST_RESPONSE,
        frequency=TEST_FREQUENCY,
        confidence=TEST_CONFIDENCE,
    )

    assert localization.x == TEST_X
    assert localization.y == TEST_Y
    assert localization.z == TEST_Z

    assert localization.response == TEST_RESPONSE
    assert localization.frequency == TEST_FREQUENCY
    assert localization.confidence == TEST_CONFIDENCE

    assert localization.valid is True
    assert localization.source_count == 1

    print()
    print("[PASS] AcousticLocalization created successfully")

    print(
        f"       Position   : "
        f"({localization.x:.3f}, "
        f"{localization.y:.3f}, "
        f"{localization.z:.3f}) m"
    )

    print(
        f"       Response   : "
        f"{localization.response:.3f}"
    )

    print(
        f"       Frequency  : "
        f"{localization.frequency:.1f} Hz"
    )

    print(
        f"       Confidence : "
        f"{localization.confidence:.3f}"
    )

    # --------------------------------------------------------
    # Position helpers
    # --------------------------------------------------------

    assert localization.position == (
        TEST_X,
        TEST_Y,
        TEST_Z,
    )

    assert localization.xy_position == (
        TEST_X,
        TEST_Y,
    )

    distance = localization.distance_from_origin()

    expected_distance = math.sqrt(
        TEST_X ** 2
        + TEST_Y ** 2
        + TEST_Z ** 2
    )

    assert math.isclose(
        distance,
        expected_distance,
        rel_tol=1e-9,
    )

    print("[PASS] Position helpers work")

    print(
        f"       Distance from array origin: "
        f"{distance:.4f} m"
    )

    # --------------------------------------------------------
    # AcousticFrame
    # --------------------------------------------------------

    frame = AcousticFrame(
        frame_id=TEST_FRAME_ID,
        timestamp=TEST_TIMESTAMP,
        localization=localization,
        sample_rate=48000.0,
        frame_size=4096,
        processing_time=0.085,
    )

    assert frame.frame_id == TEST_FRAME_ID
    assert frame.timestamp == TEST_TIMESTAMP

    assert frame.localization is localization

    assert frame.sample_rate == 48000.0
    assert frame.frame_size == 4096
    assert frame.processing_time == 0.085

    print()
    print("[PASS] AcousticFrame created successfully")

    print(
        f"       Frame ID    : "
        f"{frame.frame_id}"
    )

    print(
        f"       Timestamp   : "
        f"{frame.timestamp}"
    )

    print(
        f"       Sample rate : "
        f"{frame.sample_rate:.0f} Hz"
    )

    print(
        f"       Frame size  : "
        f"{frame.frame_size}"
    )

    # --------------------------------------------------------
    # Frame convenience properties
    # --------------------------------------------------------

    assert frame.has_localization is True
    assert frame.position == (
        TEST_X,
        TEST_Y,
        TEST_Z,
    )

    assert frame.response == TEST_RESPONSE
    assert frame.frequency == TEST_FREQUENCY
    assert frame.confidence == TEST_CONFIDENCE

    print("[PASS] AcousticFrame convenience properties work")

    # --------------------------------------------------------
    # Factory helper
    # --------------------------------------------------------

    localization_2 = create_acoustic_localization(
        position=(
            -0.020,
            0.015,
            0.070,
        ),
        response=0.88,
        frequency=1000.0,
        confidence=0.80,
    )

    frame_2 = create_acoustic_frame(
        frame_id=18,
        timestamp=12345.763,
        localization=localization_2,
        sample_rate=48000.0,
        frame_size=4096,
        processing_time=0.086,
        metadata={
            "source": "MUSIC",
            "test": True,
        },
    )

    assert frame_2.frame_id == 18
    assert frame_2.localization is localization_2

    assert frame_2.metadata["source"] == "MUSIC"
    assert frame_2.metadata["test"] is True

    print("[PASS] Factory helpers work")

    # --------------------------------------------------------
    # Empty acoustic frame
    # --------------------------------------------------------

    empty_frame = AcousticFrame(
        frame_id=19,
        timestamp=12345.850,
        localization=None,
    )

    assert empty_frame.has_localization is False
    assert empty_frame.position is None
    assert empty_frame.response is None
    assert empty_frame.frequency is None
    assert empty_frame.confidence is None

    print("[PASS] Empty AcousticFrame handled correctly")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    try:

        AcousticLocalization(
            x=float("nan"),
            y=0.0,
            z=0.0,
            response=1.0,
        )

    except ValueError:

        print("[PASS] Invalid localization rejected")

    else:

        raise AssertionError(
            "Invalid NaN localization was not rejected."
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("MILESTONE 15 TEST PASSED")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()