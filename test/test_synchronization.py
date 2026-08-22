"""
test_synchronization.py
=======================

Milestone 16 test.

Tests timestamp synchronization without requiring:

- ESP32-CAM
- UMA-16
- YOLO
- MUSIC
- Acoular
- OpenCV camera access

The test uses synthetic VisionFrame and AcousticFrame
objects to verify temporal matching.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

from acoustic.models import (
    AcousticFrame,
    AcousticLocalization,
)

from fusion.synchronization import (
    SYNC_TOLERANCE_MS,
    TimestampSynchronizer,
)

from vision.models import VisionFrame


# ============================================================
# Test object factories
# ============================================================


def create_vision_frame(
    frame_id: int,
    timestamp: float,
) -> VisionFrame:
    """
    Create a minimal VisionFrame for synchronization testing.

    Only fields actually required by the VisionFrame data model
    should be supplied here.
    """

    return VisionFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        width=320,
        height=240,
        detections=[],
    )


def create_acoustic_frame(
    frame_id: int,
    timestamp: float,
) -> AcousticFrame:
    """
    Create a minimal AcousticFrame for synchronization testing.
    """

    localization = AcousticLocalization(
        x=0.0,
        y=0.0,
        z=1.0,
        response=0.90,
        frequency=1000.0,
        confidence=0.80,
    )

    return AcousticFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        localization=localization,
        sample_rate=48000.0,
        frame_size=4096,
    )


# ============================================================
# Main test
# ============================================================


def main() -> None:

    print()
    print("=" * 72)
    print("MILESTONE 16 - TIMESTAMP SYNCHRONIZATION TEST")
    print("=" * 72)

    print()
    print(
        f"[System] Synchronization tolerance : "
        f"{SYNC_TOLERANCE_MS:.1f} ms"
    )

    synchronizer = TimestampSynchronizer()

    # ========================================================
    # Test 1
    # ========================================================

    assert synchronizer.get_tolerance_ms() == 100.0

    print(
        "[PASS] Synchronizer initialized with 100 ms tolerance"
    )

    # ========================================================
    # Test 2
    # 30 ms difference
    # ========================================================

    vision = create_vision_frame(
        frame_id=1,
        timestamp=100.000,
    )

    acoustic = create_acoustic_frame(
        frame_id=1,
        timestamp=100.030,
    )

    pair = synchronizer.synchronize_pair(
        vision,
        acoustic,
    )

    assert pair is not None

    assert abs(
        pair.timestamp_difference_ms - 30.0
    ) < 1e-6

    print(
        "[PASS] 30 ms difference → MATCH"
    )

    # ========================================================
    # Test 3
    # 95 ms difference
    # ========================================================

    acoustic = create_acoustic_frame(
        frame_id=2,
        timestamp=100.095,
    )

    pair = synchronizer.synchronize_pair(
        vision,
        acoustic,
    )

    assert pair is not None

    assert abs(
        pair.timestamp_difference_ms - 95.0
    ) < 1e-6

    print(
        "[PASS] 95 ms difference → MATCH"
    )

    # ========================================================
    # Test 4
    # Exactly 100 ms
    # ========================================================

    acoustic = create_acoustic_frame(
        frame_id=3,
        timestamp=100.100,
    )

    pair = synchronizer.synchronize_pair(
        vision,
        acoustic,
    )

    assert pair is not None

    print(
        "[PASS] 100 ms difference → MATCH"
    )

    # ========================================================
    # Test 5
    # 150 ms difference
    # ========================================================

    acoustic = create_acoustic_frame(
        frame_id=4,
        timestamp=100.150,
    )

    pair = synchronizer.synchronize_pair(
        vision,
        acoustic,
    )

    assert pair is None

    print(
        "[PASS] 150 ms difference → NO MATCH"
    )

    # ========================================================
    # Test 6
    # Acoustic frame before vision frame
    # ========================================================

    acoustic = create_acoustic_frame(
        frame_id=5,
        timestamp=99.950,
    )

    pair = synchronizer.synchronize_pair(
        vision,
        acoustic,
    )

    assert pair is not None

    assert abs(
        pair.timestamp_difference_ms - 50.0
    ) < 1e-6

    print(
        "[PASS] Acoustic-before-vision matching works"
    )

    # ========================================================
    # Test 7
    # Closest-frame selection
    # ========================================================

    acoustic_frames = [
        create_acoustic_frame(
            frame_id=10,
            timestamp=99.930,
        ),
        create_acoustic_frame(
            frame_id=11,
            timestamp=100.020,
        ),
        create_acoustic_frame(
            frame_id=12,
            timestamp=100.080,
        ),
        create_acoustic_frame(
            frame_id=13,
            timestamp=100.140,
        ),
    ]

    pair = synchronizer.match(
        vision,
        acoustic_frames,
    )

    assert pair is not None

    # CHOOSE 100.020 because it is only 20 ms
    # away from the vision timestamp.
    assert pair.acoustic.frame_id == 11

    assert abs(
        pair.timestamp_difference_ms - 20.0
    ) < 1e-6

    print(
        "[PASS] Closest acoustic frame selected correctly"
    )

    # ========================================================
    # Test 8
    # Empty acoustic buffer
    # ========================================================

    pair = synchronizer.match(
        vision,
        [],
    )

    assert pair is None

    print(
        "[PASS] Empty acoustic frame list handled correctly"
    )

    # ========================================================
    # Test 9
    # Boolean synchronization
    # ========================================================

    acoustic_good = create_acoustic_frame(
        frame_id=20,
        timestamp=100.050,
    )

    acoustic_bad = create_acoustic_frame(
        frame_id=21,
        timestamp=100.250,
    )

    assert synchronizer.is_synchronized(
        vision,
        acoustic_good,
    )

    assert not synchronizer.is_synchronized(
        vision,
        acoustic_bad,
    )

    print(
        "[PASS] Boolean synchronization check works"
    )

    # ========================================================
    # Test 10
    # Midpoint timestamp
    # ========================================================

    pair = synchronizer.synchronize_pair(
        vision,
        acoustic_good,
    )

    assert pair is not None

    expected_midpoint = (
        100.000
        + 100.050
    ) / 2.0

    assert abs(
        pair.timestamp - expected_midpoint
    ) < 1e-9

    print(
        "[PASS] Synchronized midpoint timestamp works"
    )

    # ========================================================
    # Final result
    # ========================================================

    print()
    print("=" * 72)
    print("MILESTONE 16 TEST PASSED")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()