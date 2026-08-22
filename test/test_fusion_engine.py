"""
test_fusion_engine.py
=====================

Milestone 17 - FusionEngine integration test.

This test validates the first multimodal fusion pipeline
using synthetic VisionFrame and AcousticFrame objects.

No live hardware is required.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import numpy as np

from acoustic.models import (
    AcousticFrame,
    AcousticLocalization,
)

from fusion.engine import FusionEngine

from vision.models import (
    VisionDetection,
    VisionFrame,
)


# ============================================================
# Test data factories
# ============================================================


def create_vision_frame(
    frame_id: int,
    timestamp: float,
) -> VisionFrame:

    mask = np.zeros(
        (240, 320),
        dtype=np.uint8,
    )

    mask[80:160, 120:200] = 1

    detection = VisionDetection(
        class_id=67,
        class_name="cell phone",
        confidence=0.91,
        bbox=(120.0, 80.0, 200.0, 160.0),
        bbox_centroid=(160.0, 120.0),
        mask=mask,
    )

    return VisionFrame(
        timestamp=timestamp,
        frame_id=frame_id,
        detections=[detection],
        width=320,
        height=240,
        inference_time_ms=184.0,
    )


def create_acoustic_frame(
    frame_id: int,
    timestamp: float,
) -> AcousticFrame:

    localization = AcousticLocalization(
        x=0.025,
        y=-0.010,
        z=0.070,
        response=0.92,
        frequency=1000.0,
        confidence=0.85,
    )

    return AcousticFrame(
        timestamp=timestamp,
        frame_id=frame_id,
        localization=localization,
        sample_rate=48000,
        frame_size=4096,
    )


# ============================================================
# Main
# ============================================================


def main() -> None:

    print()
    print("=" * 72)
    print("MILESTONE 17 - FUSION ENGINE TEST")
    print("=" * 72)

    # ========================================================
    # Initialize
    # ========================================================

    engine = FusionEngine()

    print()
    print("[PASS] FusionEngine initialized")

    # ========================================================
    # Test 1: Successful fusion
    # ========================================================

    vision = create_vision_frame(
        frame_id=1,
        timestamp=100.000,
    )

    acoustic = create_acoustic_frame(
        frame_id=1,
        timestamp=100.030,
    )

    result = engine.fuse(
        vision,
        acoustic,
    )

    assert result is not None

    print(
        "[PASS] Synchronized frames fused successfully"
    )

    # ========================================================
    # Validate result
    # ========================================================

    assert result.synchronized is True

    assert result.vision.frame_id == 1

    assert result.acoustic.frame_id == 1

    assert (
        abs(
            result.timestamp_difference_ms
            - 30.0
        )
        < 1e-6
    )

    assert result.status == "synchronized"

    print(
        "[PASS] Synchronization information preserved"
    )

    # ========================================================
    # Validate vision
    # ========================================================

    assert result.vision_detection_count == 1

    assert (
        result.vision
        .detections[0]
        .class_name
        == "cell phone"
    )

    print(
        "[PASS] Vision information preserved"
    )

    # ========================================================
    # Validate acoustic
    # ========================================================

    assert result.acoustic_available is True

    assert (
        result.acoustic.localization.x
        == 0.025
    )

    assert (
        result.acoustic.localization.frequency
        == 1000.0
    )

    print(
        "[PASS] Acoustic information preserved"
    )

    # ========================================================
    # Validate confidence
    # ========================================================

    assert (
        0.0
        <= result.fusion_confidence
        <= 1.0
    )

    print(
        "[PASS] Fusion confidence is valid"
    )

    # ========================================================
    # Test 2: Unsynchronized frames
    # ========================================================

    vision_bad = create_vision_frame(
        frame_id=2,
        timestamp=200.000,
    )

    acoustic_bad = create_acoustic_frame(
        frame_id=2,
        timestamp=200.250,
    )

    rejected = engine.fuse(
        vision_bad,
        acoustic_bad,
    )

    assert rejected is None

    print(
        "[PASS] Unsynchronized frames correctly rejected"
    )

    # ========================================================
    # Test 3: Exact tolerance
    # ========================================================

    vision_boundary = create_vision_frame(
        frame_id=3,
        timestamp=300.000,
    )

    acoustic_boundary = create_acoustic_frame(
        frame_id=3,
        timestamp=300.100,
    )

    boundary_result = engine.fuse(
        vision_boundary,
        acoustic_boundary,
    )

    assert boundary_result is not None

    print(
        "[PASS] 100 ms boundary correctly accepted"
    )

    # ========================================================
    # Statistics
    # ========================================================

    statistics = engine.get_statistics()

    assert statistics["processed"] == 3

    assert statistics["successful"] == 2

    assert statistics["rejected"] == 1

    print()
    print("[PASS] Fusion statistics correct")

    print(
        f"       Processed : "
        f"{statistics['processed']}"
    )

    print(
        f"       Successful: "
        f"{statistics['successful']}"
    )

    print(
        f"       Rejected  : "
        f"{statistics['rejected']}"
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 72)
    print("MILESTONE 17 - FUSION ENGINE TEST PASSED")
    print("=" * 72)

    print()
    print("Current multimodal pipeline:")
    print()
    print("  VisionFrame")
    print("       +")
    print("  AcousticFrame")
    print("       ↓")
    print("  TimestampSynchronizer")
    print("       ↓")
    print("  FusionEngine")
    print("       ↓")
    print("  MultimodalResult")
    print()


if __name__ == "__main__":
    main()