"""
test_fusion_models.py
====================

Milestone 17 - MultimodalResult data model test.

This test verifies the fusion-layer data contract.

It does NOT:
- connect to the ESP32-CAM
- load YOLO
- connect to the UMA-16
- run MUSIC
- perform live synchronization
- perform real acoustic/visual association

It only verifies the MultimodalResult interface.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import numpy as np

from acoustic.models import (
    AcousticFrame,
    AcousticLocalization,
)

from fusion.models import MultimodalResult

from vision.models import (
    VisionDetection,
    VisionFrame,
)


def main() -> None:

    print()
    print("=" * 72)
    print("MILESTONE 17 - MULTIMODAL RESULT DATA MODEL TEST")
    print("=" * 72)

    # ========================================================
    # Create synthetic vision data
    # ========================================================

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

    print()
    print("[PASS] VisionDetection created")
    print(f"       Class      : {detection.class_name}")
    print(f"       Confidence : {detection.confidence:.3f}")
    print(f"       BBox center: {detection.bbox_centroid}")

    # ========================================================
    # Create VisionFrame
    # ========================================================

    vision = VisionFrame(
        timestamp=12345.678,
        frame_id=42,
        detections=[
            detection,
        ],
        width=320,
        height=240,
        inference_time_ms=184.0,
    )

    print()
    print("[PASS] VisionFrame created")
    print(f"       Frame ID   : {vision.frame_id}")
    print(f"       Timestamp  : {vision.timestamp}")
    print(f"       Detections : {vision.detection_count}")

    # ========================================================
    # Create acoustic localization
    # ========================================================

    localization = AcousticLocalization(
        x=0.025,
        y=-0.010,
        z=0.070,
        response=0.92,
        frequency=1000.0,
        confidence=0.85,
    )

    print()
    print("[PASS] AcousticLocalization created")
    print(
        f"       Position   : "
        f"({localization.x:.3f}, "
        f"{localization.y:.3f}, "
        f"{localization.z:.3f}) m"
    )

    # ========================================================
    # Create AcousticFrame
    # ========================================================

    acoustic = AcousticFrame(
        timestamp=12345.708,
        frame_id=17,
        localization=localization,
        sample_rate=48000,
        frame_size=4096,
    )

    print()
    print("[PASS] AcousticFrame created")
    print(f"       Frame ID   : {acoustic.frame_id}")
    print(f"       Timestamp  : {acoustic.timestamp}")
    print(f"       Sample rate: {acoustic.sample_rate} Hz")
    print(f"       Frame size : {acoustic.frame_size}")

    # ========================================================
    # Create MultimodalResult
    # ========================================================

    result = MultimodalResult(
        timestamp=12345.693,
        vision=vision,
        acoustic=acoustic,
        timestamp_difference_ms=30.0,
        synchronized=True,
        fusion_confidence=0.85,
        status="synchronized",
    )

    print()
    print("[PASS] MultimodalResult created")

    # ========================================================
    # Validate core fields
    # ========================================================

    assert result.timestamp == 12345.693

    assert result.vision is vision

    assert result.acoustic is acoustic

    assert result.timestamp_difference_ms == 30.0

    assert result.synchronized is True

    assert result.fusion_confidence == 0.85

    assert result.status == "synchronized"

    print("[PASS] Core fusion fields validated")

    # ========================================================
    # Vision convenience property
    # ========================================================

    assert result.vision_detection_count == 1

    print(
        "[PASS] Vision detection count works"
    )

    # ========================================================
    # Acoustic availability
    # ========================================================

    assert result.acoustic_available is True

    print(
        "[PASS] Acoustic availability check works"
    )

    # ========================================================
    # Validity
    # ========================================================

    assert result.is_valid is True

    print(
        "[PASS] MultimodalResult validity check works"
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = result.summary()

    assert isinstance(summary, dict)

    assert summary["timestamp"] == 12345.693

    assert summary["timestamp_difference_ms"] == 30.0

    assert summary["synchronized"] is True

    assert summary["fusion_confidence"] == 0.85

    assert summary["status"] == "synchronized"

    assert summary["vision_frame_id"] == 42

    assert summary["vision_detection_count"] == 1

    assert summary["acoustic_frame_id"] == 17

    assert summary["acoustic_available"] is True

    print(
        "[PASS] Summary generation works"
    )

    # ========================================================
    # Metadata
    # ========================================================

    result.metadata["source"] = "synthetic_test"

    result.metadata["stage"] = "milestone_17"

    assert result.metadata["source"] == "synthetic_test"

    assert result.metadata["stage"] == "milestone_17"

    print(
        "[PASS] Metadata storage works"
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 72)
    print("MILESTONE 17 - STEP 1 PASSED")
    print("=" * 72)

    print()
    print("Fusion data contract established:")
    print("  VisionFrame")
    print("       +")
    print("  AcousticFrame")
    print("       ↓")
    print("  MultimodalResult")
    print()

    print("Next:")
    print("  Build FusionEngine")
    print()


if __name__ == "__main__":
    main()