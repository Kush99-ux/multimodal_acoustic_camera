"""
test_vision_models.py
=====================

Milestone 14 unit test for the project-level vision data models.

This test does NOT:
- connect to the ESP32-CAM
- load YOLO
- run inference
- use the UMA-16
- run MUSIC
- perform synchronization
- perform fusion

It only verifies the VisionDetection and VisionFrame interfaces.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import numpy as np

from vision.models import VisionDetection, VisionFrame


def main() -> None:

    print()
    print("=" * 72)
    print("MILESTONE 14 - VISION DATA MODEL TEST")
    print("=" * 72)

    # --------------------------------------------------------
    # Create a synthetic segmentation mask
    # --------------------------------------------------------

    mask = np.zeros(
        (240, 320),
        dtype=np.uint8,
    )

    mask[80:160, 120:200] = 1

    # --------------------------------------------------------
    # Create VisionDetection
    # --------------------------------------------------------

    detection = VisionDetection(
        class_id=67,
        class_name="cell phone",
        confidence=0.91,
        bbox=(
            110.0,
            70.0,
            210.0,
            170.0,
        ),
        bbox_centroid=(
            160.0,
            120.0,
        ),
        mask=mask,
        mask_centroid=(
            159.5,
            119.5,
        ),
    )

    # --------------------------------------------------------
    # Validate detection
    # --------------------------------------------------------

    assert detection.class_id == 67
    assert detection.class_name == "cell phone"

    assert 0.0 <= detection.confidence <= 1.0

    assert detection.bbox == (
        110.0,
        70.0,
        210.0,
        170.0,
    )

    assert detection.bbox_centroid == (
        160.0,
        120.0,
    )

    assert detection.mask is not None
    assert detection.mask.shape == (240, 320)

    assert detection.mask_centroid is not None

    print()
    print("[PASS] VisionDetection created successfully")
    print(f"       Class      : {detection.class_name}")
    print(f"       Confidence : {detection.confidence:.3f}")
    print(f"       BBox center: {detection.bbox_centroid}")
    print(f"       Mask shape : {detection.mask.shape}")
    print(f"       Mask center: {detection.mask_centroid}")

    # --------------------------------------------------------
    # Create VisionFrame
    # --------------------------------------------------------

    vision_frame = VisionFrame(
        timestamp=12345.678,
        frame_id=42,
        detections=[
            detection,
        ],
        width=320,
        height=240,
        inference_time_ms=184.0,
    )

    # --------------------------------------------------------
    # Validate frame
    # --------------------------------------------------------

    assert vision_frame.timestamp == 12345.678
    assert vision_frame.frame_id == 42

    assert vision_frame.width == 320
    assert vision_frame.height == 240

    assert vision_frame.detection_count == 1

    assert vision_frame.inference_time_ms == 184.0

    print()
    print("[PASS] VisionFrame created successfully")
    print(f"       Frame ID   : {vision_frame.frame_id}")
    print(f"       Timestamp  : {vision_frame.timestamp}")
    print(f"       Resolution : {vision_frame.width} × {vision_frame.height}")
    print(f"       Detections : {vision_frame.detection_count}")
    print(
        f"       Inference  : "
        f"{vision_frame.inference_time_ms:.1f} ms"
    )

    # --------------------------------------------------------
    # Test class filtering
    # --------------------------------------------------------

    phones = vision_frame.get_detections_by_class(
        "cell phone"
    )

    assert len(phones) == 1

    print()
    print("[PASS] Class filtering works")

    # --------------------------------------------------------
    # Test highest-confidence detection
    # --------------------------------------------------------

    highest = (
        vision_frame
        .get_highest_confidence_detection()
    )

    assert highest is detection

    print("[PASS] Highest-confidence lookup works")

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("MILESTONE 14 TEST PASSED")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()