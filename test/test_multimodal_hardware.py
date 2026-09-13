"""
test_multimodal_hardware.py
===========================

M20-HW - Real hardware end-to-end validation.

This test uses the actual project hardware/software pipeline:

    ESP32-CAM
        ↓
    CameraAcquisition
        ↓
    YOLO11
        ↓
    VisionFrame
        │
        ├──────────────┐
        │              │
        │      TimestampSynchronizer
        │              │
    UMA-16             │
        ↓              │
    AudioFrame         │
        ↓              │
    MUSIC              │
        ↓              │
    AcousticFrame ─────┘
                       ↓
                 FusionEngine
                       ↓
                MultimodalResult

Unlike test_multimodal_pipeline.py, this test does NOT use
fake camera, audio, segmentation, or localization components.

It is intended to validate the real hardware/software integration.

This test does NOT evaluate:
- acoustic localization accuracy
- microphone geometry accuracy
- camera/acoustic spatial calibration
- object-to-sound association
- acoustic heatmap accuracy

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import time

from pipeline.multimodal import MultimodalPipeline


# ============================================================
# TEST CONFIGURATION
# ============================================================

TEST_ITERATIONS = 10


# ============================================================
# MAIN TEST
# ============================================================


def main() -> None:

    print()
    print("=" * 72)
    print("M20-HW - REAL HARDWARE END-TO-END VALIDATION")
    print("=" * 72)

    print()
    print("Pipeline:")
    print("  ESP32-CAM")
    print("      ↓")
    print("  CameraAcquisition")
    print("      ↓")
    print("  YOLO11")
    print("      ↓")
    print("  VisionFrame")
    print("      │")
    print("      ├──────────────┐")
    print("      │              │")
    print("      │      TimestampSynchronizer")
    print("      │              │")
    print("  UMA-16             │")
    print("      ↓              │")
    print("  AudioFrame         │")
    print("      ↓              │")
    print("  MUSIC              │")
    print("      ↓              │")
    print("  AcousticFrame ─────┘")
    print("                     ↓")
    print("               FusionEngine")
    print("                     ↓")
    print("              MultimodalResult")

    print()
    print("This test uses REAL hardware/software components.")
    print()
    print(f"[Configuration]")
    print(f"  Test iterations : {TEST_ITERATIONS}")
    print("  Sync tolerance  : 100 ms")

    pipeline = MultimodalPipeline()

    results = []

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Start real hardware
    # --------------------------------------------------------

    try:

        print()
        print("[System] Starting multimodal pipeline...")

        pipeline.start()

        print("[PASS] Multimodal pipeline started")

        # ----------------------------------------------------
        # Run real acquisition/fusion iterations
        # ----------------------------------------------------

        print()
        print("[Measurement] Running real-time iterations...")
        print()

        for iteration in range(1, TEST_ITERATIONS + 1):

            print("-" * 72)
            print(f"Iteration {iteration}/{TEST_ITERATIONS}")

            iteration_start = time.perf_counter()

            try:

                result = pipeline.run_once()

            except Exception as exc:

                print()
                print("[ERROR] Pipeline iteration failed")
                print(f"        {type(exc).__name__}: {exc}")

                raise

            iteration_elapsed = (
                time.perf_counter()
                - iteration_start
            )

            if result is None:

                print("  Result          : NO FUSION RESULT")

                print(
                    f"  Iteration time  : "
                    f"{iteration_elapsed * 1000.0:.1f} ms"
                )

                continue

            results.append(result)

            print("  Result          : FUSED")

            print(
                f"  Vision frame   : "
                f"{result.vision.frame_id}"
            )

            print(
                f"  Vision time    : "
                f"{result.vision.timestamp:.6f}"
            )

            print(
                f"  Detections     : "
                f"{result.vision_detection_count}"
            )

            print(
                f"  Acoustic frame : "
                f"{result.acoustic.frame_id}"
            )

            print(
                f"  Acoustic time  : "
                f"{result.acoustic.timestamp:.6f}"
            )

            print(
                f"  Time difference: "
                f"{result.timestamp_difference_ms:.3f} ms"
            )

            print(
                f"  Acoustic loc.  : "
                f"{result.acoustic_available}"
            )

            print(
                f"  Fusion conf.   : "
                f"{result.fusion_confidence:.3f}"
            )

            print(
                f"  Status         : "
                f"{result.status}"
            )

            print(
                f"  Iteration time : "
                f"{iteration_elapsed * 1000.0:.1f} ms"
            )

    finally:

        # ----------------------------------------------------
        # Always stop hardware
        # ----------------------------------------------------

        print()
        print("[System] Stopping multimodal pipeline...")

        pipeline.stop()

        print("[PASS] Multimodal pipeline stopped")

    elapsed = time.perf_counter() - start_time

    # ========================================================
    # VALIDATION
    # ========================================================

    statistics = pipeline.get_statistics()

    print()
    print("=" * 72)
    print("M20-HW VALIDATION")
    print("=" * 72)

    print()
    print("[Acquisition]")
    print(
        f"  Camera frames acquired   : "
        f"{statistics['camera_frames_acquired']}"
    )

    print(
        f"  Audio frames acquired    : "
        f"{statistics['audio_frames_acquired']}"
    )

    print(
        f"  Vision frames processed  : "
        f"{statistics['vision_frames_processed']}"
    )

    print(
        f"  Acoustic frames localized: "
        f"{statistics['acoustic_frames_localized']}"
    )

    print()
    print("[Synchronization]")
    print(
        f"  Synchronization attempts : "
        f"{statistics['synchronization_attempts']}"
    )

    print(
        f"  Successful pairs         : "
        f"{statistics['successful_synchronized_pairs']}"
    )

    print(
        f"  Rejected pairs           : "
        f"{statistics['rejected_unsynchronized_frames']}"
    )

    print()
    print("[Fusion]")
    print(
        f"  Fusion results           : "
        f"{statistics['fusion_results']}"
    )

    print()
    print("[Runtime]")
    print(
        f"  Total elapsed time       : "
        f"{elapsed:.3f} s"
    )

    # ========================================================
    # PASS / FAIL CRITERIA
    # ========================================================

    print()
    print("[Validation checks]")

    assert statistics["camera_frames_acquired"] > 0
    print("[PASS] Real camera frames acquired")

    assert statistics["audio_frames_acquired"] > 0
    print("[PASS] Real audio frames acquired")

    assert statistics["vision_frames_processed"] > 0
    print("[PASS] Real YOLO vision processing executed")

    assert statistics["acoustic_frames_localized"] > 0
    print("[PASS] Real MUSIC localization executed")

    assert statistics["synchronization_attempts"] > 0
    print("[PASS] Timestamp synchronization executed")

    assert statistics["fusion_results"] > 0
    print("[PASS] At least one real multimodal fusion result produced")

    # --------------------------------------------------------
    # Validate actual fused results
    # --------------------------------------------------------

    for result in results:

        assert result.synchronized
        assert result.vision is not None
        assert result.acoustic is not None

        assert (
            0.0
            <= result.fusion_confidence
            <= 1.0
        )

        assert (
            result.timestamp_difference_ms
            <= 100.0 + 1e-6
        )

    print("[PASS] Fused results satisfy synchronization tolerance")

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 72)
    print("M20-HW - REAL HARDWARE VALIDATION PASSED")
    print("=" * 72)

    print()
    print("Validated:")
    print()
    print("  ESP32-CAM")
    print("       ↓")
    print("  CameraAcquisition")
    print("       ↓")
    print("  YOLO11")
    print("       ↓")
    print("  VisionFrame")
    print("       ↓")
    print("  TimestampSynchronizer")
    print("       ↑")
    print("       │")
    print("  AcousticFrame")
    print("       ↑")
    print("       │")
    print("  MUSIC")
    print("       ↑")
    print("       │")
    print("  AudioFrame")
    print("       ↑")
    print("       │")
    print("     UMA-16")
    print("       ↓")
    print("  FusionEngine")
    print("       ↓")
    print("  MultimodalResult")

    print()
    print("NOTE:")
    print("This validates software/hardware integration.")
    print("It does NOT validate acoustic localization accuracy")
    print("or camera-acoustic spatial calibration.")
    print()


if __name__ == "__main__":
    main()