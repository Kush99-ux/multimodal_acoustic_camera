"""
test_segmentation.py
====================

Milestone 13 integration test for the Multimodal Acoustic Camera.

Pipeline
--------
ESP32-CAM
    ↓
CameraAcquisition
    ↓
CameraFrame
    ↓
OpenCV frame
    ↓
YOLOSegmenter
    ↓
YOLO11 instance segmentation
    ↓
Structured Detection objects
    ↓
Debug visualization

This test does NOT:
- use the UMA-16 microphone array
- run MUSIC
- perform acoustic localization
- perform camera/audio synchronization
- perform multimodal fusion

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import time

import cv2

from camera import CameraAcquisition
from vision.segmentation import YOLOSegmenter


# ============================================================
# Configuration
# ============================================================

WINDOW_NAME = "Milestone 13 - YOLO Segmentation"

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720

EXIT_KEYS = {
    ord("q"),
    27,
}


# ============================================================
# Detection Reporting
# ============================================================

def print_detection(
    index: int,
    detection,
) -> None:
    """
    Print one structured detection.
    """

    print(
        f"  Detection {index:02d}"
    )

    print(
        f"    Class       : "
        f"{detection.class_name} "
        f"(ID {detection.class_id})"
    )

    print(
        f"    Confidence  : "
        f"{detection.confidence:.3f}"
    )

    x1, y1, x2, y2 = detection.bbox

    print(
        f"    Bounding box: "
        f"({x1:.1f}, {y1:.1f}) → "
        f"({x2:.1f}, {y2:.1f})"
    )

    cx, cy = detection.bbox_centroid

    print(
        f"    BBox center : "
        f"({cx:.1f}, {cy:.1f})"
    )

    if detection.mask_centroid is not None:

        mcx, mcy = detection.mask_centroid

        print(
            f"    Mask center : "
            f"({mcx:.1f}, {mcy:.1f})"
        )

    else:

        print(
            "    Mask center : unavailable"
        )

    if detection.mask is not None:

        height, width = detection.mask.shape

        print(
            f"    Mask        : "
            f"{width} × {height}"
        )

    else:

        print(
            "    Mask        : unavailable"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 72)
    print("YOLO11 SEGMENTATION INTEGRATION TEST")
    print("=" * 72)

    print()
    print("Pipeline:")
    print("  ESP32-CAM")
    print("      ↓")
    print("  CameraAcquisition")
    print("      ↓")
    print("  CameraFrame")
    print("      ↓")
    print("  OpenCV frame")
    print("      ↓")
    print("  YOLOSegmenter")
    print("      ↓")
    print("  Structured detections")

    print()
    print("This test does NOT use:")
    print("  - UMA-16 audio")
    print("  - MUSIC")
    print("  - Acoustic localization")
    print("  - Audio/video fusion")

    print()
    print("=" * 72)

    camera = None
    segmenter = None

    frames_acquired = 0
    inferences = 0

    total_inference_time = 0.0

    start_time = time.perf_counter()

    try:

        # ====================================================
        # Camera initialization
        # ====================================================

        print()
        print("[Camera] Initializing ESP32-CAM...")

        camera = CameraAcquisition()

        print(
            "[Camera] Starting stream..."
        )

        camera.start()

        print(
            "[Camera] Stream started successfully."
        )

        # ====================================================
        # YOLO initialization
        # ====================================================

        print()
        print(
            "[Vision] Initializing YOLO11 segmentation..."
        )

        segmenter = YOLOSegmenter()

        print(
            f"[Vision] {segmenter}"
        )

        # ====================================================
        # Model classes
        # ====================================================

        print()
        print(
            "[Vision] Available classes:"
        )

        for class_id, class_name in (
            segmenter.class_names().items()
        ):

            print(
                f"  {class_id:02d}: {class_name}"
            )

        # ====================================================
        # OpenCV window
        # ====================================================

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        )

        print()
        print("=" * 72)
        print("STARTING LIVE SEGMENTATION")
        print("=" * 72)

        print()
        print("Controls:")
        print("  Q / ESC : quit")

        print()

        # ====================================================
        # Main loop
        # ====================================================

        while True:

            # ------------------------------------------------
            # Acquire CameraFrame
            # ------------------------------------------------

            packet = camera.read()

            if packet is None:

                print(
                    "[Camera] Warning: received empty frame."
                )

                continue

            # ------------------------------------------------
            # Extract actual OpenCV image
            # ------------------------------------------------

            frame = packet.frame

            timestamp = packet.timestamp
            frame_id = packet.frame_id

            frames_acquired += 1

            # ------------------------------------------------
            # YOLO inference
            # ------------------------------------------------

            inference_start = time.perf_counter()

            results = segmenter.segment(
                frame
            )

            inference_time = (
                time.perf_counter()
                - inference_start
            )

            total_inference_time += inference_time

            inferences += 1

            # ------------------------------------------------
            # Structured detections
            # ------------------------------------------------

            detections = segmenter.get_detections(
                results,
                frame.shape,
            )

            # ------------------------------------------------
            # Terminal status
            # ------------------------------------------------

            print(
                f"\r[Frame {frame_id:05d}] "
                f"Detections: {len(detections):02d} "
                f"| Inference: "
                f"{inference_time * 1000:.1f} ms "
                f"| Timestamp: "
                f"{timestamp:.3f}",
                end="",
                flush=True,
            )

            # ------------------------------------------------
            # Print detection details when detections exist
            # ------------------------------------------------

            if detections:

                print()

                for index, detection in enumerate(
                    detections,
                    start=1,
                ):

                    print_detection(
                        index,
                        detection,
                    )

            # ------------------------------------------------
            # Create visualization
            # ------------------------------------------------

            annotated = segmenter.annotate(
                frame,
                results,
            )

            # ------------------------------------------------
            # Draw project-level centroids
            # ------------------------------------------------

            for detection in detections:

                # Bounding-box centroid
                cx, cy = (
                    detection.bbox_centroid
                )

                cv2.circle(
                    annotated,
                    (
                        int(round(cx)),
                        int(round(cy)),
                    ),
                    4,
                    (0, 255, 255),
                    -1,
                )

                # Segmentation-mask centroid
                if detection.mask_centroid is not None:

                    mcx, mcy = (
                        detection.mask_centroid
                    )

                    cv2.circle(
                        annotated,
                        (
                            int(round(mcx)),
                            int(round(mcy)),
                        ),
                        4,
                        (255, 255, 0),
                        -1,
                    )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                annotated,
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key in EXIT_KEYS:
                break

    except KeyboardInterrupt:

        print()
        print()
        print(
            "[Test] Interrupted by user."
        )

    except Exception as exc:

        print()
        print()
        print(
            "[ERROR] Segmentation test failed:"
        )

        print(
            f"  {type(exc).__name__}: {exc}"
        )

        raise

    finally:

        # ====================================================
        # Shutdown
        # ====================================================

        if camera is not None:

            print()
            print(
                "[Camera] Stopping camera..."
            )

            camera.stop()

        cv2.destroyAllWindows()

        # ====================================================
        # Statistics
        # ====================================================

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print()
        print("=" * 72)
        print("SEGMENTATION TEST SUMMARY")
        print("=" * 72)

        print()

        print(
            f"Frames acquired       : "
            f"{frames_acquired}"
        )

        print(
            f"Inferences performed  : "
            f"{inferences}"
        )

        print(
            f"Elapsed time          : "
            f"{elapsed:.2f} s"
        )

        if elapsed > 0:

            print(
                f"Camera acquisition FPS: "
                f"{frames_acquired / elapsed:.2f}"
            )

        if inferences > 0:

            print(
                f"Average inference FPS : "
                f"{inferences / elapsed:.2f}"
            )

            print(
                f"Average inference    : "
                f"{(total_inference_time / inferences) * 1000:.1f} ms"
            )

        print()
        print("=" * 72)
        print("MILESTONE 13 TEST COMPLETE")
        print("=" * 72)
        print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()