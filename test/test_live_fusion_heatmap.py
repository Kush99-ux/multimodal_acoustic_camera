"""
test_live_fusion_heatmap.py
===========================

Live real-hardware acoustic heatmap overlay test.

Stage D:
    MUSIC
      ↓
    azimuth / elevation
      ↓
    camera FOV
      ↓
    pixel projection
      ↓
    live acoustic heatmap

The projection is angular only. It is NOT physical
camera/acoustic calibration.
"""

from __future__ import annotations

import cv2

from pipeline.multimodal import MultimodalPipeline
from fusion.visualization import AcousticHeatmapVisualizer


def main() -> None:
    print()
    print("=" * 72)
    print("M21 - LIVE FUSION HEATMAP PROTOTYPE")
    print("=" * 72)

    pipeline = MultimodalPipeline()

    visualizer = AcousticHeatmapVisualizer(
        heatmap_radius=45,
        alpha=0.45,
    )

    pipeline.start()

    print()
    print("[PASS] Pipeline started")
    print()
    print("Controls:")
    print("  Q / ESC : quit")
    print()

    try:
        while True:

            # Audio
            audio_frame = pipeline.audio.read(
                timeout=pipeline.audio_read_timeout
            )

            if audio_frame is not None:
                pipeline._statistics[
                    "audio_frames_acquired"
                ] += 1

            # Camera
            camera_frame = pipeline.camera.read()

            if camera_frame is None:
                continue

            pipeline._statistics[
                "camera_frames_acquired"
            ] += 1

            # Vision
            vision_frame = pipeline.build_vision_frame(
                camera_frame
            )

            # Acoustic
            if audio_frame is not None:
                pipeline.ingest_audio_frame(
                    audio_frame
                )

            # Fusion
            result = pipeline.process_vision_frame(
                vision_frame
            )

            display = camera_frame.frame.copy()

            # Stage-D angular heatmap projection
            if result is not None:
                display = visualizer.render(
                    image=display,
                    result=result,
                )

                localization = result.acoustic.localization

                if localization is not None:

                    # ---------------------------------------------
                    # Extract true angles for display
                    # ---------------------------------------------
                    azimuth = float(result.acoustic.metadata.get("azimuth_deg", 90.0))
                    elevation = float(result.acoustic.metadata.get("elevation_deg", 0.0))

                    cv2.putText(
                        display,
                        f"az={azimuth:+.1f} deg  el={elevation:+.1f} deg",
                        (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        display,
                        (
                            f"sync="
                            f"{result.timestamp_difference_ms:.1f} ms"
                        ),
                        (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            else:
                cv2.putText(
                    display,
                    "NO SYNCHRONIZED ACOUSTIC FRAME",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(
                "Multimodal Acoustic Camera - M21",
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

    except KeyboardInterrupt:
        print()
        print("[Test] Interrupted by user")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        print()
        print("[PASS] Pipeline stopped")
        print()
        print("=" * 72)
        print("M21 LIVE HEATMAP TEST COMPLETE")
        print("=" * 72)
        print()


if __name__ == "__main__":
    main()