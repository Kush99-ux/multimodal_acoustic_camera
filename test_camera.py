"""
test_camera.py
==============

Milestone 12
ESP32-CAM acquisition validation.

This test verifies:

    ESP32-CAM
        ↓
    MJPEG stream
        ↓
    CameraAcquisition
        ↓
    OpenCV frames

No YOLO.
No audio.
No fusion.
"""

import cv2

from camera import CameraAcquisition


def main() -> None:

    print("=" * 70)
    print("ESP32-CAM ACQUISITION TEST")
    print("=" * 70)

    camera = CameraAcquisition()

    try:

        # -----------------------------------------------------
        # Start camera
        # -----------------------------------------------------

        camera.start()

        resolution = camera.get_resolution()

        print()
        print("[Camera]")
        print(f"Model      : {camera.camera_model}")
        print(f"Stream     : {camera.stream_url}")
        print(f"Resolution : {resolution}")
        print()

        print("Starting live acquisition.")
        print("Press Q to quit.")
        print()

        # -----------------------------------------------------
        # Acquisition loop
        # -----------------------------------------------------

        while True:

            packet = camera.read()

            if packet is None:
                continue

            frame = packet.frame

            # Display basic information on the frame
            cv2.putText(
                frame,
                f"Frame: {packet.frame_id}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {camera.get_fps():.1f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "ESP32-CAM Acquisition",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        camera.stop()
        cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("CAMERA TEST COMPLETE")
    print("=" * 70)
    print()
    print(f"Frames acquired : {camera.get_frame_count()}")
    print(f"Average FPS     : {camera.get_fps():.2f}")


if __name__ == "__main__":
    main()