"""
live_camera_positioning.py
==========================
Standalone live positioning tool for ESP32-CAM.
Draws horizontal regions (LEFT, CENTER, RIGHT), center line (x=320),
and allows clicking on target objects to inspect image coordinates.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera (Diagnostic Utility)
"""

import sys
import time
import threading
import cv2
import numpy as np

STREAM_URL = "http://10.243.98.165:81/stream"
WINDOW_NAME = "ESP32-CAM LIVE POSITIONING"

# Global state for interactive mouse clicking
target_x = None
target_y = None
target_region = None
lock = threading.Lock()


def get_region(x: int, width: int = 640) -> str:
    """Determine whether pixel X falls in LEFT, CENTER, or RIGHT third."""
    third = width / 3.0
    if x < third:
        return "LEFT"
    elif x <= 2 * third:
        return "CENTER"
    else:
        return "RIGHT"


def on_mouse(event, x, y, flags, param):
    """OpenCV mouse callback for interactive sound source targeting."""
    global target_x, target_y, target_region
    if event == cv2.EVENT_LBUTTONDOWN:
        with lock:
            target_x = x
            target_y = y
            target_region = get_region(x, 640)

        print("\nClicked target:", flush=True)
        print(f"x = {x}", flush=True)
        print(f"y = {y}", flush=True)
        print(f"Region = {target_region}", flush=True)
        print(f"Offset from Center (x=320): {x - 320:+d} px", flush=True)


class FreshFrameReader:
    """Threaded camera reader ensuring 0-latency live display."""
    def __init__(self, url: str):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.latest_frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        if not self.cap.isOpened():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


def main():
    global target_x, target_y, target_region

    print("=" * 60, flush=True)
    print("ESP32-CAM LIVE POSITIONING", flush=True)
    print("=" * 60, flush=True)
    print("Stream:", flush=True)
    print(f"{STREAM_URL}", flush=True)
    print(flush=True)
    print("Resolution:", flush=True)
    print("640 x 480", flush=True)
    print(flush=True)
    print("Image center:", flush=True)
    print("x = 320", flush=True)
    print(flush=True)
    print("Instructions:", flush=True)
    print("LEFT CLICK = mark source", flush=True)
    print("C = clear marker", flush=True)
    print("Q = quit", flush=True)
    print("=" * 60, flush=True)

    reader = FreshFrameReader(STREAM_URL)
    if not reader.start():
        print(f"\nERROR: Failed to connect to camera stream at {STREAM_URL}", flush=True)
        return 1

    # Wait for first frame
    print("\nAwaiting stream connection...", flush=True)
    frame = None
    for _ in range(50):
        ret, frame = reader.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)

    if frame is None:
        print("ERROR: Stream opened but received no valid frames.", flush=True)
        reader.stop()
        return 1

    h, w = frame.shape[:2]
    print(f"Connected! Stream resolution: {w} x {h}", flush=True)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    third = w / 3.0
    x_third1 = int(round(third))         # ~213 for 640
    x_third2 = int(round(2.0 * third))   # ~426 for 640
    x_center = w // 2                    # 320 for 640

    fps_counter = 0
    fps_start = time.perf_counter()
    current_fps = 0.0

    try:
        while True:
            ret, frame = reader.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            fps_counter += 1
            now = time.perf_counter()
            if now - fps_start >= 1.0:
                current_fps = fps_counter / (now - fps_start)
                fps_counter = 0
                fps_start = now

            display = frame.copy()

            # -------------------------------------------------------------
            # 1. Draw Region Dividers (Thirds)
            # -------------------------------------------------------------
            # Vertical lines at thirds
            cv2.line(display, (x_third1, 0), (x_third1, h), (180, 180, 180), 1, cv2.LINE_AA)
            cv2.line(display, (x_third2, 0), (x_third2, h), (180, 180, 180), 1, cv2.LINE_AA)

            # -------------------------------------------------------------
            # 2. Draw Center Line (x = 320)
            # -------------------------------------------------------------
            cv2.line(display, (x_center, 0), (x_center, h), (0, 255, 0), 2, cv2.LINE_AA)

            # Center ticks
            cv2.line(display, (x_center - 15, h // 2), (x_center + 15, h // 2), (0, 255, 0), 2, cv2.LINE_AA)

            # -------------------------------------------------------------
            # 3. Draw Region Headers
            # -------------------------------------------------------------
            header_bar = display[0:40, 0:w].copy()
            cv2.rectangle(display, (0, 0), (w, 40), (20, 20, 20), -1)
            display[0:40, 0:w] = cv2.addWeighted(display[0:40, 0:w], 0.6, header_bar, 0.4, 0)

            # Region Text
            cv2.putText(display, "LEFT", (int(x_third1 / 2) - 25, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "CENTER", (x_center - 40, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display, "RIGHT", (int(x_third2 + (w - x_third2) / 2) - 30, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)

            # -------------------------------------------------------------
            # 4. Draw HUD Info (Bottom Bar)
            # -------------------------------------------------------------
            cv2.putText(display, f"Camera: {w}x{h} ({current_fps:.1f} FPS) | Center X: {x_center}",
                        (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display, "LEFT CLICK = mark source | C = clear | Q = quit",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

            # -------------------------------------------------------------
            # 5. Draw Marked Target (if clicked)
            # -------------------------------------------------------------
            with lock:
                curr_tx = target_x
                curr_ty = target_y
                curr_reg = target_region

            if curr_tx is not None and curr_ty is not None:
                marker_color = (0, 255, 0) if curr_reg == "CENTER" else (0, 165, 255)

                # Crosshair
                cv2.drawMarker(display, (curr_tx, curr_ty), marker_color,
                               cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
                cv2.circle(display, (curr_tx, curr_ty), 14, marker_color, 2, cv2.LINE_AA)

                # Target Info Tag
                tag = f"Target: ({curr_tx}, {curr_ty}) -> {curr_reg}"
                cv2.rectangle(display, (curr_tx + 12, curr_ty - 22), (curr_tx + 12 + len(tag) * 9, curr_ty + 2), (0, 0, 0), -1)
                cv2.putText(display, tag, (curr_tx + 14, curr_ty - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, marker_color, 1, cv2.LINE_AA)

                # Also show target prominently on bottom-right of screen
                cv2.putText(display, f"TARGET: X={curr_tx} Y={curr_ty} [{curr_reg}]",
                            (w - 310, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, marker_color, 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                print("\nQuitting...")
                break
            elif key in (ord('c'), ord('C')):
                with lock:
                    target_x = None
                    target_y = None
                    target_region = None
                print("\n[Marker Cleared]")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("Camera reader stopped and window closed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
