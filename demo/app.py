"""
demo.app
========
Faculty Presentation Application for the Real-Time Multimodal Acoustic Camera.

Launch with:
    python demo/app.py
or for offline demonstration:
    python demo/app.py --simulate

Features:
- Full 1600x900 Academic Dark UI presentation dashboard
- 10 Interactive Modes:
    1: Live Multimodal Acoustic-Vision Fusion with Ground Truth Target click
    2: 2D MUSIC Spatial Spectrum (7,676 points, 101x76 grid)
    3: YOLO11n-seg Instance Segmentation & Confidence Tracking
    4: Guided Horizontal Localization Test (Left / Center / Right with 20-frame collection)
    5: Guided Vertical Elevation Localization Test (Above / Level / Below)
    6: Cross-Modal Timestamp Synchronization & Hardware Jitter Monitor
    7: 16-Microphone Physical Array Layout & 6 Sub-Band Spectral Power Meters
    8: Scientific Validation Benchmark Tables & Historical Verification Runs
    9: Multimodal System Architecture Flow Diagram
   10: Current Empirical Limitations & Research Roadmap
- Persistent live telemetry and status sidebar
- Snapshot capture [S] and continuous session telemetry recording [R]
"""

from __future__ import annotations
import argparse
from datetime import datetime
import os
import sys
import time
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np

# Enable Per-Monitor DPI Awareness on Windows to prevent DWM bitmap scaling / blur
if sys.platform == "win32":
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-Monitor DPI aware
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()       # System DPI aware fallback
    except Exception:
        pass

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.settings import STREAM_URL
from demo.core.runner import DemoPipelineRunner
from demo.data import ExperimentDatabase, ExperimentRepository, migrate_historical_records
from demo.core.experiment_manager import ExperimentManager
from demo.panels import (
    TelemetrySidebar,
    MultimodalPanel,
    AcousticPanel,
    VisionPanel,
    HorizontalTestPanel,
    VerticalTestPanel,
    SynchronizationPanel,
    DiagnosticsPanel,
    ValidationPanel,
    ArchitecturePanel,
    RoadmapPanel,
)
from demo.ui.theme import (
    CANVAS_WIDTH, CANVAS_HEIGHT, HEADER_HEIGHT, FOOTER_HEIGHT,
    FONT, FONT_TITLE, FONT_HEADER, FONT_BODY, AA,
    SCALE_TITLE, SCALE_SUBTITLE, SCALE_TAB,
    BG_CANVAS, BG_PANEL, BG_CARD, BG_HOVER, BORDER, BORDER_ACTIVE,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ACCENT,
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED, ACCENT_BLUE, ACCENT_ORANGE
)
from demo.ui.widgets import draw_card, draw_badge


TAB_LABELS = [
    "[1] Live Fusion",
    "[2] 2D MUSIC",
    "[3] YOLO Vision",
    "[4] Azimuth Test",
    "[5] Elevation Test",
    "[6] Sync Monitor",
    "[7] Array & FFT",
    "[8] Benchmarks",
    "[9] Architecture",
    "[0] Roadmap",
]


class ProfessorDemoApp:
    """Main Application Controller for the Multimodal Acoustic Camera Faculty Demo."""

    def __init__(
        self,
        stream_url: str = STREAM_URL,
        simulate: bool = False,
        output_dir: str = "scratch/demo_results",
        headless_frames: int = 0,
        audio_device: Optional[int | str] = None,
    ) -> None:
        self.stream_url = stream_url
        self.simulate = simulate
        self.output_dir = os.path.abspath(output_dir)
        self.headless_frames = headless_frames
        self.audio_device = audio_device
        os.makedirs(self.output_dir, exist_ok=True)

        self.current_mode = 1  # 1..10
        self.notification_msg = "System Ready. Press 1-0 to switch modes, S for snapshot, R to record session."
        self.notification_time = time.time()

        # Canvas layout geometry
        self.main_x = 10
        self.main_y = 68
        self.main_w = 1020
        self.main_h = 788

        self.side_x = 1040
        self.side_y = 68
        self.side_w = 550
        self.side_h = 788

        # Tab button geometry: 10 tabs across header
        self.tab_w = 152
        self.tab_h = 28
        self.tab_y = 34
        self.tab_spacing = 6
        self.tab_start_x = 12

        # Initialize SQLite database, repository, and experiment manager
        print("[App] Initializing SQLite Experiment Database & Repository...")
        self.db = ExperimentDatabase()
        self.repo = ExperimentRepository(self.db)
        migrate_historical_records(self.repo)

        self.session_id = f"SES-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        try:
            self.repo.start_session(
                session_id=self.session_id,
                mode="SIMULATION" if self.simulate else "HARDWARE",
                notes="Faculty presentation demo session",
            )
        except Exception as e:
            print(f"[App] Failed to record session start in DB: {e}")

        self.experiment_manager = ExperimentManager(self.repo, session_id=self.session_id)

        # Initialize pipeline runner
        print("[App] Initializing Demo Pipeline Runner...")
        self.runner = DemoPipelineRunner(
            stream_url=self.stream_url,
            simulate=self.simulate,
            log_dir=self.output_dir,
            audio_device=self.audio_device,
        )


        # Initialize persistent sidebar
        self.sidebar = TelemetrySidebar(
            x=self.side_x,
            y=self.side_y,
            w=self.side_w,
            h=self.side_h,
        )

        # Initialize all 10 mode panels
        self.panel_multimodal = MultimodalPanel(x=self.main_x, y=self.main_y, w=self.main_w, h=self.main_h)
        self.panel_acoustic = AcousticPanel(x=self.main_x, y=self.main_y, w=self.main_w, h=self.main_h)
        self.panel_vision = VisionPanel(x=self.main_x, y=self.main_y, w=self.main_w, h=self.main_h)
        self.panel_horizontal = HorizontalTestPanel(
            x=self.main_x,
            y=self.main_y,
            w=self.main_w,
            h=self.main_h,
            output_dir=self.output_dir,
            experiment_manager=self.experiment_manager,
        )
        self.panel_vertical = VerticalTestPanel(
            x=self.main_x,
            y=self.main_y,
            w=self.main_w,
            h=self.main_h,
            output_dir=self.output_dir,
            experiment_manager=self.experiment_manager,
        )
        self.panel_sync = SynchronizationPanel(x=self.main_x, y=self.main_y, w=self.main_w, h=self.main_h)
        self.panel_diagnostics = DiagnosticsPanel(x=self.main_x, y=self.main_y, w=self.main_w, h=self.main_h)
        self.panel_validation = ValidationPanel(
            x=self.main_x,
            y=self.main_y,
            w=self.main_w,
            h=self.main_h,
            repository=self.repo,
        )
        self.panel_architecture = ArchitecturePanel(x=self.main_x, y=self.main_y, w=self.main_w, h=self.main_h)
        self.panel_roadmap = RoadmapPanel(x=self.main_x, y=self.main_y, w=self.main_w, h=self.main_h)

        # FPS calculation
        self.last_time = time.perf_counter()
        self.fps = 0.0

        # Dynamic display scaling & letterboxing tracking (for aspect-ratio preservation)
        self.display_scale = 1.0
        self.display_pad_x = 0
        self.display_pad_y = 0

    def set_notification(self, text: str) -> None:
        """Display an ephemeral notification in the footer status bar."""
        self.notification_msg = text
        self.notification_time = time.time()
        print(f"[App Notification] {text}")

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handle mouse clicks on navigation tabs and interactive panels."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Map mouse coordinates from window client space back to native 1600x900 canvas space
        if self.display_scale != 1.0 or self.display_pad_x != 0 or self.display_pad_y != 0:
            if self.display_scale > 0:
                x = int(round((x - self.display_pad_x) / self.display_scale))
                y = int(round((y - self.display_pad_y) / self.display_scale))
            if x < 0 or x >= CANVAS_WIDTH or y < 0 or y >= CANVAS_HEIGHT:
                return  # Clicked inside letterbox padding

        # Check navigation tabs (Header row 2: y in [34, 62])
        if self.tab_y <= y <= self.tab_y + self.tab_h:
            for i in range(10):
                tx = self.tab_start_x + i * (self.tab_w + self.tab_spacing)
                if tx <= x <= tx + self.tab_w:
                    new_mode = 10 if i == 9 else i + 1
                    if new_mode != self.current_mode:
                        self.current_mode = new_mode
                        self.set_notification(f"Switched to Mode {self.current_mode}: {TAB_LABELS[i]}")
                    return

        # Handle Mode 1: Multimodal ground truth click & YOLO toggle
        if self.current_mode == 1:
            if self.panel_multimodal.handle_yolo_click(x, y):
                new_state = self.runner.toggle_yolo()
                status_str = "ENABLED" if new_state else "DISABLED (Inference Skipped)"
                self.set_notification(f"YOLO Mode switched: {status_str}")
                return

            cam_coords = self.panel_multimodal.handle_click(x, y)
            if cam_coords is not None:
                cx, cy = cam_coords
                self.runner.set_ground_truth(cx, cy)
                self.set_notification(f"Ground Truth Target set at camera coordinate ({cx}, {cy}) px.")
                return

        # Handle Mode 4: Horizontal guided test
        if self.current_mode == 4:
            clicked = self.panel_horizontal.handle_click(x, y, gt_click=self.runner.gt_click)
            if clicked:
                pos = self.panel_horizontal.selected_pos
                running = self.panel_horizontal.test_running
                status_txt = "RUNNING" if running else "READY"
                self.set_notification(f"Horizontal Test: Position [{pos}] ({status_txt})")
                return

        # Handle Mode 5: Vertical guided test
        if self.current_mode == 5:
            clicked = self.panel_vertical.handle_click(x, y, gt_click=self.runner.gt_click)
            if clicked:
                pos = self.panel_vertical.selected_pos
                running = self.panel_vertical.test_running
                status_txt = "RUNNING" if running else "READY"
                self.set_notification(f"Vertical Test: Position [{pos}] ({status_txt})")
                return

    def _render_header(self, canvas: np.ndarray, telemetry: dict) -> None:
        """Render application title bar, status indicators, and mode selection tabs."""
        # Top banner background
        cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, HEADER_HEIGHT), BG_PANEL, -1)
        cv2.line(canvas, (0, HEADER_HEIGHT), (CANVAS_WIDTH, HEADER_HEIGHT), BORDER, 1)

        # Title & subtitle (large readable title with thickness 2, simplex font)
        cv2.putText(
            canvas,
            "REAL-TIME MULTIMODAL ACOUSTIC CAMERA",
            (14, 25),
            FONT_TITLE,
            SCALE_TITLE,
            TEXT_PRIMARY,
            2,
            AA,
        )
        cv2.putText(
            canvas,
            "|  Faculty Presentation & Validation Suite",
            (575, 24),
            FONT_HEADER,
            SCALE_SUBTITLE,
            TEXT_ACCENT,
            1,
            AA,
        )

        # Hardware & loop status indicators (top-right)
        status_text = "SIMULATION" if self.simulate else ("ONLINE" if telemetry.get("camera_connected") else "HARDWARE PENDING")
        draw_badge(canvas, 1220, 5, status_text, status_ok=(not self.simulate and telemetry.get("camera_connected", False)))

        fps_text = f"{self.fps:4.1f} FPS"
        cv2.putText(canvas, fps_text, (1395, 22), FONT_HEADER, 0.48, TEXT_SECONDARY, 1, AA)

        rec_status = "[REC]" if self.runner.logger.recording_session else "IDLE"
        rec_color = ACCENT_RED if self.runner.logger.recording_session else TEXT_MUTED
        cv2.putText(canvas, rec_status, (1495, 22), FONT_HEADER, 0.48, rec_color, 1, AA)

        # 10 Navigation Tab Buttons with centered, clean typography
        for i, label in enumerate(TAB_LABELS):
            tx = self.tab_start_x + i * (self.tab_w + self.tab_spacing)
            tab_mode = 10 if i == 9 else i + 1
            is_active = (tab_mode == self.current_mode)

            # Tab button appearance
            bg_col = BG_CARD if is_active else BG_PANEL
            border_col = ACCENT_CYAN if is_active else BORDER
            text_col = TEXT_PRIMARY if is_active else TEXT_SECONDARY

            cv2.rectangle(canvas, (tx, self.tab_y), (tx + self.tab_w, self.tab_y + self.tab_h), bg_col, -1)
            cv2.rectangle(canvas, (tx, self.tab_y), (tx + self.tab_w, self.tab_y + self.tab_h), border_col, 2 if is_active else 1)

            if is_active:
                cv2.rectangle(canvas, (tx, self.tab_y + self.tab_h - 3), (tx + self.tab_w, self.tab_y + self.tab_h), ACCENT_CYAN, -1)

            (tw, th), _ = cv2.getTextSize(label, FONT_BODY, SCALE_TAB, 1)
            tx_cen = tx + (self.tab_w - tw) // 2
            ty_cen = self.tab_y + (self.tab_h + th) // 2 - 1
            cv2.putText(canvas, label, (tx_cen, ty_cen), FONT_BODY, SCALE_TAB, text_col, 1, AA)

    def _render_footer(self, canvas: np.ndarray) -> None:
        """Render keyboard shortcuts and live notification ticker."""
        fy = CANVAS_HEIGHT - FOOTER_HEIGHT
        cv2.rectangle(canvas, (0, fy), (CANVAS_WIDTH, CANVAS_HEIGHT), BG_PANEL, -1)
        cv2.line(canvas, (0, fy), (CANVAS_WIDTH, fy), BORDER, 1)

        # Hotkeys guide (left)
        guide = "[1-0]: Modes  |  [Y]: YOLO ON/OFF  |  [S]: Snapshot  |  [E]: Export  |  [R]: Rec  |  [C]: Clear GT  |  [Q]: Quit"
        cv2.putText(canvas, guide, (16, fy + 27), FONT_BODY, 0.44, TEXT_SECONDARY, 1, AA)

        # Dynamic notification message (right)
        if time.time() - self.notification_time < 8.0:
            cv2.putText(
                canvas,
                self.notification_msg,
                (710, fy + 27),
                FONT_BODY,
                0.46,
                ACCENT_CYAN,
                1,
                AA,
            )

    def compose_frame(self, telemetry: dict) -> np.ndarray:
        """Compose the complete 1600x900 GUI canvas for the current active mode."""
        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), BG_CANVAS, dtype=np.uint8)

        # 1. Render top header
        self._render_header(canvas, telemetry)

        # 2. Render active mode panel
        if self.current_mode == 1:
            self.panel_multimodal.render(canvas, self.runner.latest_overlaid_frame, telemetry)
        elif self.current_mode == 2:
            self.panel_acoustic.render(canvas, self.runner.latest_heatmap, telemetry)
        elif self.current_mode == 3:
            self.panel_vision.render(canvas, self.runner.latest_camera_frame, telemetry)
        elif self.current_mode == 4:
            self.panel_horizontal.render(canvas, telemetry)
        elif self.current_mode == 5:
            self.panel_vertical.render(canvas, telemetry)
        elif self.current_mode == 6:
            self.panel_sync.render(canvas, telemetry)
        elif self.current_mode == 7:
            self.panel_diagnostics.render(canvas, telemetry)
        elif self.current_mode == 8:
            self.panel_validation.render(canvas, telemetry)
        elif self.current_mode == 9:
            self.panel_architecture.render(canvas, telemetry)
        elif self.current_mode == 10:
            self.panel_roadmap.render(canvas, telemetry)

        # 3. Render persistent telemetry sidebar on the right
        self.sidebar.render(
            canvas,
            telemetry,
            recording_session=self.runner.logger.recording_session,
        )

        # 4. Render footer
        self._render_footer(canvas)

        return canvas

    def run(self) -> None:
        """Main application display loop."""
        window_name = "Real-Time Multimodal Acoustic Camera | Faculty Presentation Demo"

        if not self.headless_frames:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, CANVAS_WIDTH, CANVAS_HEIGHT)
            cv2.setMouseCallback(window_name, self.on_mouse)

            # Center window on screen if display dimensions allow
            if sys.platform == "win32":
                try:
                    import ctypes
                    from ctypes import wintypes
                    user32 = ctypes.windll.user32
                    rect_w = wintypes.RECT()
                    user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect_w), 0)  # SPI_GETWORKAREA
                    work_w = rect_w.right - rect_w.left
                    work_h = rect_w.bottom - rect_w.top
                    pos_x = max(0, (work_w - CANVAS_WIDTH) // 2)
                    pos_y = max(0, (work_h - CANVAS_HEIGHT) // 2)
                    cv2.moveWindow(window_name, pos_x, pos_y)
                except Exception:
                    pass

        frame_count = 0
        print("\n========================================================")
        print("  REAL-TIME MULTIMODAL ACOUSTIC CAMERA: PROFESSOR DEMO")
        print("========================================================")
        print("Controls:")
        print("  [1-0]       : Switch between presentation modes 1 to 10")
        print("  [S]         : Capture high-resolution snapshot & JSON telemetry")
        print("  [E]         : Export experiment database to CSV & JSON")
        print("  [R]         : Toggle continuous session recording to scratch/")
        print("  [C]         : Clear ground-truth target point")
        print("  [Q] / [ESC] : Clean shutdown and exit")
        print("========================================================\n")

        try:
            while True:
                t0 = time.perf_counter()

                # 1. Execute hardware acquisition & inference pipeline step
                telemetry = self.runner.step()

                # 2. Accumulate automated test frames if active
                raw_img = self.runner.latest_camera_frame
                ovl_img = self.runner.latest_overlaid_frame
                hmap = self.runner.latest_heatmap

                h_res = self.panel_horizontal.update_frame(
                    telemetry, raw_image=raw_img, overlaid_image=ovl_img, heatmap=hmap
                )
                if h_res:
                    h_err = h_res.get("pixel_error") or 0.0
                    self.set_notification(
                        f"Trial {h_res.get('run_id')} completed: {h_res.get('outcome_status')} "
                        f"(Err {h_err:.1f}px)"
                    )

                v_res = self.panel_vertical.update_frame(
                    telemetry, raw_image=raw_img, overlaid_image=ovl_img, heatmap=hmap
                )
                if v_res:
                    v_err = v_res.get("pixel_error") or 0.0
                    self.set_notification(
                        f"Trial {v_res.get('run_id')} completed: {v_res.get('outcome_status')} "
                        f"(Err {v_err:.1f}px)"
                    )

                # 3. Compose full GUI frame
                canvas = self.compose_frame(telemetry)

                # Calculate measured display loop FPS
                dt = time.perf_counter() - t0
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self.fps = 0.9 * self.fps + 0.1 * inst_fps if self.fps > 0 else inst_fps

                frame_count += 1
                if self.headless_frames and frame_count >= self.headless_frames:
                    print(f"[App Headless] Completed {frame_count} frames successfully.")
                    break

                # 8. Display canvas and handle keyboard input
                if not self.headless_frames:
                    rect = cv2.getWindowImageRect(window_name)
                    win_w, win_h = rect[2], rect[3]

                    if win_w > 0 and win_h > 0 and (win_w != CANVAS_WIDTH or win_h != CANVAS_HEIGHT):
                        # Window size differs from native 1600x900: preserve strict 16:9 aspect ratio
                        scale = min(win_w / CANVAS_WIDTH, win_h / CANVAS_HEIGHT)
                        target_w = int(round(CANVAS_WIDTH * scale))
                        target_h = int(round(CANVAS_HEIGHT * scale))
                        self.display_scale = scale
                        self.display_pad_x = (win_w - target_w) // 2
                        self.display_pad_y = (win_h - target_h) // 2

                        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                        scaled = cv2.resize(canvas, (target_w, target_h), interpolation=interp)
                        display_frame = np.full((win_h, win_w, 3), BG_CANVAS, dtype=np.uint8)
                        display_frame[
                            self.display_pad_y : self.display_pad_y + target_h,
                            self.display_pad_x : self.display_pad_x + target_w,
                        ] = scaled
                        cv2.imshow(window_name, display_frame)
                    else:
                        self.display_scale = 1.0
                        self.display_pad_x = 0
                        self.display_pad_y = 0
                        cv2.imshow(window_name, canvas)

                    key = cv2.waitKey(1) & 0xFF

                    if key in [ord('q'), ord('Q'), 27]:  # 27 = ESC
                        print("[App] Quit key pressed. Exiting...")
                        break
                    elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]:
                        self.current_mode = key - ord('0')
                        self.set_notification(f"Switched to Mode {self.current_mode}: {TAB_LABELS[self.current_mode - 1]}")
                    elif key == ord('0'):
                        self.current_mode = 10
                        self.set_notification("Switched to Mode 10: [0] ROADMAP")
                    elif key in [ord('s'), ord('S')]:
                        try:
                            saved_path = self.runner.save_snapshot(canvas.copy())
                            self.set_notification(f"Snapshot saved: {os.path.basename(saved_path)}")
                        except Exception as e:
                            print(f"[App] Snapshot error: {e}")
                            self.set_notification(f"Snapshot error: {e}")
                    elif key in [ord('e'), ord('E')]:
                        try:
                            csv_path = self.repo.export_csv()
                            json_path = self.repo.export_json()
                            self.set_notification(f"DB Exported to CSV ({os.path.basename(csv_path)}) & JSON")
                        except Exception as e:
                            print(f"[App] Export error: {e}")
                            self.set_notification(f"Export error: {e}")
                    elif key in [ord('r'), ord('R')]:
                        try:
                            is_rec, rec_file = self.runner.toggle_session_recording()
                            if is_rec:
                                self.set_notification(f"Session recording STARTED: {os.path.basename(rec_file or '')}")
                            else:
                                self.set_notification(f"Session recording STOPPED & SAVED: {os.path.basename(rec_file or '')}")
                        except Exception as e:
                            print(f"[App] Recording error: {e}")
                            self.set_notification(f"Recording error: {e}")
                    elif key in [ord('c'), ord('C')]:
                        self.runner.clear_ground_truth()
                        self.set_notification("Cleared Ground Truth Target.")
                    elif key in [ord('y'), ord('Y')]:
                        new_state = self.runner.toggle_yolo()
                        status_str = "ENABLED" if new_state else "DISABLED (Inference Skipped)"
                        self.set_notification(f"YOLO Mode switched: {status_str}")

        except KeyboardInterrupt:
            print("\n[App] Interrupted by user.")
        finally:
            print("[App] Shutting down runner and releasing devices...")
            try:
                self.repo.end_session(self.session_id)
            except Exception as e:
                print(f"[App] Failed to close session record: {e}")
            self.runner.close()
            if not self.headless_frames:
                cv2.destroyAllWindows()
            print("[App] Shutdown complete.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Faculty Presentation Application for Real-Time Multimodal Acoustic Camera"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in offline simulation/replay mode without requiring live ESP32-CAM or UMA-16 hardware",
    )
    parser.add_argument(
        "--stream-url",
        type=str,
        default=STREAM_URL,
        help=f"ESP32-CAM MJPEG stream URL (default: {STREAM_URL})",
    )
    parser.add_argument(
        "--audio-device",
        type=str,
        default=None,
        help="Device index or name substring for miniDSP UMA-16 (default: auto-detect)",
    )
    parser.add_argument(
        "--probe-hardware",
        action="store_true",
        help="Probe audio and camera hardware, print diagnostic report to console, and exit",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scratch/demo_results",
        help="Directory to save snapshots and session JSON telemetry (default: scratch/demo_results)",
    )
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=0,
        help="Run for N frames without opening an OpenCV GUI window (useful for automated testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.probe_hardware:
        from hardware.devices import probe_all_hardware
        print("\nProbing hardware components (ESP32-CAM & miniDSP UMA-16)...")
        report = probe_all_hardware(
            audio_preference=args.audio_device,
            camera_url=args.stream_url,
        )
        print()
        print(report.formatted_banner())
        return

    app = ProfessorDemoApp(
        stream_url=args.stream_url,
        simulate=args.simulate,
        output_dir=args.output_dir,
        headless_frames=args.headless_frames,
        audio_device=args.audio_device,
    )
    app.run()


if __name__ == "__main__":
    main()

