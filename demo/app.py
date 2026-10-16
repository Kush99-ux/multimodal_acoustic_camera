"""
demo.app
========
Faculty Presentation Application for the Real-Time Multimodal Acoustic Camera.

Launch with:
    python demo/app.py
or for offline demonstration:
    python demo/app.py --simulate

Features:
- Full 1600x900 Ivory & Navy scientific dashboard
- 8 Interactive Modes:
    1: Live Multimodal Acoustic-Vision Fusion with Ground Truth Target click
    2: 2D MUSIC Spatial Spectrum (7,676 points, 101x76 grid)
    3: YOLO11n-seg Instance Segmentation & Confidence Tracking
    4: Guided Horizontal Localization Test (Left / Center / Right with 20-frame collection)
    5: Guided Vertical Elevation Localization Test (Above / Level / Below)
    6: Cross-Modal Timestamp Synchronization & Hardware Jitter Monitor
    7: 16-Microphone Physical Array Layout & 6 Sub-Band Spectral Power Meters
    8: Scientific Validation Benchmark Tables & Historical Verification Runs
- Destination-specific live measurements and status details
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
    MultimodalPanel,
    AcousticPanel,
    VisionPanel,
    HorizontalTestPanel,
    VerticalTestPanel,
    SynchronizationPanel,
    DiagnosticsPanel,
    ValidationPanel,
)
from demo.ui.theme import (
    CANVAS_WIDTH, CANVAS_HEIGHT, FOOTER_HEIGHT,
    FONT_BODY, AA, SPACE_MD, RADIUS_BUTTON,
    BG_CANVAS, BG_PANEL, TEXT_SECONDARY, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED
)
from demo.ui.widgets import draw_app_title, draw_button, draw_divider, draw_status_chip, rounded_rect_contains


MODE_LABELS = {
    1: "Live Fusion",
    2: "2D MUSIC",
    3: "YOLO Vision",
    4: "Azimuth Test",
    5: "Elevation Test",
    6: "Sync Monitor",
    7: "Array & FFT",
    8: "Benchmarks",
}
NAV_ITEMS = (
    ("Live Fusion", 1),
    ("Acoustics", 2),
    ("Vision", 3),
    ("Sync", 6),
    ("Tests", 8),
)
ACOUSTIC_MODE_ITEMS = (
    (2, "MUSIC"),
    (4, "Azimuth"),
    (5, "Elevation"),
    (7, "Array/FFT"),
)
ACOUSTIC_MODES = tuple(mode for mode, _ in ACOUSTIC_MODE_ITEMS)
NAV_MARGIN = 16
NAV_GAP = 12
NAV_Y = 36
NAV_HEIGHT = 28
SUBNAV_Y = 68
SUBNAV_HEIGHT = 22
SUBNAV_GAP = 6
SUBNAV_RADIUS = 8
CONTENT_TOP = 68
CONTENT_TOP_WITH_SUBNAV = 96
PROFILE_BUTTON = (1010, 5, 160, 26)
YOLO_BUTTON = (1182, 5, 142, 26)
FPS_CHIP = (1340, 6)
RECORDING_CHIP = (1445, 6)


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

        self.current_mode = 1  # 1..8
        self.profile_mode = "PRODUCTION"  # "PRODUCTION" (default) or "DIAGNOSTIC"
        # Canvas layout geometry
        self.main_x_default = 10
        self.main_y_default = CONTENT_TOP
        self.main_h_default = CANVAS_HEIGHT - CONTENT_TOP - FOOTER_HEIGHT

        self.main_x = self.main_x_default
        self.main_y = self.main_y_default
        self.main_w = CANVAS_WIDTH - 20
        self.main_h = self.main_h_default

        # Five balanced conceptual destinations across the available header width.
        self.nav_button_w = (CANVAS_WIDTH - 2 * NAV_MARGIN - (len(NAV_ITEMS) - 1) * NAV_GAP) // len(NAV_ITEMS)
        self.nav_rects = tuple(
            (NAV_MARGIN + i * (self.nav_button_w + NAV_GAP), NAV_Y, self.nav_button_w, NAV_HEIGHT)
            for i in range(len(NAV_ITEMS))
        )
        self.subnav_button_w = (self.nav_button_w - (len(ACOUSTIC_MODE_ITEMS) - 1) * SUBNAV_GAP) // len(ACOUSTIC_MODE_ITEMS)

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


        # Initialize the eight active mode panels
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
        # FPS calculation
        self.last_time = time.perf_counter()
        self.fps = 0.0

        # Dynamic display scaling & letterboxing tracking (for aspect-ratio preservation)
        self.display_scale = 1.0
        self.display_pad_x = 0
        self.display_pad_y = 0

    def set_notification(self, text: str) -> None:
        """Log an application notification without adding a persistent UI ticker."""
        print(f"[App Notification] {text}")

    def _apply_profile_layout(self) -> None:
        """Update layout geometry and resize panels according to active profile."""
        if self.profile_mode == "DIAGNOSTIC":
            self.main_x = 10
            self.main_y = 38
            self.main_w = CANVAS_WIDTH - 20  # 1580 px
            self.main_h = CANVAS_HEIGHT - 38 - 10  # 852 px
        else:
            self.main_x = self.main_x_default
            self.main_y = CONTENT_TOP_WITH_SUBNAV if self.current_mode in ACOUSTIC_MODES else CONTENT_TOP
            self.main_w = CANVAS_WIDTH - 20
            self.main_h = CANVAS_HEIGHT - self.main_y - FOOTER_HEIGHT

        for panel in (
            self.panel_multimodal,
            self.panel_acoustic,
            self.panel_vision,
            self.panel_horizontal,
            self.panel_vertical,
            self.panel_sync,
            self.panel_diagnostics,
            self.panel_validation,
        ):
            if hasattr(panel, "set_layout"):
                if panel is self.panel_multimodal or panel is self.panel_vision:
                    panel.set_layout(
                        self.main_x,
                        self.main_y,
                        self.main_w,
                        self.main_h,
                        preserve_aspect=True,
                    )
                else:
                    panel.set_layout(self.main_x, self.main_y, self.main_w, self.main_h)

    def _select_mode(self, mode: int) -> bool:
        """Select one of the eight active modes and synchronize its layout."""
        if mode not in MODE_LABELS or mode == self.current_mode:
            return False

        self.current_mode = mode
        self.runner.set_diagnostics_enabled(self.profile_mode == "DIAGNOSTIC" or mode == 7)
        self._apply_profile_layout()
        self.set_notification(f"Switched to Mode {mode}: {MODE_LABELS[mode]}")
        return True

    def _navigation_group_index(self) -> int:
        """Return the conceptual navigation destination for the active mode."""
        if self.current_mode == 1:
            return 0
        if self.current_mode in ACOUSTIC_MODES:
            return 1
        if self.current_mode == 3:
            return 2
        if self.current_mode == 6:
            return 3
        return 4

    def toggle_profile_mode(self) -> str:
        """Toggle between PRODUCTION and DIAGNOSTIC execution profiles."""
        if self.profile_mode == "PRODUCTION":
            self.profile_mode = "DIAGNOSTIC"
            self.runner.set_diagnostics_enabled(True)
        else:
            self.profile_mode = "PRODUCTION"
            # In production, diagnostics run only if currently in Mode 7
            self.runner.set_diagnostics_enabled(self.current_mode == 7)
        self._apply_profile_layout()
        self.set_notification(f"Profile switched to [{self.profile_mode} MODE]")
        return self.profile_mode

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

        # Header profile toggle uses the same geometry as its rendered rounded control.
        px, py, pw, ph = PROFILE_BUTTON
        if rounded_rect_contains(x, y, px, py, pw, ph, RADIUS_BUTTON):
            self.toggle_profile_mode()
            return

        # Header YOLO toggle is available in DIAGNOSTIC mode only.
        yx, yy, yw, yh = YOLO_BUTTON
        if self.profile_mode == "DIAGNOSTIC" and rounded_rect_contains(x, y, yx, yy, yw, yh, RADIUS_BUTTON):
            new_state = self.runner.toggle_yolo()
            status_str = "ENABLED" if new_state else "DISABLED (Inference Skipped)"
            self.set_notification(f"YOLO Mode switched: {status_str}")
            return

        if self.profile_mode == "PRODUCTION":
            for rect, (_, default_mode) in zip(self.nav_rects, NAV_ITEMS):
                nx, ny, nw, nh = rect
                if rounded_rect_contains(x, y, nx, ny, nw, nh, RADIUS_BUTTON):
                    self._select_mode(default_mode)
                    return

            if self.current_mode in ACOUSTIC_MODES:
                acoustic_x = self.nav_rects[1][0]
                total_subnav_w = len(ACOUSTIC_MODE_ITEMS) * self.subnav_button_w + (len(ACOUSTIC_MODE_ITEMS) - 1) * SUBNAV_GAP
                subnav_x = acoustic_x + (self.nav_button_w - total_subnav_w) // 2
                if SUBNAV_Y <= y <= SUBNAV_Y + SUBNAV_HEIGHT:
                    for i, (mode, _) in enumerate(ACOUSTIC_MODE_ITEMS):
                        sx = subnav_x + i * (self.subnav_button_w + SUBNAV_GAP)
                        if rounded_rect_contains(x, y, sx, SUBNAV_Y, self.subnav_button_w, SUBNAV_HEIGHT, SUBNAV_RADIUS):
                            self._select_mode(mode)
                            return

        # Handle Mode 1: Multimodal ground truth click & YOLO toggle
        if self.current_mode == 1:
            if self.profile_mode == "PRODUCTION" and self.panel_multimodal.handle_yolo_click(x, y):
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
        """Render the toolbar, conceptual destinations, and contextual mode navigation."""
        header_h = self.main_y if self.profile_mode == "PRODUCTION" else 34
        cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, header_h), BG_CANVAS, -1)
        draw_app_title(
            canvas,
            "REAL-TIME MULTIMODAL ACOUSTIC CAMERA",
            18,
            26,
        )

        diag_active = (self.profile_mode == "DIAGNOSTIC")
        px, py, pw, ph = PROFILE_BUTTON
        draw_button(
            canvas,
            "[P] DIAGNOSTIC" if diag_active else "[P] PRODUCTION",
            px,
            py,
            pw,
            ph,
            active=diag_active,
            border_color=ACCENT_AMBER if diag_active else ACCENT_GREEN,
            scale=0.42,
        )

        if self.profile_mode == "DIAGNOSTIC":
            yolo_on = telemetry.get("yolo_enabled", True)
            yx, yy, yw, yh = YOLO_BUTTON
            draw_button(
                canvas,
                "[Y] YOLO ON" if yolo_on else "[Y] YOLO OFF",
                yx,
                yy,
                yw,
                yh,
                active=yolo_on,
                border_color=ACCENT_GREEN if yolo_on else ACCENT_RED,
                scale=0.42,
            )

        fps_x, fps_y = FPS_CHIP
        draw_status_chip(canvas, fps_x, fps_y, f"{self.fps:4.1f} FPS")
        rec_status = "[REC]" if self.runner.logger.recording_session else "IDLE"
        rec_tone = "error" if self.runner.logger.recording_session else "neutral"
        rec_x, rec_y = RECORDING_CHIP
        draw_status_chip(canvas, rec_x, rec_y, rec_status, tone=rec_tone)

        if self.profile_mode == "PRODUCTION":
            active_group = self._navigation_group_index()
            for i, (label, _) in enumerate(NAV_ITEMS):
                nx, ny, nw, nh = self.nav_rects[i]
                draw_button(canvas, label, nx, ny, nw, nh, active=(i == active_group), scale=0.48)

            if self.current_mode in ACOUSTIC_MODES:
                acoustic_x = self.nav_rects[1][0]
                total_subnav_w = len(ACOUSTIC_MODE_ITEMS) * self.subnav_button_w + (len(ACOUSTIC_MODE_ITEMS) - 1) * SUBNAV_GAP
                subnav_x = acoustic_x + (self.nav_button_w - total_subnav_w) // 2
                for i, (mode, label) in enumerate(ACOUSTIC_MODE_ITEMS):
                    sx = subnav_x + i * (self.subnav_button_w + SUBNAV_GAP)
                    draw_button(
                        canvas,
                        label,
                        sx,
                        SUBNAV_Y,
                        self.subnav_button_w,
                        SUBNAV_HEIGHT,
                        active=(self.current_mode == mode),
                        radius=SUBNAV_RADIUS,
                        scale=0.38,
                    )

        draw_divider(canvas, 0, header_h - 1, CANVAS_WIDTH - 1, header_h - 1)

    def _render_footer(self, canvas: np.ndarray) -> None:
        """Render keyboard shortcuts in the footer."""
        fy = CANVAS_HEIGHT - FOOTER_HEIGHT
        cv2.rectangle(canvas, (0, fy), (CANVAS_WIDTH, CANVAS_HEIGHT), BG_PANEL, -1)
        draw_divider(canvas, 0, fy, CANVAS_WIDTH - 1, fy)

        # Hotkeys guide (left)
        guide = "[1-8] Modes   [P] Profile   [Y] YOLO   [S] Snapshot   [E] Export   [R] Record   [C] Clear Target   [Q] Quit"
        cv2.putText(canvas, guide, (SPACE_MD, fy + 27), FONT_BODY, 0.40, TEXT_SECONDARY, 1, AA)

    def compose_frame(self, telemetry: dict) -> np.ndarray:
        """Compose the complete 1600x900 GUI canvas for the current active mode."""
        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), BG_CANVAS, dtype=np.uint8)

        # Ensure profile_mode is populated in telemetry for panel-level status
        telemetry["profile_mode"] = self.profile_mode

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
        if self.profile_mode == "PRODUCTION":
            self._render_footer(canvas)

        return canvas

    def run(self) -> None:
        """Main application display loop."""
        window_name = "Real-Time Multimodal Acoustic Camera"

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
        print("  [1-8]       : Switch between presentation modes 1 to 8")
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
                    elif key in [ord('p'), ord('P')]:
                        self.toggle_profile_mode()
                    elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8')]:
                        self._select_mode(key - ord('0'))
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
                    elif key in [ord('r'), ord('R')] and self.profile_mode == "PRODUCTION":
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
            print("[App] Shutting down background workers and releasing devices...")
            try:
                self.experiment_manager.shutdown(timeout=5.0)
            except Exception as e:
                print(f"[App] Failed during experiment manager shutdown: {e}")
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

