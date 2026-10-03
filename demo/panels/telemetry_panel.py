"""
demo.panels.telemetry_panel
===========================
Persistent right-hand telemetry and status sidebar for the Professor Demo.
Formatted with crisp, high-contrast, non-aliased typography.
"""

from __future__ import annotations
from typing import Any, Dict
import cv2
import numpy as np

from demo.ui.theme import (
    FONT_HEADER, FONT_BODY, AA,
    SCALE_PANEL_TITLE, SCALE_SECTION, SCALE_LABEL, SCALE_VALUE, SCALE_SMALL,
    BG_CARD, BG_PANEL, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ACCENT,
    ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER, ACCENT_CYAN,
    ACCENT_BLUE, ACCENT_PURPLE, ACCENT_ORANGE
)
from demo.ui.widgets import (
    draw_panel, draw_card, draw_badge, draw_metric_row, draw_bar, draw_section
)


class TelemetrySidebar:
    """Renders the comprehensive right-hand status and live telemetry panel."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def render(
        self,
        canvas: np.ndarray,
        telemetry: Dict[str, Any],
        recording_session: bool = False,
    ) -> None:
        """Render the complete status sidebar into the canvas."""
        x, y, w, h = self.x, self.y, self.w, self.h

        # Outer Panel container with top title bar
        draw_panel(canvas, x, y, w, h, title="SYSTEM & TELEMETRY MONITOR")

        curr_y = y + 42

        # ----------------------------------------------------
        # 1. SYSTEM STATUS BADGES
        # ----------------------------------------------------
        draw_section(canvas, "Hardware & Pipeline Status", x + 16, curr_y, TEXT_ACCENT, thickness=2)
        curr_y += 12

        # Row 1 of badges: Camera & Audio
        cam_ok = telemetry.get("camera_connected", False)
        aud_ok = telemetry.get("audio_connected", False)
        bw1 = draw_badge(canvas, x + 16, curr_y, "ESP32-CAM", status_ok=cam_ok)
        draw_badge(canvas, x + 24 + bw1, curr_y, "UMA-16 (16ch)", status_ok=aud_ok)
        curr_y += 30

        # Row 2 of badges: MUSIC, YOLO, Sync
        mus_ok = telemetry.get("music_running", False) and telemetry.get("peak_valid", False)
        yolo_enabled = telemetry.get("yolo_enabled", True)
        yolo_ok = telemetry.get("yolo_running", False) and yolo_enabled
        sync_ok = telemetry.get("synchronized", False)

        bw2 = draw_badge(canvas, x + 16, curr_y, "MUSIC Core", status_ok=mus_ok, warning=not telemetry.get("peak_valid", False))
        yolo_badge_label = "YOLO11n-seg" if yolo_enabled else "YOLO: OFF"
        bw3 = draw_badge(canvas, x + 24 + bw2, curr_y, yolo_badge_label, status_ok=yolo_ok, warning=not yolo_enabled)
        draw_badge(canvas, x + 32 + bw2 + bw3, curr_y, "Sync (<100ms)", status_ok=sync_ok)
        curr_y += 38

        # ----------------------------------------------------
        # 2. ACOUSTIC LOCALIZATION (MUSIC)
        # ----------------------------------------------------
        az = telemetry.get("azimuth_deg")
        az = 90.0 if az is None else az
        el = telemetry.get("elevation_deg")
        el = -6.0 if el is None else el
        peak_db = telemetry.get("peak_db")
        peak_db = 0.0 if peak_db is None else peak_db
        valid = telemetry.get("peak_valid", False)

        draw_card(canvas, x + 12, curr_y, w - 24, 104)
        draw_section(canvas, "Acoustic Localization (MUSIC)", x + 22, curr_y + 22, ACCENT_BLUE, thickness=2)

        draw_metric_row(canvas, x + 22, curr_y + 46, "Azimuth Angle (theta)", f"{az:.1f} deg", ACCENT_CYAN, width=w - 48, bold_value=True)
        draw_metric_row(canvas, x + 22, curr_y + 68, "Elevation Angle (phi)", f"{el:.1f} deg", ACCENT_CYAN, width=w - 48, bold_value=True)
        db_color = ACCENT_GREEN if valid else ACCENT_AMBER
        db_label = f"{peak_db:.1f} dB SPL" if valid else f"{peak_db:.1f} dB (Below Gate)"
        draw_metric_row(canvas, x + 22, curr_y + 90, "Signal Level / Peak", db_label, db_color, width=w - 48, bold_value=True)
        curr_y += 114

        # ----------------------------------------------------
        # 3. VISION PERCEPTION (YOLO11n-seg)
        # ----------------------------------------------------
        y_obj = telemetry.get("yolo_object") or "NONE"
        y_conf = telemetry.get("yolo_confidence")
        y_conf = 0.0 if y_conf is None else y_conf
        y_center = telemetry.get("yolo_center")
        y_lat = telemetry.get("yolo_latency_ms")
        y_lat = 0.0 if y_lat is None else y_lat

        draw_card(canvas, x + 12, curr_y, w - 24, 104)
        draw_section(canvas, "Vision Perception (YOLO11n-seg)", x + 22, curr_y + 22, ACCENT_PURPLE, thickness=2)

        if not yolo_enabled:
            draw_metric_row(canvas, x + 22, curr_y + 46, "Vision Inference", "DISABLED (Inference Off)", ACCENT_AMBER, width=w - 48, bold_value=True)
            draw_metric_row(canvas, x + 22, curr_y + 68, "Centroid Coordinates", "-- (Inference Bypassed)", TEXT_MUTED, width=w - 48)
            draw_metric_row(canvas, x + 22, curr_y + 90, "Inference Latency", "0.0 ms (Saved CPU)", ACCENT_GREEN, width=w - 48)
        else:
            conf_str = f"{y_conf * 100.0:.1f}%" if y_conf > 0 else "--"
            draw_metric_row(canvas, x + 22, curr_y + 46, "Detected Object Class", f"{y_obj} ({conf_str})", TEXT_PRIMARY, width=w - 48)
            c_str = f"({int(y_center[0])}, {int(y_center[1])}) px" if y_center else "None"
            draw_metric_row(canvas, x + 22, curr_y + 68, "Centroid Coordinates", c_str, TEXT_PRIMARY, width=w - 48)
            draw_metric_row(canvas, x + 22, curr_y + 90, "Inference Latency", f"{y_lat:.1f} ms", TEXT_SECONDARY, width=w - 48)
        curr_y += 114

        # ----------------------------------------------------
        # 4. CROSS-MODAL SYNCHRONIZATION
        # ----------------------------------------------------
        dt_ms = telemetry.get("sync_delta_ms")
        dt_ms = 0.0 if dt_ms is None else dt_ms
        sync_state = "SYNCHRONIZED" if telemetry.get("synchronized", False) else "UNSYNCHRONIZED"
        state_col = ACCENT_GREEN if sync_state == "SYNCHRONIZED" else ACCENT_RED

        draw_card(canvas, x + 12, curr_y, w - 24, 114)
        draw_section(canvas, "Temporal Synchronization", x + 22, curr_y + 22, ACCENT_GREEN, thickness=2)

        draw_metric_row(canvas, x + 22, curr_y + 46, "Timestamp Delta (Delta t)", f"{dt_ms:.1f} ms", state_col, width=w - 48, bold_value=True)
        draw_metric_row(canvas, x + 22, curr_y + 68, "Criterion Tolerance", "< 100.0 ms", TEXT_MUTED, width=w - 48)
        draw_metric_row(canvas, x + 22, curr_y + 90, "Synchronization State", sync_state, state_col, width=w - 48, bold_value=True)

        # Meter bar for sync delta
        draw_bar(canvas, x + 22, curr_y + 100, w - 48, 8, dt_ms / 100.0, color=state_col)
        curr_y += 124

        # ----------------------------------------------------
        # 5. PROJECTION & GROUND TRUTH ERROR
        # ----------------------------------------------------
        u = telemetry.get("overlay_u", 320)
        v = telemetry.get("overlay_v", 240)
        gt = telemetry.get("gt_click")
        err_px = telemetry.get("pixel_error_gt")

        draw_card(canvas, x + 12, curr_y, w - 24, 114)
        draw_section(canvas, "Camera-Acoustic Projection", x + 22, curr_y + 22, ACCENT_ORANGE, thickness=2)

        draw_metric_row(canvas, x + 22, curr_y + 46, "Projected Pixel (u, v)", f"({u}, {v}) px", ACCENT_CYAN, width=w - 48, bold_value=True)
        gt_str = f"({gt[0]}, {gt[1]}) px" if gt else "None (Click camera)"
        draw_metric_row(canvas, x + 22, curr_y + 68, "Target Ground Truth", gt_str, ACCENT_ORANGE if gt else TEXT_MUTED, width=w - 48)

        err_str = f"{err_px:.1f} px" if err_px is not None else "--"
        draw_metric_row(canvas, x + 22, curr_y + 90, "Image-Space Projection Error", err_str, ACCENT_AMBER if err_px else TEXT_MUTED, width=w - 48, bold_value=(err_px is not None))
        cv2.putText(canvas, "* Pinhole baseline; calibration ongoing", (x + 22, curr_y + 106), FONT_BODY, SCALE_SMALL, TEXT_MUTED, 1, AA)
        curr_y += 124

        # ----------------------------------------------------
        # 6. ACTION BUTTONS & TELEMETRY CONTROLS
        # ----------------------------------------------------
        draw_card(canvas, x + 12, curr_y, w - 24, 82)
        draw_section(canvas, "Data Capture & Recording", x + 22, curr_y + 22, TEXT_ACCENT, thickness=2)

        # Snapshot button hint
        cv2.putText(canvas, "[S] Save Snapshot", (x + 22, curr_y + 46), FONT_BODY, SCALE_LABEL, ACCENT_CYAN, 1, AA)
        cv2.putText(canvas, "Saves JSON & JPG to scratch/demo_results/", (x + 22, curr_y + 68), FONT_BODY, SCALE_SMALL, TEXT_MUTED, 1, AA)

        # Recording session status
        rec_label = "[R] Recording Active" if recording_session else "[R] Record Session"
        rec_color = ACCENT_RED if recording_session else TEXT_SECONDARY
        cv2.putText(canvas, rec_label, (x + 295, curr_y + 46), FONT_BODY, SCALE_LABEL, rec_color, 1, AA)
        cv2.putText(canvas, "Continuous telemetry stream log", (x + 295, curr_y + 68), FONT_BODY, SCALE_SMALL, TEXT_MUTED, 1, AA)
