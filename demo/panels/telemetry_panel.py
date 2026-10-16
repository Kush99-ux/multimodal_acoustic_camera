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
    TEXT_SUCCESS, TEXT_WARNING, TEXT_ERROR,
    ACCENT_UI, ACCENT_ORANGE
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

        # Row 1 of badges: Profile Mode, Camera & Audio
        diag_on = telemetry.get("profile_mode") == "DIAGNOSTIC" or telemetry.get("diagnostics_enabled", False)
        profile_label = "DIAGNOSTIC" if diag_on else "PRODUCTION"
        bw0 = draw_badge(canvas, x + 16, curr_y, profile_label, status_ok=not diag_on, warning=diag_on)
        cam_ok = telemetry.get("camera_connected", False)
        aud_ok = telemetry.get("audio_connected", False)
        bw1 = draw_badge(canvas, x + 24 + bw0, curr_y, "ESP32-CAM", status_ok=cam_ok)
        draw_badge(canvas, x + 32 + bw0 + bw1, curr_y, "UMA-16 (16ch)", status_ok=aud_ok)
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
        el = telemetry.get("elevation_deg")
        peak_db = telemetry.get("peak_db")
        valid = telemetry.get("peak_valid", False)
        audio_has_frame = bool(telemetry.get("audio_connected", False)) and bool(telemetry.get("audio_timestamp"))

        draw_card(canvas, x + 12, curr_y, w - 24, 104)
        draw_section(canvas, "Acoustic Localization (MUSIC)", x + 22, curr_y + 22, ACCENT_UI, thickness=2)

        angle_color = ACCENT_UI if valid else TEXT_MUTED
        az_label = f"{az:.1f} deg" if valid and az is not None else "--"
        el_label = f"{el:.1f} deg" if valid and el is not None else "--"
        draw_metric_row(canvas, x + 22, curr_y + 46, "Azimuth Angle (theta)", az_label, angle_color, width=w - 48, bold_value=valid)
        draw_metric_row(canvas, x + 22, curr_y + 68, "Elevation Angle (phi)", el_label, angle_color, width=w - 48, bold_value=valid)
        db_color = TEXT_SUCCESS if valid else TEXT_WARNING
        db_label = (f"{peak_db:.1f} dB SPL" if valid else f"{peak_db:.1f} dB (Below Gate)") if audio_has_frame and peak_db is not None else "--"
        if not audio_has_frame:
            db_color = TEXT_MUTED
        draw_metric_row(canvas, x + 22, curr_y + 90, "Signal Level / Peak", db_label, db_color, width=w - 48, bold_value=True)
        curr_y += 114

        # ----------------------------------------------------
        # 3. VISION PERCEPTION (YOLO11n-seg)
        # ----------------------------------------------------
        y_obj = telemetry.get("yolo_object") or "NONE"
        y_conf = telemetry.get("yolo_confidence")
        y_center = telemetry.get("yolo_center")
        y_lat = telemetry.get("yolo_latency_ms")
        vision_available = bool(telemetry.get("camera_connected", False)) and bool(telemetry.get("yolo_running", False)) and yolo_enabled

        draw_card(canvas, x + 12, curr_y, w - 24, 104)
        draw_section(canvas, "Vision Perception (YOLO11n-seg)", x + 22, curr_y + 22, ACCENT_UI, thickness=2)

        if not yolo_enabled:
            draw_metric_row(canvas, x + 22, curr_y + 46, "Vision Inference", "DISABLED (Inference Off)", TEXT_WARNING, width=w - 48, bold_value=True)
            draw_metric_row(canvas, x + 22, curr_y + 68, "Centroid Coordinates", "-- (Inference Bypassed)", TEXT_MUTED, width=w - 48)
            draw_metric_row(canvas, x + 22, curr_y + 90, "Inference Latency", "--", TEXT_MUTED, width=w - 48)
        else:
            object_label = "No detection" if vision_available and y_obj == "NONE" else (y_obj or "--")
            conf_str = f"{y_conf * 100.0:.1f}%" if vision_available and y_conf is not None and y_obj != "NONE" else "--"
            inference_color = TEXT_PRIMARY if vision_available else TEXT_MUTED
            draw_metric_row(canvas, x + 22, curr_y + 46, "Detected Object Class", f"{object_label} ({conf_str})" if conf_str != "--" else object_label, inference_color, width=w - 48)
            c_str = f"({int(y_center[0])}, {int(y_center[1])}) px" if vision_available and y_center else "--"
            draw_metric_row(canvas, x + 22, curr_y + 68, "Centroid Coordinates", c_str, inference_color, width=w - 48)
            latency_str = f"{y_lat:.1f} ms" if vision_available and y_lat is not None else "--"
            draw_metric_row(canvas, x + 22, curr_y + 90, "Inference Latency", latency_str, inference_color, width=w - 48)
        curr_y += 114

        # ----------------------------------------------------
        # 4. CROSS-MODAL SYNCHRONIZATION
        # ----------------------------------------------------
        dt_ms = telemetry.get("sync_delta_ms")
        sync_available = (
            bool(telemetry.get("camera_connected", False))
            and bool(telemetry.get("audio_connected", False))
            and bool(telemetry.get("camera_timestamp"))
            and bool(telemetry.get("audio_timestamp"))
            and dt_ms is not None
        )
        sync_state = ("SYNCHRONIZED" if telemetry.get("synchronized", False) else "OUTSIDE TOLERANCE") if sync_available else "WAITING FOR PAIRED FRAMES"
        state_col = TEXT_SUCCESS if sync_available and telemetry.get("synchronized", False) else (TEXT_ERROR if sync_available else TEXT_MUTED)

        draw_card(canvas, x + 12, curr_y, w - 24, 114)
        draw_section(canvas, "Temporal Synchronization", x + 22, curr_y + 22, TEXT_SUCCESS, thickness=2)

        draw_metric_row(canvas, x + 22, curr_y + 46, "Timestamp Delta (Delta t)", f"{dt_ms:.1f} ms" if sync_available else "--", state_col, width=w - 48, bold_value=sync_available)
        draw_metric_row(canvas, x + 22, curr_y + 68, "Criterion Tolerance", "< 100.0 ms", TEXT_MUTED, width=w - 48)
        draw_metric_row(canvas, x + 22, curr_y + 90, "Synchronization State", sync_state, state_col, width=w - 48, bold_value=True)

        # Meter bar for sync delta
        if sync_available:
            draw_bar(canvas, x + 22, curr_y + 100, w - 48, 8, dt_ms / 100.0, color=state_col)
        curr_y += 124

        # ----------------------------------------------------
        # 5. PROJECTION & GROUND TRUTH ERROR
        # ----------------------------------------------------
        u = telemetry.get("overlay_u")
        v = telemetry.get("overlay_v")
        projection_available = bool(telemetry.get("peak_valid", False)) and u is not None and v is not None
        gt = telemetry.get("gt_click")
        err_px = telemetry.get("pixel_error_gt")

        draw_card(canvas, x + 12, curr_y, w - 24, 114)
        draw_section(canvas, "Camera-Acoustic Projection", x + 22, curr_y + 22, ACCENT_UI, thickness=2)

        draw_metric_row(canvas, x + 22, curr_y + 46, "Projected Pixel (u, v)", f"({u}, {v}) px" if projection_available else "--", ACCENT_UI if projection_available else TEXT_MUTED, width=w - 48, bold_value=projection_available)
        gt_str = f"({gt[0]}, {gt[1]}) px" if gt else "None (Click camera)"
        draw_metric_row(canvas, x + 22, curr_y + 68, "Target Ground Truth", gt_str, ACCENT_ORANGE if gt else TEXT_MUTED, width=w - 48)

        err_str = f"{err_px:.1f} px" if err_px is not None else "--"
        draw_metric_row(canvas, x + 22, curr_y + 90, "Image-Space Projection Error", err_str, TEXT_WARNING if err_px else TEXT_MUTED, width=w - 48, bold_value=(err_px is not None))
        cv2.putText(canvas, "* Pinhole baseline; calibration ongoing", (x + 22, curr_y + 106), FONT_BODY, SCALE_SMALL, TEXT_MUTED, 1, AA)
        curr_y += 124

        # ----------------------------------------------------
        # 6. ACTION BUTTONS & TELEMETRY CONTROLS
        # ----------------------------------------------------
        draw_card(canvas, x + 12, curr_y, w - 24, 82)
        draw_section(canvas, "Data Capture & Recording", x + 22, curr_y + 22, TEXT_ACCENT, thickness=2)

        # Snapshot button hint
        cv2.putText(canvas, "[S] Save Snapshot", (x + 22, curr_y + 46), FONT_BODY, SCALE_LABEL, ACCENT_UI, 1, AA)
        cv2.putText(canvas, "Saves JSON & JPG to scratch/demo_results/", (x + 22, curr_y + 68), FONT_BODY, SCALE_SMALL, TEXT_MUTED, 1, AA)

        # Recording session status
        rec_label = "[R] Recording Active" if recording_session else "[R] Record Session"
        rec_color = TEXT_ERROR if recording_session else TEXT_SECONDARY
        cv2.putText(canvas, rec_label, (x + 295, curr_y + 46), FONT_BODY, SCALE_LABEL, rec_color, 1, AA)
        cv2.putText(canvas, "Continuous telemetry stream log", (x + 295, curr_y + 68), FONT_BODY, SCALE_SMALL, TEXT_MUTED, 1, AA)
