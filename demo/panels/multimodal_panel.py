"""
demo.panels.multimodal_panel
============================
Mode 1: Real-time fused camera feed with acoustic heatmap overlay,
YOLO11n-seg instance segmentation, and interactive ground-truth click.
Upgraded with high-contrast, dual-stroke, non-aliased typography.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from demo.ui.theme import (
    FONT_BODY, AA, BG_PANEL, BG_CARD, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ACCENT,
    ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER
)
from demo.ui.widgets import draw_panel


class MultimodalPanel:
    """Renders Mode 1: Live Multimodal Camera & Heatmap Fusion."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        # Video viewport bounds inside the panel with proper top clearance
        self.vid_x = self.x + 20
        self.vid_y = self.y + 54
        self.vid_w = self.w - 40
        self.vid_h = self.h - 74

        # YOLO toggle button placed on title bar
        self.btn_yolo_w = 120
        self.btn_yolo_h = 24
        self.btn_yolo_x = self.x + self.w - self.btn_yolo_w - 16
        self.btn_yolo_y = self.y + 4

    def handle_yolo_click(self, px: int, py: int) -> bool:
        """Check if click occurred within the YOLO toggle button."""
        return (
            self.btn_yolo_x <= px <= self.btn_yolo_x + self.btn_yolo_w
            and self.btn_yolo_y <= py <= self.btn_yolo_y + self.btn_yolo_h
        )

    def handle_click(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        """
        If a click occurred inside the video viewport, map the canvas
        coordinate (px, py) to native camera coordinate (0..640, 0..480).
        """
        if (self.vid_x <= px <= self.vid_x + self.vid_w) and (self.vid_y <= py <= self.vid_y + self.vid_h):
            norm_x = (px - self.vid_x) / float(self.vid_w)
            norm_y = (py - self.vid_y) / float(self.vid_h)
            cam_x = int(np.clip(norm_x * 640.0, 0, 639))
            cam_y = int(np.clip(norm_y * 480.0, 0, 479))
            return (cam_x, cam_y)
        return None

    def render(
        self,
        canvas: np.ndarray,
        overlaid_image: Optional[np.ndarray],
        telemetry: Dict[str, Any],
    ) -> None:
        """Render the fused video frame and status header into canvas."""
        x, y, w, h = self.x, self.y, self.w, self.h
        draw_panel(canvas, x, y, w, h, title="MODE 1: LIVE MULTIMODAL ACOUSTIC-VISION FUSION")

        # YOLO ON/OFF toggle button on panel title bar
        yolo_on = telemetry.get("yolo_enabled", True)
        btn_bg = (18, 38, 22) if yolo_on else (38, 18, 18)
        btn_border = ACCENT_GREEN if yolo_on else ACCENT_RED
        btn_txt = "[ YOLO: ON ]" if yolo_on else "[ YOLO: OFF ]"
        btn_col = ACCENT_GREEN if yolo_on else ACCENT_RED

        cv2.rectangle(
            canvas,
            (self.btn_yolo_x, self.btn_yolo_y),
            (self.btn_yolo_x + self.btn_yolo_w, self.btn_yolo_y + self.btn_yolo_h),
            btn_bg,
            -1,
        )
        cv2.rectangle(
            canvas,
            (self.btn_yolo_x, self.btn_yolo_y),
            (self.btn_yolo_x + self.btn_yolo_w, self.btn_yolo_y + self.btn_yolo_h),
            btn_border,
            1,
        )
        (tw, th), _ = cv2.getTextSize(btn_txt, FONT_BODY, 0.44, 1)
        tx = self.btn_yolo_x + (self.btn_yolo_w - tw) // 2
        ty = self.btn_yolo_y + (self.btn_yolo_h + th) // 2 - 1
        cv2.putText(canvas, btn_txt, (tx, ty), FONT_BODY, 0.44, btn_col, 1, AA)

        # Sub-header instructions with clear separation from title bar
        info_txt = "Interactive: Click video for Ground-Truth  |  [Y]: Toggle YOLO  |  [C]: Clear Target"
        cv2.putText(canvas, info_txt, (x + 20, y + 46), FONT_BODY, 0.44, TEXT_SECONDARY, 1, AA)

        if overlaid_image is not None and overlaid_image.size > 0:
            resized_video = cv2.resize(overlaid_image, (self.vid_w, self.vid_h), interpolation=cv2.INTER_LINEAR)
            canvas[self.vid_y : self.vid_y + self.vid_h, self.vid_x : self.vid_x + self.vid_w] = resized_video
        else:
            # Placeholder dark screen
            cv2.rectangle(
                canvas,
                (self.vid_x, self.vid_y),
                (self.vid_x + self.vid_w, self.vid_y + self.vid_h),
                (20, 16, 14),
                -1,
            )
            cv2.putText(
                canvas,
                "[!] NO CAMERA FRAME AVAILABLE",
                (self.vid_x + self.vid_w // 2 - 180, self.vid_y + self.vid_h // 2),
                FONT_BODY,
                0.65,
                ACCENT_AMBER,
                2,
                AA,
            )

        # Draw clean border around viewport
        cv2.rectangle(
            canvas,
            (self.vid_x, self.vid_y),
            (self.vid_x + self.vid_w, self.vid_y + self.vid_h),
            BORDER,
            1,
        )

        # On-viewport live telemetry overlay pills
        az = telemetry.get("azimuth_deg")
        az = 90.0 if az is None else az
        el = telemetry.get("elevation_deg")
        el = -6.0 if el is None else el
        dt = telemetry.get("sync_delta_ms")
        dt = 0.0 if dt is None else dt
        y_obj = telemetry.get("yolo_object") or "NONE"
        conf = telemetry.get("yolo_confidence")
        conf = 0.0 if conf is None else conf

        # Top-left HUD info box with clean padding
        hud_txt1 = f"Acoustic DoA: Az = {az:+.1f} deg, El = {el:+.1f} deg"
        if yolo_on:
            hud_txt2 = f"Sync Delta: {dt:.1f} ms  |  YOLO: {y_obj} ({conf*100:.0f}%)"
        else:
            hud_txt2 = f"Sync Delta: {dt:.1f} ms  |  YOLO: OFF (Inference Disabled)"

        cv2.rectangle(canvas, (self.vid_x + 12, self.vid_y + 12), (self.vid_x + 440, self.vid_y + 72), (24, 18, 15), -1)
        cv2.rectangle(canvas, (self.vid_x + 12, self.vid_y + 12), (self.vid_x + 440, self.vid_y + 72), (80, 68, 55), 1)
        cv2.putText(canvas, hud_txt1, (self.vid_x + 22, self.vid_y + 36), FONT_BODY, 0.48, ACCENT_CYAN, 1, AA)
        cv2.putText(canvas, hud_txt2, (self.vid_x + 22, self.vid_y + 59), FONT_BODY, 0.46, TEXT_PRIMARY, 1, AA)
