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
    FONT_BODY, AA, BG_VISUALIZATION, BORDER_ON_DARK,
    TEXT_SECONDARY, TEXT_ON_DARK, TEXT_SECONDARY_ON_DARK, TEXT_MUTED_ON_DARK,
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED,
    RADIUS_CARD, RADIUS_BUTTON, SPACE_SM, SPACE_MD, SPACE_LG
)
from demo.ui.widgets import (
    draw_button,
    draw_measurement_card,
    draw_mode_title,
    draw_note,
    draw_rounded_rect,
    rounded_rect_contains,
)


class MultimodalPanel:
    """Renders Mode 1: Live Multimodal Camera & Heatmap Fusion."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.preserve_aspect = True
        self._source_size = (640, 480)
        self.set_layout(x, y, w, h, preserve_aspect=True)

    def set_layout(self, x: int, y: int, w: int, h: int, preserve_aspect: bool = False) -> None:
        """Update panel position and recalculate viewport & button geometry."""
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.vid_x = self.x + SPACE_LG
        self.vid_y = self.y + 62
        self.vid_w = max(1, self.w - 2 * SPACE_LG)
        self.vid_h = max(1, self.h - 184)
        self.preserve_aspect = preserve_aspect
        self._image_rect = self._fit_image_rect(*self._source_size)

        self.btn_yolo_w = 128
        self.btn_yolo_h = 28
        self.btn_yolo_x = self.x + self.w - self.btn_yolo_w - 16
        self.btn_yolo_y = self.y + 10
        self.cards_y = self.y + self.h - 110
        self.status_y = self.y + self.h - SPACE_MD

    def _fit_image_rect(self, source_w: int, source_h: int) -> Tuple[int, int, int, int]:
        """Return the displayed image bounds, letterboxing only when requested."""
        self._source_size = (source_w, source_h)
        if not self.preserve_aspect:
            return (self.vid_x, self.vid_y, self.vid_w, self.vid_h)

        scale = min(self.vid_w / source_w, self.vid_h / source_h)
        image_w = max(1, int(round(source_w * scale)))
        image_h = max(1, int(round(source_h * scale)))
        image_x = self.vid_x + (self.vid_w - image_w) // 2
        image_y = self.vid_y + (self.vid_h - image_h) // 2
        return (image_x, image_y, image_w, image_h)

    def handle_yolo_click(self, px: int, py: int) -> bool:
        """Check if click occurred within the YOLO toggle button."""
        return rounded_rect_contains(
            px,
            py,
            self.btn_yolo_x,
            self.btn_yolo_y,
            self.btn_yolo_w,
            self.btn_yolo_h,
            RADIUS_BUTTON,
        )

    @staticmethod
    def _format_measurement(value: Any, unit: str, signed: bool = False) -> str:
        """Format a telemetry value without inventing a reading when unavailable."""
        if value is None:
            return "—"
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "—"
        if not np.isfinite(numeric_value):
            return "—"
        number_format = "+.1f" if signed else ".1f"
        return f"{format(numeric_value, number_format)} {unit}"

    def handle_click(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        """
        If a click occurred inside the video viewport, map the canvas
        coordinate (px, py) to native camera coordinate (0..640, 0..480).
        """
        image_x, image_y, image_w, image_h = self._image_rect
        source_w, source_h = self._source_size
        if (image_x <= px < image_x + image_w) and (image_y <= py < image_y + image_h):
            norm_x = (px - image_x) / float(image_w)
            norm_y = (py - image_y) / float(image_h)
            cam_x = int(np.clip(norm_x * source_w, 0, source_w - 1))
            cam_y = int(np.clip(norm_y * source_h, 0, source_h - 1))
            return (cam_x, cam_y)
        return None

    def render(
        self,
        canvas: np.ndarray,
        overlaid_image: Optional[np.ndarray],
        telemetry: Dict[str, Any],
    ) -> None:
        """Render the Live Fusion workspace, camera view, and live measurements."""
        x, y, w, h = self.x, self.y, self.w, self.h
        yolo_on = telemetry.get("yolo_enabled", True)
        draw_mode_title(canvas, "Live Fusion", x + SPACE_LG, y + 28)
        draw_note(canvas, "Acoustic localization | Visual perception", x + SPACE_LG, y + 48, TEXT_SECONDARY)

        if telemetry.get("profile_mode") != "DIAGNOSTIC":
            draw_button(
                canvas,
                "[Y] YOLO: ON" if yolo_on else "[Y] YOLO: OFF",
                self.btn_yolo_x,
                self.btn_yolo_y,
                self.btn_yolo_w,
                self.btn_yolo_h,
                active=yolo_on,
                border_color=ACCENT_GREEN if yolo_on else ACCENT_RED,
                scale=0.42,
            )

        draw_rounded_rect(canvas, self.vid_x, self.vid_y, self.vid_w, self.vid_h, BG_VISUALIZATION, RADIUS_CARD)
        camera_connected = bool(telemetry.get("camera_connected", False))
        has_frame = overlaid_image is not None and overlaid_image.size > 0
        if camera_connected and has_frame:
            source_h, source_w = overlaid_image.shape[:2]
            self._image_rect = self._fit_image_rect(source_w, source_h)
            image_x, image_y, image_w, image_h = self._image_rect
            resized_video = cv2.resize(overlaid_image, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
            canvas[image_y : image_y + image_h, image_x : image_x + image_w] = resized_video

            az = telemetry.get("azimuth_deg")
            az = 90.0 if az is None else az
            el = telemetry.get("elevation_deg")
            el = -6.0 if el is None else el
            dt = telemetry.get("sync_delta_ms")
            dt = 0.0 if dt is None else dt
            yolo_object = telemetry.get("yolo_object") or "NONE"
            confidence = telemetry.get("yolo_confidence")
            confidence = 0.0 if confidence is None else confidence
            hud_line_1 = f"Acoustic DoA: Az = {az:+.1f} deg, El = {el:+.1f} deg"
            if yolo_on:
                hud_line_2 = f"Sync Delta: {dt:.1f} ms  |  YOLO: {yolo_object} ({confidence * 100:.0f}%)"
            else:
                hud_line_2 = f"Sync Delta: {dt:.1f} ms  |  YOLO: OFF (Inference Disabled)"

            hud_x, hud_y = image_x + 12, image_y + 12
            cv2.rectangle(canvas, (hud_x, hud_y), (hud_x + 428, hud_y + 60), BG_VISUALIZATION, -1)
            cv2.rectangle(canvas, (hud_x, hud_y), (hud_x + 428, hud_y + 60), BORDER_ON_DARK, 1)
            cv2.putText(canvas, hud_line_1, (hud_x + 10, hud_y + 24), FONT_BODY, 0.48, ACCENT_CYAN, 1, AA)
            cv2.putText(canvas, hud_line_2, (hud_x + 10, hud_y + 47), FONT_BODY, 0.46, TEXT_ON_DARK, 1, AA)
        else:
            self._image_rect = self._fit_image_rect(*self._source_size)
            if not camera_connected:
                self._draw_camera_unavailable(canvas)
            else:
                self._draw_waiting_for_frame(canvas)

        draw_rounded_rect(canvas, self.vid_x, self.vid_y, self.vid_w, self.vid_h, BORDER_ON_DARK, RADIUS_CARD, 1)

        card_gap = SPACE_MD
        card_w = (w - 2 * SPACE_LG - 2 * card_gap) // 3
        card_h = 78
        peak_valid = bool(telemetry.get("peak_valid", False))
        sync_valid = (
            camera_connected
            and bool(telemetry.get("audio_connected", False))
            and bool(telemetry.get("camera_timestamp"))
            and bool(telemetry.get("audio_timestamp"))
        )
        cards = (
            ("AZIMUTH", self._format_measurement(telemetry.get("azimuth_deg") if peak_valid else None, "deg", signed=True)),
            ("ELEVATION", self._format_measurement(telemetry.get("elevation_deg") if peak_valid else None, "deg", signed=True)),
            ("SYNC ERROR", self._format_measurement(telemetry.get("sync_delta_ms") if sync_valid else None, "ms")),
        )
        for index, (label, value) in enumerate(cards):
            draw_measurement_card(
                canvas,
                x + SPACE_LG + index * (card_w + card_gap),
                self.cards_y,
                card_w,
                card_h,
                label,
                value,
            )

        status = "Camera connected" if camera_connected else "Camera unavailable | retry active"
        (status_w, _), _ = cv2.getTextSize(status, FONT_BODY, 0.40, 1)
        cv2.putText(
            canvas,
            status,
            (x + (w - status_w) // 2, self.status_y),
            FONT_BODY,
            0.40,
            TEXT_SECONDARY,
            1,
            AA,
        )

    def _draw_camera_unavailable(self, canvas: np.ndarray) -> None:
        """Present the runner's existing disconnected state without its placeholder frame."""
        center_x = self.vid_x + self.vid_w // 2
        center_y = self.vid_y + self.vid_h // 2
        icon_x, icon_y = center_x - 16, center_y - 32
        cv2.rectangle(canvas, (icon_x - 12, icon_y - 8), (icon_x + 12, icon_y + 8), ACCENT_RED, 2)
        cv2.line(canvas, (icon_x + 12, icon_y - 5), (icon_x + 20, icon_y - 10), ACCENT_RED, 2, AA)
        cv2.line(canvas, (icon_x + 12, icon_y + 5), (icon_x + 20, icon_y + 10), ACCENT_RED, 2, AA)
        cv2.line(canvas, (icon_x - 19, icon_y + 16), (icon_x + 19, icon_y - 16), ACCENT_RED, 2, AA)

        heading = "Camera unavailable"
        secondary = "Waiting for camera stream | Retry active"
        (heading_w, _), _ = cv2.getTextSize(heading, FONT_BODY, 0.62, 2)
        (secondary_w, _), _ = cv2.getTextSize(secondary, FONT_BODY, 0.42, 1)
        cv2.putText(canvas, heading, (center_x - heading_w // 2, center_y + 16), FONT_BODY, 0.62, TEXT_ON_DARK, 2, AA)
        cv2.putText(
            canvas,
            secondary,
            (center_x - secondary_w // 2, center_y + 40),
            FONT_BODY,
            0.38,
            TEXT_MUTED_ON_DARK,
            1,
            AA,
        )

    def _draw_waiting_for_frame(self, canvas: np.ndarray) -> None:
        """Show a neutral message if the stream is connected but has no frame yet."""
        text = "Waiting for camera frame"
        (text_w, _), _ = cv2.getTextSize(text, FONT_BODY, 0.52, 1)
        cv2.putText(
            canvas,
            text,
            (self.vid_x + (self.vid_w - text_w) // 2, self.vid_y + self.vid_h // 2),
            FONT_BODY,
            0.52,
            TEXT_SECONDARY_ON_DARK,
            1,
            AA,
        )
