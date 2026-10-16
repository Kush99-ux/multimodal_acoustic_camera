"""
demo.panels.vision_panel
========================
Mode 3: Dedicated Vision & YOLO11n-seg Instance Segmentation view.
Displays the clean camera feed with segmentation masks, bounding boxes,
and detailed object detection metrics.
Upgraded with high-contrast, dual-stroke, non-aliased typography.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import cv2
import numpy as np

from demo.ui.theme import (
    FONT_BODY, AA, BG_VISUALIZATION, BORDER_ON_DARK,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ON_DARK, TEXT_SECONDARY_ON_DARK,
    TEXT_SUCCESS, TEXT_WARNING,
    ACCENT_PURPLE, ACCENT_AMBER
)
from demo.ui.widgets import draw_panel, draw_card, draw_metric_row, draw_note, draw_rounded_rect


class VisionPanel:
    """Renders Mode 3: YOLO11n-seg Instance Segmentation View."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.preserve_aspect = False
        self.set_layout(x, y, w, h)

    def set_layout(self, x: int, y: int, w: int, h: int, preserve_aspect: bool = False) -> None:
        """Update panel position and dimensions."""
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.vid_x = self.x + 20
        self.vid_y = self.y + 64
        self.vid_w = max(1, self.w - 40)
        self.vid_h = max(1, self.h - 194)
        self.preserve_aspect = preserve_aspect

    def _fit_image_rect(self, source_w: int, source_h: int) -> tuple[int, int, int, int]:
        """Return the displayed image bounds, letterboxing only when requested."""
        if not self.preserve_aspect:
            return (self.vid_x, self.vid_y, self.vid_w, self.vid_h)

        scale = min(self.vid_w / source_w, self.vid_h / source_h)
        image_w = max(1, int(round(source_w * scale)))
        image_h = max(1, int(round(source_h * scale)))
        image_x = self.vid_x + (self.vid_w - image_w) // 2
        image_y = self.vid_y + (self.vid_h - image_h) // 2
        return (image_x, image_y, image_w, image_h)

    def render(
        self,
        canvas: np.ndarray,
        camera_image: Optional[np.ndarray],
        telemetry: Dict[str, Any],
    ) -> None:
        """Render the vision camera feed and YOLO detection details."""
        x, y, w, h = self.x, self.y, self.w, self.h
        draw_panel(canvas, x, y, w, h, title="Vision")

        y_obj = telemetry.get("yolo_object")
        y_conf = telemetry.get("yolo_confidence")
        y_box = telemetry.get("yolo_box")
        y_center = telemetry.get("yolo_center")
        y_lat = telemetry.get("yolo_latency_ms")
        y_running = bool(telemetry.get("yolo_running", False))
        yolo_enabled = bool(telemetry.get("yolo_enabled", True))
        camera_connected = bool(telemetry.get("camera_connected", False))

        camera_status = "Camera connected" if camera_connected else "Camera unavailable"
        yolo_status = "YOLO enabled" if yolo_enabled else "YOLO disabled"
        draw_note(canvas, f"{camera_status}  |  {yolo_status}", x + 20, y + 53, TEXT_SECONDARY)

        # Draw main camera feed with YOLO boxes
        has_frame = camera_image is not None and camera_image.size > 0
        if camera_connected and has_frame:
            source_h, source_w = camera_image.shape[:2]
            view = camera_image.copy()
            if y_box is not None:
                x1, y1, x2, y2 = [int(v) for v in y_box]
                cv2.rectangle(view, (x1, y1), (x2, y2), ACCENT_PURPLE, 2)
                if y_center is not None:
                    cv2.circle(view, (int(y_center[0]), int(y_center[1])), 5, ACCENT_PURPLE, -1)
                cv2.putText(
                    view,
                    f"{y_obj} {y_conf:.2f}",
                    (x1, max(22, y1 - 8)),
                    FONT_BODY,
                    0.52,
                    ACCENT_PURPLE,
                    1,
                    AA,
                )

            image_x, image_y, image_w, image_h = self._fit_image_rect(source_w, source_h)
            draw_rounded_rect(canvas, self.vid_x, self.vid_y, self.vid_w, self.vid_h, BG_VISUALIZATION)
            resized = cv2.resize(view, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
            canvas[image_y : image_y + image_h, image_x : image_x + image_w] = resized
        else:
            draw_rounded_rect(canvas, self.vid_x, self.vid_y, self.vid_w, self.vid_h, BG_VISUALIZATION)
            center_x = self.vid_x + self.vid_w // 2
            center_y = self.vid_y + self.vid_h // 2
            if not camera_connected:
                icon_x, icon_y = center_x - 16, center_y - 25
                cv2.rectangle(canvas, (icon_x - 12, icon_y - 8), (icon_x + 12, icon_y + 8), ACCENT_AMBER, 2)
                cv2.line(canvas, (icon_x - 17, icon_y + 14), (icon_x + 17, icon_y - 14), ACCENT_AMBER, 2, AA)
                heading, secondary = "Camera unavailable", "Waiting for camera stream | Retry active"
                (heading_w, _), _ = cv2.getTextSize(heading, FONT_BODY, 0.62, 2)
                (secondary_w, _), _ = cv2.getTextSize(secondary, FONT_BODY, 0.40, 1)
                cv2.putText(canvas, heading, (center_x - heading_w // 2, center_y + 10), FONT_BODY, 0.62, TEXT_ON_DARK, 2, AA)
                cv2.putText(canvas, secondary, (center_x - secondary_w // 2, center_y + 34), FONT_BODY, 0.40, TEXT_SECONDARY_ON_DARK, 1, AA)
            else:
                waiting_text = "Waiting for camera frame"
                (waiting_w, _), _ = cv2.getTextSize(waiting_text, FONT_BODY, 0.52, 1)
                cv2.putText(canvas, waiting_text, (center_x - waiting_w // 2, center_y), FONT_BODY, 0.52, TEXT_SECONDARY_ON_DARK, 1, AA)

        draw_rounded_rect(canvas, self.vid_x, self.vid_y, self.vid_w, self.vid_h, BORDER_ON_DARK, thickness=1)

        # Keep the unavailable/no-detection state compact; expand only when
        # there are actual detection details to show.
        card_y = self.vid_y + self.vid_h + 10
        has_detection = (
            camera_connected
            and yolo_enabled
            and y_running
            and y_obj is not None
            and y_obj != "NONE"
        )
        card_h = 114 if has_detection else 58
        draw_card(canvas, self.vid_x, card_y, self.vid_w, card_h)

        if not camera_connected:
            detection_status, detection_color = "Camera unavailable", TEXT_MUTED
        elif not yolo_enabled:
            detection_status, detection_color = "YOLO disabled", TEXT_MUTED
        elif not y_running:
            detection_status, detection_color = "Inference unavailable", TEXT_WARNING
        elif y_obj and y_obj != "NONE":
            detection_status, detection_color = f"{y_obj} detected", TEXT_SUCCESS
        else:
            detection_status, detection_color = "No detection", TEXT_SECONDARY

        class_confidence = "--"
        if has_detection and y_conf is not None:
            class_confidence = f"{y_obj} ({float(y_conf) * 100:.1f}%)"
        centroid = f"({int(y_center[0])}, {int(y_center[1])}) px" if y_center is not None else "--"
        box_str = f"({int(y_box[0])}, {int(y_box[1])}) to ({int(y_box[2])}, {int(y_box[3])})" if y_box is not None else "--"
        latency = f"{float(y_lat):.1f} ms" if yolo_enabled and y_running and camera_connected and y_lat is not None else "--"

        draw_metric_row(canvas, self.vid_x + 16, card_y + 38, "Detection status", detection_status, detection_color, width=self.vid_w - 32, bold_value=True)
        if has_detection:
            draw_metric_row(canvas, self.vid_x + 16, card_y + 62, "Class / confidence", class_confidence, TEXT_PRIMARY if class_confidence != "--" else TEXT_MUTED, width=self.vid_w - 32)
            draw_metric_row(canvas, self.vid_x + 16, card_y + 86, "Centroid / bounding box", f"{centroid}  |  {box_str}", TEXT_SECONDARY, width=self.vid_w - 32)
            draw_metric_row(canvas, self.vid_x + 16, card_y + 110, "Inference latency", latency, TEXT_SECONDARY, width=self.vid_w - 32)
