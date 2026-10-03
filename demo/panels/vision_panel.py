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
    FONT_BODY, FONT_HEADER, AA, BG_PANEL, BG_CARD, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ACCENT,
    ACCENT_PURPLE, ACCENT_GREEN, ACCENT_AMBER
)
from demo.ui.widgets import draw_panel, draw_card, draw_metric_row


class VisionPanel:
    """Renders Mode 3: YOLO11n-seg Instance Segmentation View."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.vid_x = self.x + 20
        self.vid_y = self.y + 46
        self.vid_w = self.w - 40
        self.vid_h = self.h - 176

    def render(
        self,
        canvas: np.ndarray,
        camera_image: Optional[np.ndarray],
        telemetry: Dict[str, Any],
    ) -> None:
        """Render the vision camera feed and YOLO detection details."""
        x, y, w, h = self.x, self.y, self.w, self.h
        draw_panel(canvas, x, y, w, h, title="MODE 3: YOLO11n-SEG INSTANCE SEGMENTATION")

        y_obj = telemetry.get("yolo_object") or "NONE"
        y_conf = telemetry.get("yolo_confidence") or 0.0
        y_box = telemetry.get("yolo_box")
        y_center = telemetry.get("yolo_center")
        y_lat = telemetry.get("yolo_latency_ms") or 0.0
        y_running = telemetry.get("yolo_running", False)

        # Draw main camera feed with YOLO boxes
        if camera_image is not None and camera_image.size > 0:
            view = camera_image.copy()
            if y_box is not None:
                x1, y1, x2, y2 = [int(v) for v in y_box]
                cv2.rectangle(view, (x1, y1), (x2, y2), ACCENT_PURPLE, 2)
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

            resized = cv2.resize(view, (self.vid_w, self.vid_h), interpolation=cv2.INTER_LINEAR)
            canvas[self.vid_y : self.vid_y + self.vid_h, self.vid_x : self.vid_x + self.vid_w] = resized
        else:
            cv2.rectangle(
                canvas,
                (self.vid_x, self.vid_y),
                (self.vid_x + self.vid_w, self.vid_y + self.vid_h),
                (20, 16, 14),
                -1,
            )
            cv2.putText(
                canvas,
                "[!] CAMERA FEED UNAVAILABLE",
                (self.vid_x + self.vid_w // 2 - 160, self.vid_y + self.vid_h // 2),
                FONT_BODY,
                0.62,
                ACCENT_AMBER,
                1,
                AA,
            )

        cv2.rectangle(canvas, (self.vid_x, self.vid_y), (self.vid_x + self.vid_w, self.vid_y + self.vid_h), BORDER, 1)

        # Bottom detailed telemetry card
        card_y = self.vid_y + self.vid_h + 10
        card_h = 114
        draw_card(canvas, self.vid_x, card_y, self.vid_w, card_h)

        status_str = "Running" if y_running else "Stopped"
        det_status = "Object Detected" if y_obj != "NONE" else "No Detection"
        det_color = ACCENT_GREEN if y_obj != "NONE" else TEXT_MUTED

        draw_metric_row(canvas, self.vid_x + 16, card_y + 26, "Model Architecture", "Ultralytics YOLO11n-seg (CPU Inference)", TEXT_PRIMARY, width=self.vid_w - 32)
        draw_metric_row(canvas, self.vid_x + 16, card_y + 50, "Inference Status", f"{status_str} ({det_status})", det_color, width=self.vid_w - 32, bold_value=True)

        conf_str = f"{y_conf * 100:.1f}%" if y_conf > 0 else "--"
        draw_metric_row(canvas, self.vid_x + 16, card_y + 74, "Top-1 Class / Confidence", f"{y_obj} ({conf_str})", ACCENT_PURPLE, width=self.vid_w - 32, bold_value=True)

        box_str = f"({int(y_box[0])}, {int(y_box[1])}) -> ({int(y_box[2])}, {int(y_box[3])})" if y_box else "None"
        center_str = f"({int(y_center[0])}, {int(y_center[1])}) px" if y_center else "None"
        draw_metric_row(canvas, self.vid_x + 16, card_y + 98, "BBox & Centroid Coordinates", f"Box: {box_str}  |  Center: {center_str}  |  Latency: {y_lat:.1f} ms", TEXT_SECONDARY, width=self.vid_w - 32)
