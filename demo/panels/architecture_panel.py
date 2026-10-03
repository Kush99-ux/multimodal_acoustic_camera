"""
demo.panels.architecture_panel
==============================
Mode 9: System Architecture Flow Diagram.
Visualizes the concurrent dual-branch data flow:
Acoustic Array Branch (UMA-16 -> CSM -> MUSIC) and
Optical Vision Branch (ESP32-CAM -> YOLO11n-seg),
converging at the Synchronizer and Projection Engine.
Upgraded with high-contrast, dual-stroke, non-aliased typography.
"""

from __future__ import annotations
from typing import Any, Dict
import cv2
import numpy as np

from demo.ui.theme import (
    FONT_BODY, FONT_HEADER, AA, BG_CARD, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ACCENT,
    ACCENT_BLUE, ACCENT_PURPLE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_CYAN
)
from demo.ui.widgets import draw_panel, draw_card


class ArchitecturePanel:
    """Renders Mode 9: System Architecture Flow Diagram."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def render(self, canvas: np.ndarray, telemetry: Dict[str, Any]) -> None:
        """Render the system architecture visual block diagram."""
        x, y, w, h = self.x, self.y, self.w, self.h
        draw_panel(canvas, x, y, w, h, title="MODE 9: MULTIMODAL SYSTEM ARCHITECTURE & DATA FLOW")

        # Top subtitle
        cv2.putText(canvas, "Concurrent Sensing Pipeline: Acoustic Array & Optical Vision Branches", (x + 30, y + 42), FONT_HEADER, 0.50, TEXT_ACCENT, 2, AA)

        # ----------------------------------------------------
        # Left Branch: Acoustic Signal Processing
        # ----------------------------------------------------
        bx_w = 290
        bx_h = 58
        ac_x = x + 50

        nodes_acoustic = [
            ("miniDSP UMA-16 v2", "16 MEMS Mics @ 48 kHz (42mm Spacing)", ACCENT_BLUE),
            ("Audio Driver (PortAudio)", "4096 Window, 2048 Hop (Non-blocking)", ACCENT_BLUE),
            ("Sub-Band STFT & CSM", "6 Bands (500--3000 Hz) EMA alpha=0.30", ACCENT_BLUE),
            ("Steering Matrix Projection", "Acoular Equivalence (7,676 Grid Points)", ACCENT_BLUE),
            ("MUSIC Noise Decomposition", "Eigen Minor Subspace, Sigmoid Gate", ACCENT_BLUE),
            ("AcousticFrame Object", "Azimuth theta, Elevation phi, 2D Heatmap", ACCENT_CYAN),
        ]

        curr_y = y + 64
        for idx, (title, sub, col) in enumerate(nodes_acoustic):
            draw_card(canvas, ac_x, curr_y, bx_w, bx_h, bg=(28, 22, 18))
            cv2.rectangle(canvas, (ac_x, curr_y), (ac_x + bx_w, curr_y + bx_h), col, 1)
            cv2.putText(canvas, title, (ac_x + 12, curr_y + 24), FONT_HEADER, 0.46, col, 1, AA)
            cv2.putText(canvas, sub, (ac_x + 12, curr_y + 45), FONT_BODY, 0.40, TEXT_SECONDARY, 1, AA)

            if idx < len(nodes_acoustic) - 1:
                cv2.arrowedLine(canvas, (ac_x + bx_w // 2, curr_y + bx_h), (ac_x + bx_w // 2, curr_y + bx_h + 16), col, 1, tipLength=0.25)
                curr_y += bx_h + 16

        # ----------------------------------------------------
        # Right Branch: Computer Vision Pipeline
        # ----------------------------------------------------
        vi_x = x + w - 50 - bx_w
        nodes_vision = [
            ("ESP32-CAM AI Thinker", "Omnivision OV2640 (640x480 MJPEG Stream)", ACCENT_PURPLE),
            ("Camera Stream Ingestion", "Thread-isolated latest-frame socket drain", ACCENT_PURPLE),
            ("Optical Preprocessing", "Color space normalization & BGR matrix", ACCENT_PURPLE),
            ("YOLO11n-seg Engine", "Instance Segmentation (CPU Inference)", ACCENT_PURPLE),
            ("Bounding Boxes & Centroids", "Class Label, Confidence, Centroid (cx, cy)", ACCENT_PURPLE),
            ("VisionFrame Object", "Timestamp, Frame ID, Detections list", ACCENT_CYAN),
        ]

        curr_y = y + 64
        for idx, (title, sub, col) in enumerate(nodes_vision):
            draw_card(canvas, vi_x, curr_y, bx_w, bx_h, bg=(28, 22, 18))
            cv2.rectangle(canvas, (vi_x, curr_y), (vi_x + bx_w, curr_y + bx_h), col, 1)
            cv2.putText(canvas, title, (vi_x + 12, curr_y + 24), FONT_HEADER, 0.46, col, 1, AA)
            cv2.putText(canvas, sub, (vi_x + 12, curr_y + 45), FONT_BODY, 0.40, TEXT_SECONDARY, 1, AA)

            if idx < len(nodes_vision) - 1:
                cv2.arrowedLine(canvas, (vi_x + bx_w // 2, curr_y + bx_h), (vi_x + bx_w // 2, curr_y + bx_h + 16), col, 1, tipLength=0.25)
                curr_y += bx_h + 16

        # ----------------------------------------------------
        # Center: Convergence into Synchronization & Fusion
        # ----------------------------------------------------
        mid_cx = x + w // 2
        fuse_w = 270
        fuse_x = mid_cx - fuse_w // 2
        fuse_y = y + 215

        # Sync Box
        draw_card(canvas, fuse_x, fuse_y, fuse_w, 74, bg=(34, 26, 20))
        cv2.rectangle(canvas, (fuse_x, fuse_y), (fuse_x + fuse_w, fuse_y + 74), ACCENT_GREEN, 2)
        cv2.putText(canvas, "Timestamp Synchronizer", (fuse_x + 16, fuse_y + 24), FONT_HEADER, 0.46, ACCENT_GREEN, 2, AA)
        cv2.putText(canvas, "Queue-based Monotonic Matcher", (fuse_x + 16, fuse_y + 45), FONT_BODY, 0.40, TEXT_PRIMARY, 1, AA)
        cv2.putText(canvas, "|T_cam - T_audio| < 100.0 ms", (fuse_x + 16, fuse_y + 63), FONT_BODY, 0.40, ACCENT_CYAN, 1, AA)

        # Arrows converging from branches into Synchronizer
        ac_bottom_y = y + 64 + 5 * (bx_h + 16) + bx_h // 2
        cv2.line(canvas, (ac_x + bx_w, ac_bottom_y), (mid_cx - 150, ac_bottom_y), ACCENT_CYAN, 1, AA)
        cv2.arrowedLine(canvas, (mid_cx - 150, ac_bottom_y), (fuse_x, fuse_y + 37), ACCENT_CYAN, 1, tipLength=0.15)

        vi_bottom_y = ac_bottom_y
        cv2.line(canvas, (vi_x, vi_bottom_y), (mid_cx + 150, vi_bottom_y), ACCENT_CYAN, 1, AA)
        cv2.arrowedLine(canvas, (mid_cx + 150, vi_bottom_y), (fuse_x + fuse_w, fuse_y + 37), ACCENT_CYAN, 1, tipLength=0.15)

        # Multimodal Fusion Engine Box
        eng_y = fuse_y + 98
        draw_card(canvas, fuse_x, eng_y, fuse_w, 74, bg=(34, 26, 20))
        cv2.rectangle(canvas, (fuse_x, eng_y), (fuse_x + fuse_w, eng_y + 74), ACCENT_ORANGE, 2)
        cv2.putText(canvas, "Multimodal Fusion Engine", (fuse_x + 14, eng_y + 24), FONT_HEADER, 0.46, ACCENT_ORANGE, 2, AA)
        cv2.putText(canvas, "Pinhole Spatial Projection", (fuse_x + 14, eng_y + 45), FONT_BODY, 0.40, TEXT_PRIMARY, 1, AA)
        cv2.putText(canvas, "(theta, phi) -> (u, v) Image Space", (fuse_x + 14, eng_y + 63), FONT_BODY, 0.40, ACCENT_CYAN, 1, AA)

        cv2.arrowedLine(canvas, (mid_cx, fuse_y + 74), (mid_cx, eng_y), ACCENT_GREEN, 1, tipLength=0.20)

        # Output Overlay Render Box
        ren_y = eng_y + 98
        draw_card(canvas, fuse_x, ren_y, fuse_w, 66, bg=(34, 26, 20))
        cv2.rectangle(canvas, (fuse_x, ren_y), (fuse_x + fuse_w, ren_y + 66), ACCENT_CYAN, 2)
        cv2.putText(canvas, "Live Heatmap Overlay", (fuse_x + 28, ren_y + 26), FONT_HEADER, 0.46, ACCENT_CYAN, 2, AA)
        cv2.putText(canvas, "OpenCV Colormap Blending & Marker", (fuse_x + 16, ren_y + 48), FONT_BODY, 0.40, TEXT_MUTED, 1, AA)

        cv2.arrowedLine(canvas, (mid_cx, eng_y + 74), (mid_cx, ren_y), ACCENT_ORANGE, 1, tipLength=0.20)
