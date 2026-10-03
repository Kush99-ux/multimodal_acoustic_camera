"""
demo.panels.acoustic_panel
==========================
Mode 2: 2D MUSIC Spatial Spectrum Heatmap view with angular axes and crosshair.
Upgraded with high-contrast, dual-stroke, non-aliased typography.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import cv2
import numpy as np

from demo.ui.theme import (
    FONT_BODY, FONT_HEADER, AA, BG_CARD, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ACCENT,
    ACCENT_CYAN, ACCENT_AMBER
)
from demo.ui.widgets import draw_panel, draw_card


class AcousticPanel:
    """Renders Mode 2: 2D MUSIC Spatial Spectrum Heatmap."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        # Heatmap viewport coordinates
        self.plot_x = self.x + 84
        self.plot_y = self.y + 60
        self.plot_w = self.w - 148
        self.plot_h = self.h - 180

    def render(
        self,
        canvas: np.ndarray,
        heatmap_2d: Optional[np.ndarray],
        telemetry: Dict[str, Any],
    ) -> None:
        """Render the 2D MUSIC spatial spectrum and angular axes."""
        x, y, w, h = self.x, self.y, self.w, self.h
        draw_panel(canvas, x, y, w, h, title="MODE 2: 2D MUSIC SPATIAL SPECTRUM (7,676 ANGULAR POINTS)")

        az = telemetry.get("azimuth_deg")
        az = 90.0 if az is None else az
        el = telemetry.get("elevation_deg")
        el = -6.0 if el is None else el
        peak_db = telemetry.get("peak_db")
        peak_db = 0.0 if peak_db is None else peak_db
        valid = telemetry.get("peak_valid", False)

        # Top summary card
        draw_card(canvas, x + 20, y + 36, w - 40, 42)
        info_str = (
            f"Azimuth: {az:+.1f} deg  |  Elevation: {el:+.1f} deg  |  "
            f"Peak: {peak_db:.1f} dB SPL  |  Search Grid: 101 x 76 (7,676 points)  |  "
            f"Bands: 500--3000 Hz"
        )
        cv2.putText(canvas, info_str, (x + 36, y + 63), FONT_BODY, 0.48, ACCENT_CYAN, 1, AA)

        # Heatmap viewport geometry
        px = x + 95
        py = y + 112
        pw = 780
        ph = 580

        if heatmap_2d is not None and heatmap_2d.size > 0:
            h_oriented = np.flipud(heatmap_2d)
            h_norm = (np.clip(h_oriented, 0.0, 1.0) * 255.0).astype(np.uint8)
            h_color = cv2.applyColorMap(h_norm, cv2.COLORMAP_INFERNO)
            h_resized = cv2.resize(h_color, (pw, ph), interpolation=cv2.INTER_LINEAR)

            canvas[py : py + ph, px : px + pw] = h_resized

            # Draw angular peak marker on heatmap
            norm_az = (az - 65.0) / 50.0
            norm_el = (el - (-18.75)) / 37.5

            peak_px = int(np.clip(px + norm_az * pw, px, px + pw - 1))
            peak_py = int(np.clip(py + (1.0 - norm_el) * ph, py, py + ph - 1))

            cv2.drawMarker(canvas, (peak_px, peak_py), (255, 255, 255), cv2.MARKER_CROSS, 26, 2, AA)
            cv2.circle(canvas, (peak_px, peak_py), 14, (255, 255, 255), 2, AA)
            cv2.putText(
                canvas,
                f"Peak: ({az:.1f} deg, {el:.1f} deg)",
                (peak_px + 14, peak_py - 12),
                FONT_BODY,
                0.48,
                (255, 255, 255),
                1,
                AA,
            )
        else:
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (20, 16, 14), -1)
            cv2.putText(
                canvas,
                "[!] NO ACOUSTIC SPECTRUM COMPUTED (WAITING FOR AUDIO FRAME)",
                (px + pw // 2 - 250, py + ph // 2),
                FONT_BODY,
                0.52,
                ACCENT_AMBER,
                1,
                AA,
            )

        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), BORDER, 1)

        # Draw Grid Axes Labels
        az_ticks = [65.0, 77.5, 90.0, 102.5, 115.0]
        for t_az in az_ticks:
            frac = (t_az - 65.0) / 50.0
            tx = int(px + frac * pw)
            cv2.line(canvas, (tx, py + ph), (tx, py + ph + 6), TEXT_MUTED, 1, AA)
            offset_x = 42 if t_az == 115.0 else (16 if t_az == 65.0 else 32)
            cv2.putText(canvas, f"{t_az:.1f} deg", (tx - offset_x, py + ph + 24), FONT_BODY, 0.46, TEXT_SECONDARY, 1, AA)
        cv2.putText(canvas, "Azimuth Angle (theta)", (px + pw // 2 - 95, py + ph + 50), FONT_HEADER, 0.52, TEXT_ACCENT, 2, AA)

        el_ticks = [18.75, 9.38, 0.0, -9.38, -18.75]
        for t_el in el_ticks:
            frac = (18.75 - t_el) / 37.5
            ty = int(py + frac * ph)
            cv2.line(canvas, (px - 6, ty), (px, ty), TEXT_MUTED, 1, AA)
            cv2.putText(canvas, f"{t_el:+.1f} deg", (px - 82, ty + 5), FONT_BODY, 0.46, TEXT_SECONDARY, 1, AA)
        cv2.putText(canvas, "Elevation (phi)", (px - 82, py - 14), FONT_HEADER, 0.50, TEXT_ACCENT, 2, AA)

        # Colorbar
        cb_x = px + pw + 32
        cb_w = 18
        cb_h = ph
        cv2.rectangle(canvas, (cb_x, py), (cb_x + cb_w, py + cb_h), BORDER, 1)
        grad = np.linspace(255, 0, cb_h, dtype=np.uint8)[:, None]
        grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_INFERNO)
        grad_resized = cv2.resize(grad_color, (cb_w, cb_h))
        canvas[py : py + cb_h, cb_x : cb_x + cb_w] = grad_resized
        cv2.putText(canvas, "1.0", (cb_x + 24, py + 14), FONT_BODY, 0.44, TEXT_SECONDARY, 1, AA)
        cv2.putText(canvas, "0.5", (cb_x + 24, py + cb_h // 2 + 5), FONT_BODY, 0.44, TEXT_SECONDARY, 1, AA)
        cv2.putText(canvas, "0.0", (cb_x + 24, py + cb_h), FONT_BODY, 0.44, TEXT_SECONDARY, 1, AA)
        cv2.putText(canvas, "Power (Norm)", (cb_x - 28, py - 14), FONT_HEADER, 0.46, TEXT_MUTED, 1, AA)
