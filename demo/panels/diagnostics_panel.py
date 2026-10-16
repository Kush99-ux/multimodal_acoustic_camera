"""
demo.panels.diagnostics_panel
=============================
Mode 7: Microphone Array & Sub-Band Spectral Diagnostics.
Displays physical 4x4 microphone geometry with live RMS energy heatmaps
alongside 6 sub-band FFT energy meters (500 Hz to 3000 Hz).
Upgraded with high-contrast, dual-stroke, non-aliased typography.
"""

from __future__ import annotations
from typing import Any, Dict
import cv2
import numpy as np

from demo.ui.theme import (
    FONT_BODY, FONT_HEADER, AA, BG_CARD, BG_VISUALIZATION, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ACCENT,
    ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED, ACCENT_CYAN, ACCENT_UI
)
from demo.ui.widgets import draw_panel, draw_card, draw_metric_row, draw_bar, draw_section
from demo.core.diagnostics import MIC_LAYOUT_GRID


class DiagnosticsPanel:
    """Renders Mode 7: Microphone Array & Sub-Band Spectral Diagnostics."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.set_layout(x, y, w, h)

    def set_layout(self, x: int, y: int, w: int, h: int) -> None:
        """Update panel dimensions."""
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def render(self, canvas: np.ndarray, telemetry: Dict[str, Any]) -> None:
        """Render the 16-mic physical array and 6-band spectrum panels."""
        x, y, w, h = self.x, self.y, self.w, self.h
        draw_panel(canvas, x, y, w, h, title="Array & FFT")

        mid_x = x + (w // 2) - 10
        left_w = mid_x - x - 20
        right_w = (x + w - 20) - mid_x - 10

        # ====================================================
        # LEFT: 16-MICROPHONE PHYSICAL ARRAY ARRANGEMENT
        # ====================================================
        draw_card(canvas, x + 20, y + 42, left_w, h - 64)
        draw_section(canvas, "UMA-16 Physical Microphone Arrangement", x + 36, y + 68, ACCENT_UI, thickness=2)
        cv2.putText(canvas, "Physical layout (outward broadside facing +Y):", (x + 36, y + 88), FONT_BODY, 0.42, TEXT_SECONDARY, 1, AA)

        mic_rms = telemetry.get("mic_rms")
        mic_available = mic_rms is not None and len(mic_rms) >= 16

        # Render 4x4 Grid
        grid_start_x = x + 36
        grid_start_y = y + 106
        cell_w = (left_w - 52) // 4
        cell_h = 76

        for r_idx, row in enumerate(MIC_LAYOUT_GRID):
            for c_idx, mic_num in enumerate(row):
                cx = grid_start_x + c_idx * cell_w
                cy = grid_start_y + r_idx * cell_h

                ch_idx = mic_num - 1
                energy = float(mic_rms[ch_idx]) if mic_available else None

                # Determine card color based on activity
                if energy is None:
                    heat_col = BORDER
                elif energy > 0.65:
                    heat_col = ACCENT_RED
                elif energy > 0.35:
                    heat_col = ACCENT_AMBER
                elif energy > 0.10:
                    heat_col = ACCENT_GREEN
                else:
                    heat_col = BORDER

                draw_card(canvas, cx + 3, cy + 3, cell_w - 9, cell_h - 9, BG_CARD, heat_col)

                cv2.putText(canvas, f"MIC {mic_num}", (cx + 10, cy + 24), FONT_HEADER, 0.46, TEXT_PRIMARY, 1, AA)
                cv2.putText(canvas, f"Ch {ch_idx+1}", (cx + 10, cy + 42), FONT_BODY, 0.40, TEXT_MUTED, 1, AA)

                if energy is None:
                    cv2.putText(canvas, "NO DATA", (cx + 10, cy + 62), FONT_BODY, 0.36, TEXT_MUTED, 1, AA)
                else:
                    draw_bar(canvas, cx + 10, cy + 50, cell_w - 26, 8, energy, color=heat_col)

        # Orientation labels
        cv2.putText(canvas, "<-- Camera Left (MIC 4)", (x + 36, grid_start_y + 4 * cell_h + 22), FONT_HEADER, 0.44, ACCENT_UI, 1, AA)
        cv2.putText(canvas, "Camera Right (MIC 13) -->", (x + left_w - 210, grid_start_y + 4 * cell_h + 22), FONT_HEADER, 0.44, ACCENT_UI, 1, AA)

        # Device Specifications Footer
        spec_y = grid_start_y + 4 * cell_h + 44
        draw_card(canvas, x + 36, spec_y, left_w - 32, 88)
        draw_section(canvas, "Device Hardware Audit", x + 48, spec_y + 20, TEXT_ACCENT, thickness=2)
        draw_metric_row(canvas, x + 48, spec_y + 42, "Hardware Device Name", "miniDSP UMA-16 v2 (USB Audio)", TEXT_PRIMARY, width=left_w - 60, bold_value=True)
        draw_metric_row(canvas, x + 48, spec_y + 60, "Acoustic Sample Rate", "48,000 Hz / 24-bit PCM", TEXT_SECONDARY, width=left_w - 60)
        draw_metric_row(canvas, x + 48, spec_y + 78, "Analysis Window / Hop", "4096 samples (85.3 ms) / 2048 hop", TEXT_SECONDARY, width=left_w - 60)

        # ====================================================
        # RIGHT: 6-BAND SPECTRAL ACTIVITY
        # ====================================================
        draw_card(canvas, mid_x + 10, y + 42, right_w, h - 64)
        draw_section(canvas, "FFT Band Energy", mid_x + 26, y + 68, ACCENT_UI, thickness=2)
        cv2.putText(canvas, "Actual FFT power integrated (+/-250 Hz) across discrete MUSIC bands:", (mid_x + 26, y + 88), FONT_BODY, 0.42, TEXT_SECONDARY, 1, AA)

        band_energies = telemetry.get("band_energies", [])

        bar_start_y = y + 116
        bar_gap = 54
        bar_w = right_w - 180

        if not band_energies:
            cv2.putText(canvas, "Waiting for audio diagnostics", (mid_x + 26, bar_start_y + 26), FONT_BODY, 0.46, TEXT_MUTED, 1, AA)

        for b_idx, (fc, frac, db_val) in enumerate(band_energies):
            by = bar_start_y + b_idx * bar_gap

            # Label
            cv2.putText(canvas, f"{int(fc):4d} Hz", (mid_x + 26, by + 19), FONT_HEADER, 0.48, TEXT_PRIMARY, 1, AA)

            # Energy Meter Bar
            bar_col = ACCENT_CYAN if fc <= 2000 else ACCENT_AMBER
            draw_bar(canvas, mid_x + 106, by + 4, bar_w, 20, frac, color=bar_col)

            # Power dB readout
            cv2.putText(canvas, f"{db_val:+.1f} dB", (mid_x + 116 + bar_w, by + 19), FONT_BODY, 0.46, TEXT_PRIMARY, 1, AA)

            # Band description tag
            if fc == 500:
                tag = "Fundamental voice / fan noise; broad mainlobe"
            elif fc == 1000:
                tag = "Primary target speech / mobile chime resonance"
            elif fc == 1500:
                tag = "Acoustic reflection sensitive; high multipath coupling"
            elif fc == 2000:
                tag = "Transition band; stable azimuth DoA"
            elif fc == 2500:
                tag = "High resolution; secondary reflection lobe"
            else:
                tag = "Upper bound; approaching basic aliasing threshold"

            cv2.putText(canvas, tag, (mid_x + 106, by + 37), FONT_BODY, 0.40, TEXT_MUTED, 1, AA)

        # Spectral note
        note_y = bar_start_y + 6 * bar_gap + 16
        draw_card(canvas, mid_x + 26, note_y, right_w - 48, 86)
        draw_section(canvas, "Algorithmic Multi-Band Normalization", mid_x + 38, note_y + 20, TEXT_ACCENT, thickness=2)
        cv2.putText(canvas, "* Basic spatial aliasing threshold for 42 mm spacing is ~4.08 kHz (c = 343 m/s).", (mid_x + 38, note_y + 40), FONT_BODY, 0.40, TEXT_SECONDARY, 1, AA)
        cv2.putText(canvas, "* Implemented bands (500--3000 Hz) operate completely below this threshold.", (mid_x + 38, note_y + 58), FONT_BODY, 0.40, TEXT_SECONDARY, 1, AA)
        cv2.putText(canvas, "* Each sub-band is min-max normalized prior to broadband accumulation.", (mid_x + 38, note_y + 76), FONT_BODY, 0.40, TEXT_MUTED, 1, AA)
