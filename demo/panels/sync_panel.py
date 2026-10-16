"""
demo.panels.sync_panel
======================
Mode 6: Real-Time Synchronization Strip-Chart & Jitter Analysis.
Displays hardware arrival time differences between ESP32-CAM and UMA-16 frames.
Upgraded with high-contrast, dual-stroke, non-aliased typography.
"""

from __future__ import annotations
from typing import Any, Dict
import cv2
import numpy as np

from demo.ui.theme import (
    FONT_BODY, FONT_HEADER, AA, BG_CARD, BG_VISUALIZATION, BORDER, BORDER_ON_DARK,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ON_DARK, TEXT_MUTED_ON_DARK,
    TEXT_ACCENT, TEXT_SUCCESS, TEXT_WARNING, TEXT_ERROR,
    ACCENT_RED, ACCENT_UI
)
from demo.ui.widgets import draw_panel, draw_card, draw_measurement_card, draw_metric_row, draw_section, draw_note


class SynchronizationPanel:
    """Renders Mode 6: Real-Time Synchronization Strip-Chart & Jitter Analysis."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.set_layout(x, y, w, h)

    def set_layout(self, x: int, y: int, w: int, h: int) -> None:
        """Update panel position and chart dimensions."""
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.chart_x = self.x + 40
        self.chart_y = self.y + 60
        self.chart_w = self.w - 80
        self.chart_h = 240

    def render(self, canvas: np.ndarray, telemetry: Dict[str, Any]) -> None:
        """Render the synchronization monitor and strip chart."""
        x, y, w, h = self.x, self.y, self.w, self.h
        draw_panel(canvas, x, y, w, h, title="Synchronization")
        stats = telemetry.get("sync_stats") or {}
        cam_t = telemetry.get("camera_timestamp")
        aud_t = telemetry.get("audio_timestamp")
        sample_count = int(stats.get("count", 0) or 0)
        stats_available = sample_count > 0
        cur_dt = telemetry.get("sync_delta_ms")
        current_available = (
            bool(telemetry.get("camera_connected", False))
            and bool(telemetry.get("audio_connected", False))
            and cam_t is not None
            and aud_t is not None
            and cam_t > 0
            and aud_t > 0
            and cur_dt is not None
        )
        is_synced = bool(telemetry.get("synchronized", False))
        current_status = "Synchronized" if current_available and is_synced else (
            "Outside tolerance" if current_available else "Waiting for paired frames"
        )
        status_color = TEXT_SUCCESS if current_available and is_synced else (
            TEXT_ERROR if current_available else TEXT_MUTED
        )
        median = stats.get("median") if stats_available else None
        max_delta = stats.get("max") if stats_available else None
        pass_rate = stats.get("pass_rate") if stats_available else None

        draw_note(canvas, "Timestamp difference between camera and audio frames | Criterion: under 100 ms", x + 20, y + 52, TEXT_SECONDARY)
        kpi_y = y + 62
        kpi_gap = 10
        kpi_w = (w - 40 - 3 * kpi_gap) // 4
        kpis = (
            ("CURRENT DELTA", f"{cur_dt:.2f} ms" if current_available else "—", status_color),
            ("ROLLING MEDIAN", f"{median:.2f} ms" if median is not None else "—", ACCENT_UI),
            ("MAX DISCREPANCY", f"{max_delta:.2f} ms" if max_delta is not None else "—", TEXT_WARNING if max_delta is not None and max_delta > 100.0 else ACCENT_UI),
            ("PASS RATE", f"{pass_rate:.1f}%" if pass_rate is not None else "—", TEXT_SUCCESS if pass_rate is not None else TEXT_MUTED),
        )
        for index, (label, value, color) in enumerate(kpis):
            draw_measurement_card(
                canvas,
                x + 20 + index * (kpi_w + kpi_gap),
                kpi_y,
                kpi_w,
                78,
                label,
                value,
                color,
            )

        chart_x = x + 24
        chart_y = y + 154
        chart_w = w - 48
        chart_h = min(320, max(220, h - 430))
        draw_card(canvas, chart_x, chart_y, chart_w, chart_h, bg=BG_VISUALIZATION)
        plot_x = chart_x + 210
        plot_w = chart_w - 270
        plot_right = plot_x + plot_w
        tolerance_x = plot_x + int(plot_w * (100.0 / 120.0))
        mid_x = plot_x + int(plot_w * (50.0 / 120.0))
        cv2.line(canvas, (mid_x, chart_y + 34), (mid_x, chart_y + chart_h - 34), BORDER_ON_DARK, 1, AA)
        cv2.line(canvas, (tolerance_x, chart_y + 30), (tolerance_x, chart_y + chart_h - 28), ACCENT_RED, 2, AA)
        cv2.putText(canvas, "100 ms tolerance", (min(tolerance_x + 8, plot_right - 130), chart_y + 22), FONT_HEADER, 0.40, TEXT_ON_DARK, 1, AA)
        cv2.putText(canvas, "0", (plot_x, chart_y + chart_h - 10), FONT_BODY, 0.38, TEXT_MUTED_ON_DARK, 1, AA)
        cv2.putText(canvas, "50 ms", (mid_x - 18, chart_y + chart_h - 10), FONT_BODY, 0.38, TEXT_MUTED_ON_DARK, 1, AA)
        cv2.putText(canvas, "120 ms", (plot_right - 45, chart_y + chart_h - 10), FONT_BODY, 0.38, TEXT_MUTED_ON_DARK, 1, AA)

        rows = (
            ("Current", cur_dt if current_available else None, status_color),
            ("Median", median, ACCENT_UI),
            ("Maximum", max_delta, TEXT_WARNING if max_delta is not None and max_delta > 100.0 else ACCENT_UI),
        )
        row_start = chart_y + 78
        row_gap = 58
        for index, (label, value, color) in enumerate(rows):
            row_y = row_start + index * row_gap
            cv2.putText(canvas, label, (chart_x + 18, row_y + 5), FONT_HEADER, 0.48, TEXT_ON_DARK, 1, AA)
            cv2.line(canvas, (plot_x, row_y), (plot_right, row_y), BORDER_ON_DARK, 1, AA)
            if value is not None:
                marker_x = plot_x + int(plot_w * min(max(float(value), 0.0), 120.0) / 120.0)
                cv2.line(canvas, (plot_x, row_y), (marker_x, row_y), color, 4, AA)
                cv2.drawMarker(canvas, (marker_x, row_y), color, cv2.MARKER_DIAMOND, 12, 2, AA)
                cv2.putText(canvas, f"{value:.2f} ms", (chart_x + 105, row_y + 5), FONT_BODY, 0.44, TEXT_ON_DARK, 1, AA)
            else:
                cv2.line(canvas, (chart_x + 105, row_y), (chart_x + 120, row_y), TEXT_MUTED_ON_DARK, 2, AA)

        details_y = chart_y + chart_h + 14
        details_h = max(160, min(190, y + h - details_y - 12))
        draw_card(canvas, chart_x, details_y, chart_w, details_h)
        draw_section(canvas, "Frame Timing Details", chart_x + 16, details_y + 24, TEXT_ACCENT, thickness=2)
        cam_text = f"{cam_t:.6f} s" if current_available else "--"
        aud_text = f"{aud_t:.6f} s" if current_available else "--"
        delta_text = f"{cur_dt:.3f} ms" if current_available else "--"
        jitter = stats.get("std") if stats_available else None
        jitter_text = f"{jitter:.2f} ms" if jitter is not None else "--"
        draw_metric_row(canvas, chart_x + 16, details_y + 50, "Camera frame timestamp", cam_text, TEXT_PRIMARY if current_available else TEXT_MUTED, width=chart_w - 32)
        draw_metric_row(canvas, chart_x + 16, details_y + 74, "Audio frame timestamp", aud_text, TEXT_PRIMARY if current_available else TEXT_MUTED, width=chart_w - 32)
        draw_metric_row(canvas, chart_x + 16, details_y + 98, "Current timestamp delta", delta_text, status_color, width=chart_w - 32, bold_value=current_available)
        draw_metric_row(canvas, chart_x + 16, details_y + 122, "Rolling jitter (standard deviation)", jitter_text, TEXT_SECONDARY if jitter is not None else TEXT_MUTED, width=chart_w - 32)
        draw_metric_row(canvas, chart_x + 16, details_y + 146, "Current status / samples", f"{current_status}  |  {sample_count} samples", status_color, width=chart_w - 32)
