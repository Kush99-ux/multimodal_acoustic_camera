"""
demo.panels.validation_panel
============================
Mode 8: Scientific Validation & Historical Experiments Table.
Presents the verified benchmark results: Synthetic 12/12 test,
Acoular equivalence proof, and representative physical test runs.
Engineered for high readability and crisp typography on a 1600x900 display.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from demo.ui.theme import (
    FONT_BODY, FONT_HEADER, AA, BG_CARD, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ACCENT,
    TEXT_SUCCESS, TEXT_WARNING, TEXT_ERROR,
    ACCENT_UI,
    SCALE_LABEL, SCALE_VALUE, SCALE_SMALL, SCALE_TABLE_HEADER, SCALE_TABLE_VALUE
)
from demo.ui.widgets import (
    draw_panel, draw_card, draw_section, draw_metric_row, draw_note
)
from demo.data.models import ExperimentRecord, DatabaseStats
from demo.data.repository import ExperimentRepository
from demo.data.migrations import migrate_historical_records


class ValidationPanel:
    """Renders Mode 8: Scientific Validation & Historical Experiments Table."""

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        repository: Optional[ExperimentRepository] = None,
    ) -> None:
        self.set_layout(x, y, w, h)
        if repository is None:
            self.repository = ExperimentRepository()
            migrate_historical_records(self.repository)
        else:
            self.repository = repository

        # Query cache to throttle SQLite disk I/O (initialized once in __init__)
        self._cached_recent_experiments: List[ExperimentRecord] = []
        self._cached_stats: DatabaseStats = DatabaseStats()
        self._last_query_time: float = 0.0
        self._cache_interval_sec: float = 1.0

    def set_layout(self, x: int, y: int, w: int, h: int) -> None:
        """Update panel dimensions."""
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def render(self, canvas: np.ndarray, telemetry: Dict[str, Any]) -> None:
        """Render the scientific validation benchmarks and experimental runs."""
        x, y, w, h = self.x, self.y, self.w, self.h
        draw_panel(canvas, x, y, w, h, title="MODE 8: SCIENTIFIC BENCHMARK RESULTS & HISTORICAL EXPERIMENTS")

        # Top 3 Benchmark Summary Cards
        card_w = (w - 70) // 3
        card_h = 138
        c_y = y + 42

        # Card 1: Synthetic MUSIC Benchmark
        draw_card(canvas, x + 20, c_y, card_w, card_h)
        draw_section(canvas, "Synthetic MUSIC Benchmark", x + 32, c_y + 22, ACCENT_UI, thickness=2)
        draw_metric_row(canvas, x + 32, c_y + 48, "Simulated Cases Passed", "12 / 12 PASS", TEXT_SUCCESS, width=card_w - 24, bold_value=True)
        draw_metric_row(canvas, x + 32, c_y + 72, "Max Discretization Error", "<= 0.25 deg", ACCENT_UI, width=card_w - 24, bold_value=True)
        draw_metric_row(canvas, x + 32, c_y + 96, "Grid Angular Step", "0.50 deg", TEXT_PRIMARY, width=card_w - 24)
        draw_note(canvas, "* Tested synthetic conditions (free-field)", x + 32, c_y + 122, TEXT_MUTED)

        # Card 2: Acoular Steering Equivalence
        c2_x = x + 20 + card_w + 15
        draw_card(canvas, c2_x, c_y, card_w, card_h)
        draw_section(canvas, "Steering Matrix Equivalence", c2_x + 12, c_y + 22, ACCENT_UI, thickness=2)
        draw_metric_row(canvas, c2_x + 12, c_y + 48, "Max Steering Matrix Diff", "0.0 (Exact)", TEXT_SUCCESS, width=card_w - 24, bold_value=True)
        draw_metric_row(canvas, c2_x + 12, c_y + 72, "Max Heatmap Diff", "6.88e-15", TEXT_SUCCESS, width=card_w - 24, bold_value=True)
        draw_metric_row(canvas, c2_x + 12, c_y + 96, "Numerical Baseline", "Acoular Reference", TEXT_PRIMARY, width=card_w - 24)
        draw_note(canvas, "* Mathematically identical to golden model", c2_x + 12, c_y + 122, TEXT_MUTED)

        # Card 3: Multimodal Synchronization
        c3_x = x + 20 + (card_w + 15) * 2
        draw_card(canvas, c3_x, c_y, card_w, card_h)
        draw_section(canvas, "Integrated Synchronization", c3_x + 12, c_y + 22, TEXT_SUCCESS, thickness=2)
        draw_metric_row(canvas, c3_x + 12, c_y + 48, "Validation Campaign", "20 Consecutive Frames", TEXT_PRIMARY, width=card_w - 24)
        draw_metric_row(canvas, c3_x + 12, c_y + 72, "Median Timestamp Delta", "28.52 ms", TEXT_SUCCESS, width=card_w - 24, bold_value=True)
        draw_metric_row(canvas, c3_x + 12, c_y + 96, "Max Discrepancy", "49.47 ms (< 100 ms)", ACCENT_UI, width=card_w - 24, bold_value=True)
        draw_note(canvas, "* 100% of frames within criterion limit", c3_x + 12, c_y + 122, TEXT_MUTED)

        # ----------------------------------------------------
        # Middle: Dynamic Experiments Table (Read from SQLite Database)
        # ----------------------------------------------------
        t_y = c_y + card_h + 14
        t_h = 434
        draw_card(canvas, x + 20, t_y, w - 40, t_h)

        # Query dynamic records and statistics from SQLite with 1.0s caching
        now = time.time()
        if self.repository is not None:
            if (now - self._last_query_time) >= self._cache_interval_sec or not self._cached_recent_experiments:
                try:
                    self._cached_recent_experiments = self.repository.get_recent_experiments(limit=11)
                    self._cached_stats = self.repository.get_database_stats()
                    self._last_query_time = now
                except Exception as e:
                    print(f"[ValidationPanel] Database query error: {e}")

        recent_experiments = self._cached_recent_experiments
        stats = self._cached_stats

        # Section header with live database aggregate counts
        draw_section(canvas, "Recent Validation Trials & Historical Campaigns", x + 34, t_y + 24, TEXT_ACCENT, thickness=2)
        stats_str = (
            f"DB: {stats.total_experiments} total  |  "
            f"H: {stats.horizontal_count}  |  "
            f"V: {stats.vertical_count}  |  "
            f"Completed: {stats.completed_count}  |  "
            f"Pass: {stats.pass_count}  |  "
            f"Flagged/Obs: {stats.flagged_failed_count}"
        )
        cv2.putText(canvas, stats_str, (x + 34, t_y + 46), FONT_BODY, 0.38, TEXT_SECONDARY, 1, AA)

        headers = ["Run", "Experimental Condition / Target", "Frames", "Median Az", "Median El", "Az Std", "El Std", "Pixel Error", "Sync / Status"]
        column_fractions = (0.0, 0.14, 0.36, 0.425, 0.49, 0.555, 0.62, 0.69, 0.78)
        table_width = w - 68
        cols = [x + 34 + int(table_width * fraction) for fraction in column_fractions]

        for i, th in enumerate(headers):
            cv2.putText(canvas, th, (cols[i], t_y + 75), FONT_HEADER, 0.40, TEXT_SECONDARY, 1, AA)
        cv2.line(canvas, (x + 34, t_y + 84), (x + w - 34, t_y + 84), BORDER, 1)

        row_y = t_y + 108
        for r_idx, exp in enumerate(recent_experiments):
            # Alternating background row highlight for maximum legibility
            if r_idx % 2 == 1:
                cv2.rectangle(canvas, (x + 24, row_y - 18), (x + w - 24, row_y + 9), BG_CARD, -1)

            # Build formatted row columns
            run_str = exp.run_id
            cond_str = exp.notes if exp.notes else f"{exp.test_type.capitalize()} {exp.target_label}"
            # Truncate condition string if necessary to prevent overflow
            if len(cond_str) > 31:
                cond_str = cond_str[:28] + "..."

            frames_str = str(exp.frame_count)
            az_str = f"{exp.median_azimuth:.1f} deg" if exp.median_azimuth is not None else "--"
            el_str = f"{exp.median_elevation:+.1f} deg" if exp.median_elevation is not None else "--"
            az_std_str = f"{exp.azimuth_std:.2f} deg" if exp.azimuth_std is not None else "--"
            el_std_str = f"{exp.elevation_std:.2f} deg" if exp.elevation_std is not None else "--"
            px_str = f"{exp.pixel_error:.1f} px" if exp.pixel_error is not None else "--"
            sync_pct = exp.sync_percentage
            sync_str = f"{sync_pct:.0f}% {exp.status}" if sync_pct is not None and exp.status else (
                f"{sync_pct:.0f}%" if sync_pct is not None else (exp.status or "—")
            )

            row_data = [
                (run_str, ACCENT_UI, 2),
                (cond_str, TEXT_PRIMARY, 1),
                (frames_str, TEXT_SECONDARY, 1),
                (az_str, TEXT_PRIMARY, 1),
                (el_str, TEXT_PRIMARY, 1),
                (az_std_str, TEXT_PRIMARY, 1),
                (el_std_str, TEXT_PRIMARY, 1),
                (px_str, TEXT_SUCCESS if exp.pixel_error is not None and exp.pixel_error <= 100 else (TEXT_WARNING if exp.pixel_error is not None else TEXT_MUTED), 2 if exp.pixel_error is not None else 1),
                (sync_str, TEXT_SUCCESS if exp.status == "PASS" else (TEXT_WARNING if exp.status in ["OBSERVED", "FLAGGED"] else (TEXT_ERROR if exp.status == "FAIL" else TEXT_MUTED)), 2 if exp.status in ["PASS", "FAIL", "OBSERVED", "FLAGGED"] else 1),
            ]

            for c_idx, (val, col, thick) in enumerate(row_data):
                available_w = cols[c_idx + 1] - cols[c_idx] - 8 if c_idx + 1 < len(cols) else x + w - 34 - cols[c_idx]
                if val == "—":
                    cv2.line(canvas, (cols[c_idx], row_y - 4), (cols[c_idx] + min(16, available_w), row_y - 4), col, 2, AA)
                else:
                    (text_w, _), _ = cv2.getTextSize(val, FONT_BODY, 0.44, thick)
                    fit_scale = 0.44 * min(1.0, max(1, available_w) / max(1, text_w))
                    cv2.putText(canvas, val, (cols[c_idx], row_y), FONT_BODY, fit_scale, col, thick, AA)

            row_y += 27

        footer_msg = (
            f"* Showing {len(recent_experiments)} most recent trials from SQLite (data/experiments.db). "
            f"All {stats.total_experiments} trials preserved. Press [E] to Export CSV/JSON."
        )
        draw_note(canvas, footer_msg, x + 34, t_y + t_h - 12, TEXT_MUTED)

        # ----------------------------------------------------
        # Bottom: Experimental Milestones Completed Checklist
        # ----------------------------------------------------
        b_y = t_y + t_h + 12
        draw_card(canvas, x + 20, b_y, w - 40, 134)
        draw_section(canvas, "Systematic Research & Experimental Milestones Completed", x + 34, b_y + 24, TEXT_ACCENT, thickness=2)

        m1 = [
            ("[PASS]", " Synthetic 12/12 MUSIC Validation"),
            ("[PASS]", " Acoular Geometry Equivalence"),
            ("[PASS]", " Horizontal Azimuth Tracking (L/C/R)"),
            ("[PASS]", " Vertical Monotonic Elevation"),
        ]
        m2 = [
            ("[PASS]", " Physical Array Inversion Fix"),
            ("[PASS]", " Tabletop Reflection Investigation"),
            ("[PASS]", " Sub-Band Spectral Analysis (6 Bands)"),
            ("[PASS]", " 7 Accumulation Strategies Evaluated"),
        ]
        m3 = [
            ("[PASS]", " Sub-100 ms Temporal Synchronization"),
            ("[PASS]", " Real-Time Camera Heatmap Overlay"),
            ("[PASS]", " 20-Frame Synchronized Campaigns"),
            ("[PASS]", " Continuous 8-Second Audio Capture"),
        ]

        milestone_col_w = (w - 68) // 3
        milestone_xs = (x + 34, x + 34 + milestone_col_w, x + 34 + 2 * milestone_col_w)
        for idx, ((tag1, desc1), (tag2, desc2), (tag3, desc3)) in enumerate(zip(m1, m2, m3)):
            ly = b_y + 50 + idx * 20
            for col_x, tag, desc in zip(milestone_xs, (tag1, tag2, tag3), (desc1, desc2, desc3)):
                cv2.putText(canvas, tag, (col_x, ly), FONT_HEADER, 0.42, TEXT_SUCCESS, 1, AA)
                cv2.putText(canvas, desc, (col_x + 51, ly), FONT_BODY, 0.42, TEXT_PRIMARY, 1, AA)
