"""
demo.panels.vertical_panel
==========================
Mode 5: Guided Vertical Elevation Localization Validation Test.
Guides testing across Above, Level, and Below broadside positions,
accumulates 20 valid acoustic frames, and reports empirical statistics.
Upgraded with high-contrast, dual-stroke, non-aliased typography.
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from demo.ui.theme import (
    FONT_BODY, FONT_HEADER, AA, BG_CARD, BG_PANEL, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ACCENT,
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_ORANGE
)
from demo.ui.widgets import draw_panel, draw_card, draw_metric_row, draw_bar, draw_section, Button


class VerticalTestPanel:
    """Renders Mode 5: Guided Vertical Elevation Localization Test."""

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        output_dir: str = "scratch/demo_results",
        experiment_manager: Optional[Any] = None,
    ) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.output_dir = os.path.abspath(output_dir)
        self.experiment_manager = experiment_manager
        os.makedirs(self.output_dir, exist_ok=True)

        self.positions = ["ABOVE", "LEVEL", "BELOW"]
        self.selected_pos = "LEVEL"

        # Interactive buttons with clean 32px height
        self.btn_above = Button("pos_above", "[1] ABOVE", x + 30, y + 46, 110, 32)
        self.btn_level = Button("pos_level", "[2] LEVEL", x + 150, y + 46, 110, 32)
        self.btn_below = Button("pos_below", "[3] BELOW", x + 270, y + 46, 110, 32)
        self.btn_start = Button("start_test", "START 20-FRAME TEST", x + 400, y + 46, 220, 32, color=ACCENT_GREEN)

        self._update_tab_states()

        # Test state
        self.test_running = False
        self.test_target_frames = 20
        self.collected_az: List[float] = []
        self.collected_el: List[float] = []
        self.last_results: Dict[str, Optional[Dict[str, Any]]] = {
            "ABOVE": None,
            "LEVEL": None,
            "BELOW": None,
        }

        # Historical references from report validation
        self.historical_ref = {
            "ABOVE": {"median_el": 12.25, "median_az": 90.5, "desc": "Phone raised +25cm"},
            "LEVEL": {"median_el": -6.00, "median_az": 90.75, "desc": "Visually level with lens (array pitched -6 deg)"},
            "BELOW": {"median_el": -17.00, "median_az": 90.0, "desc": "Phone lowered -25cm"},
        }

    def _update_tab_states(self) -> None:
        self.btn_above.active = (self.selected_pos == "ABOVE")
        self.btn_level.active = (self.selected_pos == "LEVEL")
        self.btn_below.active = (self.selected_pos == "BELOW")

    def handle_click(self, px: int, py: int, gt_click: Optional[Tuple[int, int]] = None) -> bool:
        """Handle button clicks for tab selection and test initiation."""
        if self.btn_above.contains(px, py):
            self.selected_pos = "ABOVE"
            self._update_tab_states()
            return True
        if self.btn_level.contains(px, py):
            self.selected_pos = "LEVEL"
            self._update_tab_states()
            return True
        if self.btn_below.contains(px, py):
            self.selected_pos = "BELOW"
            self._update_tab_states()
            return True
        if self.btn_start.contains(px, py) and not self.test_running:
            self.start_test(gt_click=gt_click)
            return True
        return False

    def start_test(self, gt_click: Optional[Tuple[int, int]] = None) -> None:
        """Begin automated 20-frame collection for current position."""
        self.test_running = True
        self.collected_az = []
        self.collected_el = []

        if self.experiment_manager is not None:
            self.experiment_manager.start_trial(
                test_type="vertical",
                target_label=self.selected_pos,
                target_frames=self.test_target_frames,
                gt_click=gt_click,
            )

    def update_frame(
        self,
        telemetry: Dict[str, Any],
        raw_image: Optional[np.ndarray] = None,
        overlaid_image: Optional[np.ndarray] = None,
        heatmap: Optional[np.ndarray] = None,
    ) -> Optional[Any]:
        """Accumulate live frames if test is in progress."""
        if not self.test_running:
            return None

        if self.experiment_manager is not None:
            finished, exp_rec = self.experiment_manager.add_frame(
                telemetry,
                raw_image=raw_image,
                overlaid_image=overlaid_image,
                heatmap=heatmap,
            )
            if finished and exp_rec is not None:
                self.test_running = False
                res = {
                    "run_id": exp_rec.run_id,
                    "status": exp_rec.status,
                    "position": exp_rec.target_label,
                    "frame_count": exp_rec.frame_count,
                    "median_el": exp_rec.median_elevation,
                    "std_el": exp_rec.elevation_std,
                    "median_az": exp_rec.median_azimuth,
                    "std_az": exp_rec.azimuth_std,
                    "pixel_error": exp_rec.pixel_error,
                    "sync_percentage": exp_rec.sync_percentage,
                    "timestamp": exp_rec.timestamp,
                }
                self.last_results[self.selected_pos] = res
                return exp_rec
        else:
            if telemetry.get("peak_valid", False):
                self.collected_az.append(float(telemetry.get("azimuth_deg", 90.0)))
                self.collected_el.append(float(telemetry.get("elevation_deg", -6.0)))

            if len(self.collected_el) >= self.test_target_frames:
                self._finish_test()

        return None

    def _finish_test(self) -> None:
        """Compute summary statistics and save JSON report (fallback)."""
        self.test_running = False
        az_arr = np.array(self.collected_az)
        el_arr = np.array(self.collected_el)

        res = {
            "position": self.selected_pos,
            "frame_count": len(el_arr),
            "median_el": float(np.median(el_arr)),
            "mean_el": float(np.mean(el_arr)),
            "std_el": float(np.std(el_arr)),
            "min_el": float(np.min(el_arr)),
            "max_el": float(np.max(el_arr)),
            "median_az": float(np.median(az_arr)),
            "std_az": float(np.std(az_arr)),
            "timestamp": datetime.now().isoformat(),
        }
        self.last_results[self.selected_pos] = res

        # Save to JSON
        fname = f"vertical_test_{self.selected_pos}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        fpath = os.path.join(self.output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[VerticalTest] Saved test results to {fpath}")

    def render(self, canvas: np.ndarray, telemetry: Dict[str, Any]) -> None:
        """Render the vertical guided test interface."""
        x, y, w, h = self.x, self.y, self.w, self.h
        draw_panel(canvas, x, y, w, h, title="MODE 5: VERTICAL ELEVATION LOCALIZATION TEST")

        # Draw Tab Buttons
        self.btn_above.draw(canvas)
        self.btn_level.draw(canvas)
        self.btn_below.draw(canvas)

        # Draw Start Button or Progress
        if self.test_running:
            prog = len(self.collected_el) / float(self.test_target_frames)
            draw_card(canvas, x + 400, y + 46, 220, 32)
            draw_bar(canvas, x + 402, y + 48, 216, 28, prog, color=ACCENT_ORANGE)
            txt = f"RECORDING ({len(self.collected_el)}/{self.test_target_frames})"
            cv2.putText(canvas, txt, (x + 418, y + 68), FONT_BODY, 0.44, (255, 255, 255), 1, AA)
        else:
            self.btn_start.draw(canvas)

        # ----------------------------------------------------
        # Current Position Live Monitor & Test Results
        # ----------------------------------------------------
        draw_card(canvas, x + 30, y + 92, w - 60, 224)
        draw_section(canvas, f"Target Position: {self.selected_pos}", x + 46, y + 120, TEXT_ACCENT, thickness=2)

        live_el = telemetry.get("elevation_deg")
        live_el = -6.0 if live_el is None else live_el
        draw_metric_row(canvas, x + 46, y + 148, "Current Live Elevation", f"{live_el:+.1f} deg", ACCENT_CYAN, width=w - 120, bold_value=True)

        # Display Test Results for this position
        res = self.last_results[self.selected_pos]
        if res is not None:
            run_title = f"Latest Recorded Test: {res.get('run_id', 'RUN')} [{res.get('status', 'COMPLETED')}] (20 Frames)"
            draw_section(canvas, run_title, x + 46, y + 178, ACCENT_GREEN if res.get('status') == 'PASS' else ACCENT_AMBER, thickness=2)
            draw_metric_row(canvas, x + 46, y + 204, "Measured Median Elevation", f"{res['median_el']:+.2f} deg", ACCENT_GREEN, width=w - 120, bold_value=True)
            draw_metric_row(canvas, x + 46, y + 228, "Elevation Std Deviation", f"{res['std_el']:.2f} deg", TEXT_PRIMARY, width=w - 120)
            if res.get('pixel_error') is not None:
                draw_metric_row(canvas, x + 46, y + 252, "Ground Truth Pixel Error", f"{res['pixel_error']:.1f} px", ACCENT_CYAN, width=w - 120, bold_value=True)
            elif "min_el" in res and "max_el" in res:
                draw_metric_row(canvas, x + 46, y + 252, "Measured Range (Min -- Max)", f"{res['min_el']:+.1f} deg -- {res['max_el']:+.1f} deg", TEXT_SECONDARY, width=w - 120)
            else:
                draw_metric_row(canvas, x + 46, y + 252, "Azimuth Context Median", f"{res['median_az']:.2f} deg", TEXT_MUTED, width=w - 120)
            draw_metric_row(canvas, x + 46, y + 276, "Persistent Run ID / Status", f"{res.get('run_id', 'RUN')}  |  {res.get('status', 'SAVED')}", ACCENT_CYAN, width=w - 120, bold_value=True)
        else:
            cv2.putText(canvas, "No test run recorded yet for this position. Click [START 20-FRAME TEST] above.", (x + 46, y + 200), FONT_BODY, 0.46, TEXT_MUTED, 1, AA)

        # ----------------------------------------------------
        # Historical Reference Comparison Table
        # ----------------------------------------------------
        table_y = y + 330
        draw_card(canvas, x + 30, table_y, w - 60, 260)
        draw_section(canvas, "Historical Three-Position Elevation Benchmark", x + 46, table_y + 28, TEXT_ACCENT, thickness=2)

        # Table Header
        headers = ["Position", "Historical Median El", "Context Azimuth", "Live Measured Median", "Directional Status"]
        col_x = [x + 46, x + 170, x + 370, x + 560, x + 750]
        for h_idx, title in enumerate(headers):
            cv2.putText(canvas, title, (col_x[h_idx], table_y + 56), FONT_HEADER, 0.46, TEXT_SECONDARY, 1, AA)
        cv2.line(canvas, (x + 46, table_y + 64), (x + w - 46, table_y + 64), BORDER, 1)

        row_y = table_y + 94
        for pos in self.positions:
            ref = self.historical_ref[pos]
            pos_res = self.last_results[pos]
            live_str = f"{pos_res['median_el']:+.1f} deg ({pos_res.get('run_id', 'LIVE')})" if pos_res else "--"

            status = "PENDING"
            status_col = TEXT_MUTED
            if pos_res:
                if pos == "ABOVE" and pos_res["median_el"] > 0.0:
                    status = "PASS (El > 0 deg)"
                    status_col = ACCENT_GREEN
                elif pos == "LEVEL" and -10.0 <= pos_res["median_el"] <= 0.0:
                    status = "PASS (El ~ -6 deg)"
                    status_col = ACCENT_GREEN
                elif pos == "BELOW" and pos_res["median_el"] < -10.0:
                    status = "PASS (El < -10 deg)"
                    status_col = ACCENT_GREEN
                else:
                    status = "OBSERVED"
                    status_col = ACCENT_AMBER

            cv2.putText(canvas, pos, (col_x[0], row_y), FONT_BODY, 0.48, ACCENT_CYAN if pos == self.selected_pos else TEXT_PRIMARY, 2 if pos == self.selected_pos else 1, AA)
            cv2.putText(canvas, f"{ref['median_el']:+.2f} deg", (col_x[1], row_y), FONT_BODY, 0.48, TEXT_SECONDARY, 1, AA)
            cv2.putText(canvas, f"{ref['median_az']:.1f} deg", (col_x[2], row_y), FONT_BODY, 0.46, TEXT_MUTED, 1, AA)
            cv2.putText(canvas, live_str, (col_x[3], row_y), FONT_BODY, 0.48, ACCENT_GREEN if pos_res else TEXT_MUTED, 2 if pos_res else 1, AA)
            cv2.putText(canvas, status, (col_x[4], row_y), FONT_BODY, 0.48, status_col, 2 if "PASS" in status else 1, AA)

            row_y += 36

        # Monotonicity check summary
        cv2.putText(canvas, "Validated Physical Ordering: ABOVE (+12.25 deg) > LEVEL (-6.00 deg) > BELOW (-17.00 deg)", (x + 46, table_y + 218), FONT_HEADER, 0.48, TEXT_PRIMARY, 2, AA)
        cv2.putText(canvas, "* Empirical elevation baseline: LEVEL visually centered at v ~ 231 px maps to acoustic El ~ -6.0 deg.", (x + 46, table_y + 242), FONT_BODY, 0.40, TEXT_MUTED, 1, AA)
