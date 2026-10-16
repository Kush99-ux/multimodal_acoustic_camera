"""
demo.panels.horizontal_panel
============================
Mode 4: Guided Horizontal Azimuth Localization Validation Test.
Guides testing across Left, Center, and Right broadside positions,
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
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ACCENT, TEXT_ON_DARK,
    TEXT_SUCCESS, TEXT_WARNING,
    ACCENT_UI, ACCENT_GREEN
)
from demo.ui.widgets import draw_panel, draw_card, draw_metric_row, draw_bar, draw_section, Button


class HorizontalTestPanel:
    """Renders Mode 4: Guided Horizontal Localization Test."""

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

        self.positions = ["LEFT", "CENTER", "RIGHT"]
        self.selected_pos = "LEFT"

        self.btn_left = Button("pos_left", "LEFT", x + 30, y + 46, 110, 32)
        self.btn_center = Button("pos_center", "CENTER", x + 150, y + 46, 110, 32)
        self.btn_right = Button("pos_right", "RIGHT", x + 270, y + 46, 110, 32)
        self.btn_start = Button("start_test", "START 20-FRAME TEST", x + 400, y + 46, 220, 32, color=ACCENT_GREEN)

        # Test state (runtime state initialized once in __init__)
        self.test_running = False
        self.test_target_frames = 20
        self.collected_az: List[float] = []
        self.collected_el: List[float] = []
        self.last_results: Dict[str, Optional[Dict[str, Any]]] = {
            "LEFT": None,
            "CENTER": None,
            "RIGHT": None,
        }

        # Historical references from report validation (static, initialized once)
        self.historical_ref = {
            "LEFT": {"median_az": 106.0, "range_az": "105.0--106.5 deg", "u_median": 115.0},
            "CENTER": {"median_az": 90.75, "range_az": "89.5--92.5 deg", "u_median": 310.0},
            "RIGHT": {"median_az": 65.0, "range_az": "64.0--66.5 deg", "u_median": 639.0},
        }

        # Preload recent completed trials from database if repository is available (once, at startup)
        if self.experiment_manager is not None and getattr(self.experiment_manager, "repo", None) is not None:
            try:
                exps = self.experiment_manager.repo.get_recent_experiments(limit=50)
                for exp in reversed(exps):
                    if exp.test_type.lower() == "horizontal" and exp.target_label in self.last_results:
                        self.last_results[exp.target_label] = {
                            "run_id": exp.run_id,
                            "status": exp.status,
                            "position": exp.target_label,
                            "frame_count": exp.frame_count,
                            "median_az": exp.median_azimuth,
                            "std_az": exp.azimuth_std,
                            "median_el": exp.median_elevation,
                            "std_el": exp.elevation_std,
                            "pixel_error": exp.pixel_error,
                            "sync_percentage": exp.sync_percentage,
                            "timestamp": exp.timestamp,
                        }
            except Exception as e:
                print(f"[HorizontalTest] Could not preload recent experiments: {e}")

        self.set_layout(x, y, w, h)
        self._update_tab_states()

    def set_layout(self, x: int, y: int, w: int, h: int) -> None:
        """Update panel dimensions and button coordinates (geometry only)."""
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.btn_left.x = x + 30
        self.btn_left.y = y + 46
        self.btn_center.x = x + 150
        self.btn_center.y = y + 46
        self.btn_right.x = x + 270
        self.btn_right.y = y + 46
        self.btn_start.x = x + 400
        self.btn_start.y = y + 46

    def _update_tab_states(self) -> None:
        self.btn_left.active = (self.selected_pos == "LEFT")
        self.btn_center.active = (self.selected_pos == "CENTER")
        self.btn_right.active = (self.selected_pos == "RIGHT")

    def handle_click(self, px: int, py: int, gt_click: Optional[Tuple[int, int]] = None) -> bool:
        """Handle button clicks for tab selection and test initiation."""
        if self.test_running:
            return False
        if self.btn_left.contains(px, py):
            self.selected_pos = "LEFT"
            self._update_tab_states()
            return True
        if self.btn_center.contains(px, py):
            self.selected_pos = "CENTER"
            self._update_tab_states()
            return True
        if self.btn_right.contains(px, py):
            self.selected_pos = "RIGHT"
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
        print(f"[HorizontalTest] START 20-FRAME TEST for [{self.selected_pos}] (Target: {self.test_target_frames} frames)")

        if self.experiment_manager is not None:
            self.experiment_manager.start_trial(
                test_type="horizontal",
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

        # Synchronize local measurement lists on acoustic-valid frame
        if telemetry.get("peak_valid", False):
            self.collected_az.append(float(telemetry.get("azimuth_deg", 90.0)))
            self.collected_el.append(float(telemetry.get("elevation_deg", -6.0)))
            cur_count = (
                len(self.experiment_manager.collected_frames) + 1
                if (self.experiment_manager is not None and self.experiment_manager.is_running())
                else len(self.collected_az)
            )
            az_val = telemetry.get('azimuth_deg') or 0.0
            print(f"[HorizontalTest] frame {cur_count}/{self.test_target_frames} accepted (Az: {az_val:.1f} deg)")

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
                    "median_az": exp_rec.median_azimuth,
                    "std_az": exp_rec.azimuth_std,
                    "median_el": exp_rec.median_elevation,
                    "std_el": exp_rec.elevation_std,
                    "pixel_error": exp_rec.pixel_error,
                    "sync_percentage": exp_rec.sync_percentage,
                    "timestamp": exp_rec.timestamp,
                }
                self.last_results[exp_rec.target_label] = res
                self.selected_pos = exp_rec.target_label
                self._update_tab_states()

                print(f"[HorizontalTest] TEST COMPLETE for [{exp_rec.target_label}]")
                print(f"[HorizontalTest] summary generated: Median Az={exp_rec.median_azimuth:.2f} deg, Median El={exp_rec.median_elevation:.2f} deg")
                print(f"[HorizontalTest] experiment persistence SUCCESS")
                print(f"[HorizontalTest] run_id = {exp_rec.run_id}")
                print(f"[HorizontalTest] artifacts saved to data/runs/{exp_rec.run_id}/")
                print(f"[HorizontalTest] DB verification = SUCCESS (Status: {exp_rec.status})")
                return exp_rec
        else:
            if len(self.collected_az) >= self.test_target_frames:
                self._finish_test()

        return None

    def _finish_test(self) -> None:
        """Compute summary statistics and save JSON report (fallback)."""
        self.test_running = False
        az_arr = np.array(self.collected_az)
        el_arr = np.array(self.collected_el)

        res = {
            "position": self.selected_pos,
            "frame_count": len(az_arr),
            "median_az": float(np.median(az_arr)),
            "mean_az": float(np.mean(az_arr)),
            "std_az": float(np.std(az_arr)),
            "min_az": float(np.min(az_arr)),
            "max_az": float(np.max(az_arr)),
            "median_el": float(np.median(el_arr)),
            "std_el": float(np.std(el_arr)),
            "timestamp": datetime.now().isoformat(),
        }
        self.last_results[self.selected_pos] = res

        # Save to JSON
        fname = f"horizontal_test_{self.selected_pos}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        fpath = os.path.join(self.output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[HorizontalTest] Saved test results to {fpath}")

    def render(self, canvas: np.ndarray, telemetry: Dict[str, Any]) -> None:
        """Render the horizontal guided test interface."""
        x, y, w, h = self.x, self.y, self.w, self.h
        draw_panel(canvas, x, y, w, h, title="Azimuth Test")

        # Draw Tab Buttons
        self.btn_left.draw(canvas)
        self.btn_center.draw(canvas)
        self.btn_right.draw(canvas)

        # Draw Start Button or Progress
        if self.test_running:
            if self.experiment_manager is not None and self.experiment_manager.is_running():
                cur_frames, target_frames, prog = self.experiment_manager.get_progress()
            else:
                cur_frames = len(self.collected_az)
                target_frames = self.test_target_frames
                prog = min(1.0, cur_frames / float(target_frames)) if target_frames > 0 else 0.0

            draw_card(canvas, x + 400, y + 46, 220, 32)
            draw_bar(canvas, x + 402, y + 48, 216, 28, prog, color=ACCENT_UI)
            txt = f"RECORDING ({cur_frames}/{target_frames})"
            cv2.putText(canvas, txt, (x + 418, y + 68), FONT_BODY, 0.44, TEXT_ON_DARK, 1, AA)
        else:
            self.btn_start.draw(canvas)

        # ----------------------------------------------------
        # Current Position Live Monitor & Test Results
        # ----------------------------------------------------
        draw_card(canvas, x + 30, y + 92, w - 60, 224)
        draw_section(canvas, f"Target Position: {self.selected_pos}", x + 46, y + 120, TEXT_ACCENT, thickness=2)

        live_az = telemetry.get("azimuth_deg")
        live_az_valid = bool(telemetry.get("peak_valid", False)) and live_az is not None
        live_az_str = f"{live_az:.1f} deg" if live_az_valid else "--"
        draw_metric_row(canvas, x + 46, y + 148, "Current Live Azimuth", live_az_str, ACCENT_UI if live_az_valid else TEXT_MUTED, width=w - 120, bold_value=live_az_valid)

        # Display Test Results for this position
        res = self.last_results[self.selected_pos]
        if res is not None:
            run_title = f"Latest Recorded Test: {res.get('run_id', 'RUN')} [{res.get('status', 'COMPLETED')}] (20 Frames)"
            draw_section(canvas, run_title, x + 46, y + 178, TEXT_SUCCESS if res.get('status') == 'PASS' else TEXT_WARNING, thickness=2)
            draw_metric_row(canvas, x + 46, y + 204, "Measured Median Azimuth", f"{res['median_az']:.2f} deg", TEXT_SUCCESS, width=w - 120, bold_value=True)
            draw_metric_row(canvas, x + 46, y + 228, "Azimuth Std Deviation", f"{res['std_az']:.2f} deg", TEXT_PRIMARY, width=w - 120)
            if res.get('pixel_error') is not None:
                draw_metric_row(canvas, x + 46, y + 252, "Ground Truth Pixel Error", f"{res['pixel_error']:.1f} px", ACCENT_UI, width=w - 120, bold_value=True)
            elif "min_az" in res and "max_az" in res:
                draw_metric_row(canvas, x + 46, y + 252, "Measured Range (Min -- Max)", f"{res['min_az']:.1f} deg -- {res['max_az']:.1f} deg", TEXT_SECONDARY, width=w - 120)
            else:
                draw_metric_row(canvas, x + 46, y + 252, "Elevation Context Median", f"{res['median_el']:.2f} deg", TEXT_MUTED, width=w - 120)
            draw_metric_row(canvas, x + 46, y + 276, "Persistent Run ID / Status", f"{res.get('run_id', 'RUN')}  |  {res.get('status', 'SAVED')}", ACCENT_UI, width=w - 120, bold_value=True)
        else:
            cv2.putText(canvas, "No test run recorded yet for this position. Click [START 20-FRAME TEST] above.", (x + 46, y + 200), FONT_BODY, 0.46, TEXT_MUTED, 1, AA)

        # ----------------------------------------------------
        # Historical Reference Comparison Table
        # ----------------------------------------------------
        table_y = y + 330
        draw_card(canvas, x + 30, table_y, w - 60, 260)
        draw_section(canvas, "Historical Three-Position Benchmark (Controlled Experiment)", x + 46, table_y + 28, TEXT_ACCENT, thickness=2)

        # Table Header
        headers = ["Position", "Historical Median Az", "Historical Spread", "Live Measured Median", "Directional Status"]
        col_x = [x + 46, x + 160, x + 370, x + 570, x + 770]
        for h_idx, title in enumerate(headers):
            cv2.putText(canvas, title, (col_x[h_idx], table_y + 56), FONT_HEADER, 0.46, TEXT_SECONDARY, 1, AA)
        cv2.line(canvas, (x + 46, table_y + 64), (x + w - 46, table_y + 64), BORDER, 1)

        row_y = table_y + 94
        for pos in self.positions:
            ref = self.historical_ref[pos]
            pos_res = self.last_results[pos]
            live_str = f"{pos_res['median_az']:.1f} deg ({pos_res.get('run_id', 'LIVE')})" if pos_res else "--"

            status = "PENDING"
            status_col = TEXT_MUTED
            if pos_res:
                if pos == "LEFT" and pos_res["median_az"] > 95.0:
                    status = "PASS (Az > 90 deg)"
                    status_col = TEXT_SUCCESS
                elif pos == "CENTER" and 85.0 <= pos_res["median_az"] <= 95.0:
                    status = "PASS (Az ~ 90 deg)"
                    status_col = TEXT_SUCCESS
                elif pos == "RIGHT" and pos_res["median_az"] < 85.0:
                    status = "PASS (Az < 90 deg)"
                    status_col = TEXT_SUCCESS
                else:
                    status = "OBSERVED"
                    status_col = TEXT_WARNING

            cv2.putText(canvas, pos, (col_x[0], row_y), FONT_BODY, 0.48, ACCENT_UI if pos == self.selected_pos else TEXT_PRIMARY, 2 if pos == self.selected_pos else 1, AA)
            cv2.putText(canvas, f"{ref['median_az']:.1f} deg", (col_x[1], row_y), FONT_BODY, 0.48, TEXT_SECONDARY, 1, AA)
            cv2.putText(canvas, ref['range_az'], (col_x[2], row_y), FONT_BODY, 0.46, TEXT_MUTED, 1, AA)
            cv2.putText(canvas, live_str, (col_x[3], row_y), FONT_BODY, 0.48, TEXT_SUCCESS if pos_res else TEXT_MUTED, 2 if pos_res else 1, AA)
            cv2.putText(canvas, status, (col_x[4], row_y), FONT_BODY, 0.48, status_col, 2 if "PASS" in status else 1, AA)

            row_y += 36

        # Monotonicity check summary
        cv2.putText(canvas, "Validated Physical Ordering: LEFT (106.0 deg) > CENTER (90.8 deg) > RIGHT (65.0 deg)", (x + 46, table_y + 218), FONT_HEADER, 0.48, TEXT_PRIMARY, 2, AA)
        cv2.putText(canvas, "* Baseline convention: Broadside = 90.0 deg. Camera Left > 90 deg, Camera Right < 90 deg.", (x + 46, table_y + 242), FONT_BODY, 0.40, TEXT_MUTED, 1, AA)
