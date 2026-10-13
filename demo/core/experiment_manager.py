"""
demo.core.experiment_manager
============================
Orchestrates validation trials for Mode 4 (Horizontal) and Mode 5 (Vertical),
computes empirical statistics, assigns persistent Run IDs, saves specific trial artifacts,
and records experiments to SQLite.
"""

from __future__ import annotations
import json
import os
import queue
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from demo.data.models import ExperimentRecord, ArtifactRecord
from demo.data.repository import ExperimentRepository
from demo.core.logger import to_json_serializable, NumpyJSONEncoder


class ExperimentManager:
    """Coordinates validation trials, artifact persistence, and SQLite integration."""

    def __init__(
        self,
        repository: ExperimentRepository,
        session_id: str,
        data_dir: Optional[str] = None,
        base_runs_dir: Optional[str] = None,
    ) -> None:
        self.repo = repository
        self.session_id = session_id
        self.project_root = self.repo.project_root

        if base_runs_dir is not None:
            self.runs_dir = os.path.abspath(base_runs_dir)
            self.data_dir = os.path.dirname(self.runs_dir)
        elif data_dir is not None:
            self.data_dir = os.path.abspath(data_dir)
            self.runs_dir = os.path.join(self.data_dir, "runs")
        else:
            self.data_dir = os.path.join(self.project_root, "data")
            self.runs_dir = os.path.join(self.data_dir, "runs")

        os.makedirs(self.runs_dir, exist_ok=True)

        # Thread-safe bounded save queue and background worker
        self._save_queue: queue.Queue = queue.Queue(maxsize=16)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="ExperimentArtifactWorker",
            daemon=True,
        )
        self._worker_thread.start()

        # Register working session in database
        self.repo.register_session(self.session_id, notes="GUI Demo Session")

        # Active trial state
        self.trial_active = False
        self.test_type = ""  # 'horizontal' or 'vertical'
        self.target_label = ""  # e.g. 'LEFT', 'CENTER', 'RIGHT', 'ABOVE', 'LEVEL', 'BELOW'
        self.target_frames = 20
        self.gt_click: Optional[Tuple[int, int]] = None
        self.start_timestamp = ""

        # In-progress frame accumulation
        self.collected_frames: List[Dict[str, Any]] = []
        self.latest_raw_image: Optional[np.ndarray] = None
        self.latest_overlaid_image: Optional[np.ndarray] = None
        self.latest_telemetry: Optional[Dict[str, Any]] = None
        self.latest_heatmap: Optional[np.ndarray] = None

        # Most recently completed experiment
        self.last_saved_experiment: Optional[ExperimentRecord] = None
        self.last_error: Optional[str] = None

    def is_running(self) -> bool:
        """Check whether a validation trial is actively collecting frames."""
        return self.trial_active

    def get_progress(self) -> Tuple[int, int, float]:
        """Return (collected_count, target_count, fraction_0_to_1)."""
        collected = len(self.collected_frames)
        target = max(1, self.target_frames)
        prog = min(1.0, collected / float(target))
        return collected, target, prog

    def start_trial(
        self,
        test_type: str,
        target_label: str,
        target_frames: int = 20,
        gt_click: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Start a new validation trial.
        Note: Clicking ground truth alone does NOT start a trial.
        A trial is only started when explicitly requested by user action.
        """
        self.trial_active = True
        self.test_type = test_type.lower()
        self.target_label = target_label.upper()
        self.target_frames = max(1, target_frames)
        self.gt_click = gt_click
        self.start_timestamp = datetime.now().isoformat()

        self.collected_frames = []
        self.latest_raw_image = None
        self.latest_overlaid_image = None
        self.latest_telemetry = None
        self.latest_heatmap = None
        self.last_error = None

        print(f"[ExperimentManager] Started {self.test_type} trial for [{self.target_label}] (Target: {self.target_frames} frames, GT: {self.gt_click})")

    def cancel_trial(self) -> None:
        """
        Cancel the active trial if interrupted.
        Discards collected frames without creating a database entry.
        """
        if self.trial_active:
            print(f"[ExperimentManager] Interrupted/Cancelled {self.test_type} trial [{self.target_label}]. No database record created.")
        self.trial_active = False
        self.collected_frames = []

    def add_frame(
        self,
        telemetry: Dict[str, Any],
        raw_image: Optional[np.ndarray] = None,
        overlaid_image: Optional[np.ndarray] = None,
        heatmap: Optional[np.ndarray] = None,
    ) -> Tuple[bool, Optional[ExperimentRecord]]:
        """
        Accumulate a valid pipeline frame into the active trial.
        Returns:
            (is_finished, experiment_record_if_just_finished)
        """
        if not self.trial_active:
            return False, None

        # Store latest visual assets as candidates for trial artifacts
        if raw_image is not None and raw_image.size > 0:
            self.latest_raw_image = raw_image.copy()
        if overlaid_image is not None and overlaid_image.size > 0:
            self.latest_overlaid_image = overlaid_image.copy()
        if heatmap is not None and heatmap.size > 0:
            self.latest_heatmap = heatmap.copy()
        self.latest_telemetry = telemetry.copy()

        # Check if frame is acoustic-valid
        if telemetry.get("peak_valid", False):
            az = float(telemetry.get("azimuth_deg", 90.0))
            el = float(telemetry.get("elevation_deg", -6.0))
            sync_delta = float(telemetry.get("sync_delta_ms", 0.0))
            sync_ok = bool(telemetry.get("synchronized", True))
            px_err = telemetry.get("pixel_error_gt")

            frame_entry = {
                "az": az,
                "el": el,
                "sync_delta_ms": sync_delta,
                "synchronized": sync_ok,
                "pixel_error": float(px_err) if px_err is not None else None,
            }
            self.collected_frames.append(frame_entry)

        # Check for trial completion
        if len(self.collected_frames) >= self.target_frames:
            exp_rec = self._finalize_trial()
            return True, exp_rec

        return False, None

    def _determine_status(
        self,
        test_type: str,
        target_label: str,
        median_az: float,
        median_el: float,
    ) -> str:
        """
        Determine empirical outcome status based on established physical criteria.
        Returns: 'PASS', 'OBSERVED', 'FLAGGED', or 'FAILED'.
        """
        # Multipath reflection boundary check (tabletop reflection pulls elevation to bottom boundary)
        if abs(median_el - (-18.75)) < 0.25:
            return "FLAGGED"

        if test_type == "horizontal":
            if target_label == "LEFT":
                return "PASS" if median_az > 95.0 else "OBSERVED"
            elif target_label == "CENTER":
                return "PASS" if 85.0 <= median_az <= 95.0 else "OBSERVED"
            elif target_label == "RIGHT":
                return "PASS" if median_az < 85.0 else "OBSERVED"
        elif test_type == "vertical":
            if target_label == "ABOVE":
                return "PASS" if median_el > 0.0 else "OBSERVED"
            elif target_label == "LEVEL":
                return "PASS" if -12.0 <= median_el <= 0.0 else "OBSERVED"
            elif target_label == "BELOW":
                return "PASS" if median_el < -12.0 else "OBSERVED"

        return "OBSERVED"

    def _finalize_trial(self) -> Optional[ExperimentRecord]:
        """
        Compute statistics, assign persistent Run ID, save artifacts,
        and insert into SQLite database.
        """
        self.trial_active = False

        if not self.collected_frames:
            print("[ExperimentManager] Warning: Finalize called with zero collected frames.")
            return None

        az_arr = np.array([f["az"] for f in self.collected_frames])
        el_arr = np.array([f["el"] for f in self.collected_frames])
        sync_deltas = np.array([f["sync_delta_ms"] for f in self.collected_frames])
        sync_flags = np.array([f["synchronized"] for f in self.collected_frames])

        px_errs = [f["pixel_error"] for f in self.collected_frames if f["pixel_error"] is not None]
        median_px_err = float(np.median(px_errs)) if px_errs else None

        median_az = float(np.median(az_arr))
        std_az = float(np.std(az_arr))
        median_el = float(np.median(el_arr))
        std_el = float(np.std(el_arr))
        sync_pct = float(np.mean(sync_flags) * 100.0) if len(sync_flags) > 0 else 100.0
        med_sync_ms = float(np.median(sync_deltas)) if len(sync_deltas) > 0 else 0.0
        max_sync_ms = float(np.max(sync_deltas)) if len(sync_deltas) > 0 else 0.0

        # Outcome status
        status = self._determine_status(self.test_type, self.target_label, median_az, median_el)

        # Generate unique, persistent Run ID
        run_id = self.repo.get_next_run_id(self.test_type)
        timestamp = datetime.now().isoformat()

        gt_x = self.gt_click[0] if self.gt_click else None
        gt_y = self.gt_click[1] if self.gt_click else None

        # Build experiment record
        record = ExperimentRecord(
            run_id=run_id,
            session_id=self.session_id,
            timestamp=timestamp,
            test_type=self.test_type,
            target_label=self.target_label,
            frame_count=len(self.collected_frames),
            median_azimuth=median_az,
            median_elevation=median_el,
            azimuth_std=std_az,
            elevation_std=std_el,
            ground_truth_x=gt_x,
            ground_truth_y=gt_y,
            pixel_error=median_px_err,
            sync_percentage=sync_pct,
            median_timestamp_difference_ms=med_sync_ms,
            max_timestamp_difference_ms=max_sync_ms,
            status=status,
            notes=f"{self.test_type.capitalize()} trial for {self.target_label}",
            config_metadata=json.dumps({
                "target_frames": self.target_frames,
                "start_time": self.start_timestamp,
            }),
        )

        # ----------------------------------------------------
        # Non-blocking Persistence Task Enqueue
        # ----------------------------------------------------
        task = {
            "record": record,
            "raw_frames": list(self.collected_frames),
            "raw_image": self.latest_raw_image.copy() if self.latest_raw_image is not None else None,
            "overlaid_image": self.latest_overlaid_image.copy() if self.latest_overlaid_image is not None else None,
            "telemetry": self.latest_telemetry.copy() if self.latest_telemetry is not None else None,
            "heatmap": self.latest_heatmap.copy() if self.latest_heatmap is not None else None,
        }

        try:
            self._save_queue.put_nowait(task)
        except queue.Full:
            err_msg = f"Save queue is full (maxsize=16); dropping trial {run_id}"
            print(f"[ExperimentManager] ERROR: {err_msg}")
            self.last_error = err_msg

        self.last_saved_experiment = record
        return record

    def _worker_loop(self) -> None:
        """Worker thread processing queued experiment persistence tasks off the main thread."""
        while not self._stop_event.is_set():
            try:
                task = self._save_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._execute_save_task(task)
            except Exception as e:
                self.last_error = str(e)
                print(f"[ExperimentManager] Worker unhandled error saving trial: {e}")
            finally:
                self._save_queue.task_done()

        # Flush any remaining items in the queue after stop event is signalled
        while True:
            try:
                task = self._save_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._execute_save_task(task)
            except Exception as e:
                self.last_error = str(e)
                print(f"[ExperimentManager] Worker flush error: {e}")
            finally:
                self._save_queue.task_done()

    def _execute_save_task(self, task: Dict[str, Any]) -> None:
        """Execute artifact disk serialization and SQLite persistence."""
        record: ExperimentRecord = task["record"]
        raw_frames = task["raw_frames"]
        raw_image: Optional[np.ndarray] = task["raw_image"]
        overlaid_image: Optional[np.ndarray] = task["overlaid_image"]
        telemetry: Optional[Dict[str, Any]] = task["telemetry"]
        heatmap: Optional[np.ndarray] = task["heatmap"]

        run_id = record.run_id
        timestamp = record.timestamp
        status = record.status

        run_dir = os.path.join(self.runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        artifacts: List[ArtifactRecord] = []

        # 1. Summary JSON
        summary_path = os.path.join(run_dir, "summary.json")
        rel_summary_path = os.path.relpath(summary_path, self.project_root).replace("\\", "/")
        try:
            summary_payload = record.to_dict()
            summary_payload["frames_collected"] = record.frame_count
            summary_payload["raw_frames"] = raw_frames
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2, cls=NumpyJSONEncoder)
            artifacts.append(ArtifactRecord(
                run_id=run_id,
                artifact_type="summary",
                file_path=rel_summary_path,
                created_at=timestamp,
            ))
        except Exception as e:
            self.last_error = f"Failed to save summary.json: {e}"
            print(f"[ExperimentManager] Warning: Failed to save summary.json: {e}")

        # 2. Representative camera image
        if raw_image is not None and raw_image.size > 0:
            rep_path = os.path.join(run_dir, "representative.jpg")
            rel_rep_path = os.path.relpath(rep_path, self.project_root).replace("\\", "/")
            try:
                cv2.imwrite(rep_path, raw_image)
                artifacts.append(ArtifactRecord(
                    run_id=run_id,
                    artifact_type="representative_image",
                    file_path=rel_rep_path,
                    created_at=timestamp,
                ))
            except Exception as e:
                self.last_error = f"Failed to save representative.jpg: {e}"
                print(f"[ExperimentManager] Warning: Failed to save representative.jpg: {e}")

        # 3. Overlaid multimodal image
        if overlaid_image is not None and overlaid_image.size > 0:
            overlay_path = os.path.join(run_dir, "overlay.jpg")
            rel_overlay_path = os.path.relpath(overlay_path, self.project_root).replace("\\", "/")
            try:
                cv2.imwrite(overlay_path, overlaid_image)
                artifacts.append(ArtifactRecord(
                    run_id=run_id,
                    artifact_type="overlay_image",
                    file_path=rel_overlay_path,
                    created_at=timestamp,
                ))
            except Exception as e:
                self.last_error = f"Failed to save overlay.jpg: {e}"
                print(f"[ExperimentManager] Warning: Failed to save overlay.jpg: {e}")

        # 4. Telemetry JSON
        if telemetry is not None:
            telemetry_path = os.path.join(run_dir, "telemetry.json")
            rel_telemetry_path = os.path.relpath(telemetry_path, self.project_root).replace("\\", "/")
            try:
                safe_tel = to_json_serializable(telemetry)
                with open(telemetry_path, "w", encoding="utf-8") as f:
                    json.dump(safe_tel, f, indent=2, cls=NumpyJSONEncoder)
                artifacts.append(ArtifactRecord(
                    run_id=run_id,
                    artifact_type="telemetry",
                    file_path=rel_telemetry_path,
                    created_at=timestamp,
                ))
            except Exception as e:
                self.last_error = f"Failed to save telemetry.json: {e}"
                print(f"[ExperimentManager] Warning: Failed to save telemetry.json: {e}")

        # 5. Acoustic Heatmap NPY
        if heatmap is not None and heatmap.size > 0:
            heatmap_path = os.path.join(run_dir, "heatmap.npy")
            rel_heatmap_path = os.path.relpath(heatmap_path, self.project_root).replace("\\", "/")
            try:
                np.save(heatmap_path, heatmap)
                artifacts.append(ArtifactRecord(
                    run_id=run_id,
                    artifact_type="heatmap",
                    file_path=rel_heatmap_path,
                    created_at=timestamp,
                ))
            except Exception as e:
                self.last_error = f"Failed to save heatmap.npy: {e}"
                print(f"[ExperimentManager] Warning: Failed to save heatmap.npy: {e}")

        # ----------------------------------------------------
        # Database Insertion
        # ----------------------------------------------------
        try:
            db_id = self.repo.insert_experiment(record, artifacts)
            record.id = db_id
            print(f"[ExperimentManager] SUCCESS: Trial saved to database as {run_id} (Status: {status}, ID: {db_id}, Artifacts: {len(artifacts)})")
        except Exception as e:
            self.last_error = str(e)
            print(f"[ExperimentManager] ERROR inserting experiment into SQLite: {e}")

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Gracefully shut down background artifact saver, waiting for queued trials to flush.
        """
        if not self._worker_thread.is_alive():
            return

        print(f"[ExperimentManager] Waiting up to {timeout:.1f}s for pending artifact saves to flush...")
        t0 = time.time()
        # Wait for queue items to be marked task_done
        while not self._save_queue.empty() and (time.time() - t0 < timeout):
            time.sleep(0.05)

        self._stop_event.set()
        remaining_timeout = max(0.1, timeout - (time.time() - t0))
        self._worker_thread.join(timeout=remaining_timeout)
        if self._worker_thread.is_alive():
            print("[ExperimentManager] Warning: Worker thread did not terminate within timeout.")
        else:
            print("[ExperimentManager] Background saver shutdown cleanly.")

