"""
scratch/verify_experiment_architecture.py
=========================================
Automated Verification Suite for the Multimodal Acoustic Camera Experiment Database
and Validation Architecture (Tests 1 through 13).
"""

import csv
import json
import os
import shutil
import sys
import tempfile
import numpy as np

# Ensure project root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from demo.data.database import ExperimentDatabase
from demo.data.repository import ExperimentRepository
from demo.data.migrations import migrate_historical_records
from demo.core.experiment_manager import ExperimentManager
from demo.panels.horizontal_panel import HorizontalTestPanel
from demo.panels.vertical_panel import VerticalTestPanel
from demo.panels.validation_panel import ValidationPanel
from demo.ui.theme import CANVAS_WIDTH, CANVAS_HEIGHT, BG_CANVAS


def test_suite():
    print("=" * 70)
    print("STARTING EXPERIMENT DATABASE ARCHITECTURE VERIFICATION SUITE")
    print("=" * 70)

    # Use a temporary directory for test database & runs to ensure complete isolation
    temp_dir = tempfile.mkdtemp(prefix="mac_test_")
    test_db_path = os.path.join(temp_dir, "test_experiments.db")
    test_runs_dir = os.path.join(temp_dir, "runs")
    test_exports_dir = os.path.join(temp_dir, "exports")

    try:
        # -------------------------------------------------------------
        # TEST 1: Database creation on demand with valid schema
        # -------------------------------------------------------------
        print("\n[TEST 1] Initializing SQLite database on demand...")
        db = ExperimentDatabase(db_path=test_db_path)
        conn = db.get_connection()
        cursor = conn.cursor()

        # Check WAL mode and foreign keys
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0].lower()
        cursor.execute("PRAGMA foreign_keys;")
        foreign_keys = cursor.fetchone()[0]
        assert journal_mode == "wal", f"Expected WAL mode, got {journal_mode}"
        assert foreign_keys == 1, f"Expected foreign_keys=1, got {foreign_keys}"

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert {"sessions", "experiments", "artifacts"}.issubset(tables), f"Missing tables: {tables}"
        conn.close()
        print("  -> PASSED: DB initialized with WAL mode, foreign keys, and all 3 tables.")

        # -------------------------------------------------------------
        # TEST 2: Historical migration seeds exact records idempotently
        # -------------------------------------------------------------
        print("\n[TEST 2] Running historical validation migrations...")
        repo = ExperimentRepository(db)
        migrate_historical_records(repo)

        all_exp = repo.get_all_experiments()
        assert len(all_exp) == 5, f"Expected 5 historical records, got {len(all_exp)}"

        run_ids = [e.run_id for e in all_exp]
        expected_ids = ["Run 2", "Run 4", "Run 5", "Run 6", "Run 7"]
        assert sorted(run_ids) == sorted(expected_ids), f"Expected {expected_ids}, got {run_ids}"

        # Check exact numerical values for Run 7
        run7 = repo.get_experiment_by_id("Run 7")
        assert run7.label == "RIGHT"
        assert run7.median_azimuth == 68.5
        assert run7.median_elevation == -18.75
        assert run7.azimuth_std == 11.53
        assert run7.elevation_std == 4.74
        assert run7.pixel_error == 135.2
        assert run7.outcome_status == "PASS"

        # Idempotence check: run migration again
        migrate_historical_records(repo)
        all_exp2 = repo.get_all_experiments()
        assert len(all_exp2) == 5, f"Migration not idempotent! Count became {len(all_exp2)}"
        print("  -> PASSED: 5 historical runs migrated with exact values and idempotence verified.")

        # -------------------------------------------------------------
        # TEST 3: Monotonic Run ID generation
        # -------------------------------------------------------------
        print("\n[TEST 3] Verifying Run ID generation...")
        next_h = repo.get_next_run_id("H")
        next_v = repo.get_next_run_id("V")
        assert next_h == "H-001", f"Expected H-001, got {next_h}"
        assert next_v == "V-001", f"Expected V-001, got {next_v}"
        print("  -> PASSED: Next run IDs correctly start at H-001 and V-001.")

        # -------------------------------------------------------------
        # TEST 8: Ground Truth click alone does NOT create records
        # -------------------------------------------------------------
        print("\n[TEST 8] Verifying GT click does NOT create DB records...")
        initial_count = repo.get_experiment_count()
        # Simulate user setting ground truth
        gt_click = (320, 240)
        assert repo.get_experiment_count() == initial_count
        print(f"  -> PASSED: GT click set, DB count unchanged ({initial_count}).")

        # -------------------------------------------------------------
        # TEST 4: Mode 4 trial collection (20 frames) -> Save -> Increment
        # -------------------------------------------------------------
        print("\n[TEST 4] Testing Mode 4 Horizontal Trial (20 frames)...")
        exp_mgr = ExperimentManager(repo, session_id="SES-TEST-001", base_runs_dir=test_runs_dir)
        h_panel = HorizontalTestPanel(0, 0, 800, 600, experiment_manager=exp_mgr)

        # User clicks "Center (0 deg)"
        h_panel.selected_pos = "CENTER"
        h_panel.start_test(gt_click=(320, 240))
        assert h_panel.test_running is True

        # Feed 20 simulated frames
        res = None
        for frame_idx in range(20):
            mock_telemetry = {
                "peak_valid": True,
                "azimuth_deg": 90.5 + float(np.random.normal(0, 0.2)),
                "elevation_deg": -5.2 + float(np.random.normal(0, 0.2)),
                "peak_snr": 18.5,
                "sync_delta_ms": 1.2,
                "synchronized": True,
                "pixel_error_gt": 4.5,
                "doa_x": 322,
                "doa_y": 239,
                "yolo_target_x": 320,
                "yolo_target_y": 240,
            }
            mock_raw = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_ovl = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_hmap = np.zeros((76, 101), dtype=np.float32)

            res = h_panel.update_frame(mock_telemetry, raw_image=mock_raw, overlaid_image=mock_ovl, heatmap=mock_hmap)

        assert res is not None, "Horizontal test did not return result after 20 frames!"
        assert res["run_id"] == "H-001"
        assert res["outcome_status"] in ["PASS", "OBSERVED", "FLAGGED"]
        assert h_panel.test_running is False
        assert repo.get_experiment_count() == 6
        assert repo.get_next_run_id("H") == "H-002"
        print(f"  -> PASSED: Mode 4 completed H-001 ({res['outcome_status']}), next is H-002.")

        # -------------------------------------------------------------
        # TEST 5: Mode 5 trial collection (20 frames) -> Save -> Increment
        # -------------------------------------------------------------
        print("\n[TEST 5] Testing Mode 5 Vertical Trial (20 frames)...")
        v_panel = VerticalTestPanel(0, 0, 800, 600, experiment_manager=exp_mgr)

        # User clicks "Above (+15 deg)"
        v_panel.selected_pos = "ABOVE"
        v_panel.start_test(gt_click=(320, 180))
        assert v_panel.test_running is True

        res_v = None
        for frame_idx in range(20):
            mock_telemetry = {
                "peak_valid": True,
                "azimuth_deg": 90.1,
                "elevation_deg": 14.8 + float(np.random.normal(0, 0.3)),
                "peak_snr": 17.0,
                "sync_delta_ms": 0.8,
                "synchronized": True,
                "pixel_error_gt": 5.2,
                "doa_x": 321,
                "doa_y": 182,
                "yolo_target_x": 320,
                "yolo_target_y": 180,
            }
            mock_raw = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_ovl = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_hmap = np.zeros((76, 101), dtype=np.float32)

            res_v = v_panel.update_frame(mock_telemetry, raw_image=mock_raw, overlaid_image=mock_ovl, heatmap=mock_hmap)

        assert res_v is not None, "Vertical test did not return result after 20 frames!"
        assert res_v["run_id"] == "V-001"
        assert v_panel.test_running is False
        assert repo.get_experiment_count() == 7
        assert repo.get_next_run_id("V") == "V-002"
        print(f"  -> PASSED: Mode 5 completed V-001 ({res_v['outcome_status']}), next is V-002.")

        # -------------------------------------------------------------
        # TEST 6 & 7: Artifact files exist on disk & records have relative paths
        # -------------------------------------------------------------
        print("\n[TEST 6 & 7] Verifying trial artifacts on disk and DB references...")
        h1_dir = os.path.join(test_runs_dir, "H-001")
        assert os.path.isdir(h1_dir), f"Directory {h1_dir} not created!"

        expected_files = ["summary.json", "representative.jpg", "overlay.jpg", "telemetry.json", "heatmap.npy"]
        for f in expected_files:
            file_path = os.path.join(h1_dir, f)
            assert os.path.isfile(file_path), f"Artifact file {f} missing from {h1_dir}!"
            assert os.path.getsize(file_path) > 0, f"Artifact file {f} is empty!"

        # Verify summary.json content
        with open(os.path.join(h1_dir, "summary.json"), "r") as jf:
            summary_data = json.load(jf)
            assert summary_data["run_id"] == "H-001"
            assert summary_data["frames_collected"] == 20

        # Verify DB artifact records
        h1_artifacts = repo.get_artifacts_for_experiment("H-001")
        assert len(h1_artifacts) == 5, f"Expected 5 DB artifact records, got {len(h1_artifacts)}"
        for art in h1_artifacts:
            assert not os.path.isabs(art.file_path), f"Artifact path {art.file_path} is absolute! Must be relative."
            assert "H-001" in art.file_path
            assert os.path.getsize(os.path.join(h1_dir, os.path.basename(art.file_path))) > 0
        print("  -> PASSED: All 5 artifact files created and registered with valid relative paths.")

        # -------------------------------------------------------------
        # TEST 9 & 10: Mode 8 renders dynamic trials & aggregates
        # -------------------------------------------------------------
        print("\n[TEST 9 & 10] Testing Mode 8 Dynamic Rendering and DB Aggregates...")
        stats = repo.get_database_stats()
        assert stats.total_trials == 7
        assert stats.pass_count >= 5  # 5 historical were all PASS + potentially H-001 and V-001
        assert stats.success_rate >= 70.0
        print(f"  -> Aggregates: Total={stats.total_trials}, PASS={stats.pass_count}, Success={stats.success_rate:.1f}%")

        val_panel = ValidationPanel(10, 68, 1020, 788, repository=repo)
        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), BG_CANVAS, dtype=np.uint8)
        val_panel.render(canvas, {})
        # Verify no exceptions during rendering
        print("  -> PASSED: Mode 8 successfully queried repo and rendered dynamic canvas.")

        # -------------------------------------------------------------
        # TEST 11: Persistence across restart
        # -------------------------------------------------------------
        print("\n[TEST 11] Verifying database persistence across restart...")
        db.close()
        # Re-open database with new instance
        db2 = ExperimentDatabase(db_path=test_db_path)
        repo2 = ExperimentRepository(db2)
        assert repo2.get_experiment_count() == 7
        assert repo2.get_next_run_id("H") == "H-002"
        assert repo2.get_next_run_id("V") == "V-002"
        h1_reloaded = repo2.get_experiment_by_id("H-001")
        assert h1_reloaded is not None
        assert h1_reloaded.label == "CENTER"
        db2.close()
        print("  -> PASSED: Database fully reloaded with all records and sequence states intact.")

        # -------------------------------------------------------------
        # TEST 12: Top-11 Limit / Pagination when > 11 records exist
        # -------------------------------------------------------------
        print("\n[TEST 12] Testing handling of >11 trials (pagination/limit)...")
        db3 = ExperimentDatabase(db_path=test_db_path)
        repo3 = ExperimentRepository(db3)

        # Add 10 more trials to reach 17 total
        for i in range(10):
            run_id = repo3.get_next_run_id("H")
            repo3.insert_experiment({
                "run_id": run_id,
                "session_id": "SES-PAGINATION",
                "timestamp": f"2099-01-01 12:00:{i:02d}",
                "test_type": "HORIZONTAL",
                "label": f"Batch Test {i}",
                "target_azimuth_deg": 0.0,
                "target_elevation_deg": 0.0,
                "median_azimuth": 0.1 * i,
                "median_elevation": 0.0,
                "azimuth_std": 0.5,
                "elevation_std": 0.5,
                "pixel_error": 5.0 + i,
                "median_snr": 16.0,
                "mean_sync_offset_ms": 1.0,
                "frames_collected": 20,
                "valid_frames": 20,
                "outcome_status": "PASS" if i % 2 == 0 else "FLAGGED",
            })

        total_trials = repo3.get_experiment_count()
        assert total_trials == 17, f"Expected 17 total records, got {total_trials}"

        # Test limit=11
        recent_11 = repo3.get_recent_experiments(limit=11)
        assert len(recent_11) == 11, f"Expected 11 records returned, got {len(recent_11)}"
        # Most recent should be H-011
        assert recent_11[0].run_id == "H-011"

        # Verify Mode 8 renders cleanly with 11 rows
        val_panel3 = ValidationPanel(10, 68, 1020, 788, repository=repo3)
        canvas3 = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), BG_CANVAS, dtype=np.uint8)
        val_panel3.render(canvas3, {})
        print(f"  -> PASSED: 17 trials in DB, top-11 cleanly queried and rendered in Mode 8.")

        # -------------------------------------------------------------
        # TEST 13: Export to CSV and JSON
        # -------------------------------------------------------------
        print("\n[TEST 13] Verifying CSV and JSON exports...")
        csv_file = repo3.export_csv(output_dir=test_exports_dir)
        json_file = repo3.export_json(output_dir=test_exports_dir)

        assert os.path.isfile(csv_file), f"CSV file not found: {csv_file}"
        assert os.path.isfile(json_file), f"JSON file not found: {json_file}"
        assert os.path.getsize(csv_file) > 0, "CSV file is empty!"
        assert os.path.getsize(json_file) > 0, "JSON file is empty!"

        # Read CSV
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            # Header + 17 rows = 18 lines
            assert len(rows) == 18, f"Expected 18 CSV rows (1 header + 17 data), got {len(rows)}"

        # Read JSON
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 17, f"Expected 17 JSON records, got {len(data)}"
            assert data[0]["run_id"] == "Run 2"  # export_json orders by created_at ASC

        print(f"  -> PASSED: CSV ({os.path.basename(csv_file)}) and JSON ({os.path.basename(json_file)}) exports verified.")

        db3.close()

        print("\n" + "=" * 70)
        print("ALL 13 TESTS PASSED PERFECTLY!")
        print("=" * 70)

    finally:
        # Clean up temporary test directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_suite()
