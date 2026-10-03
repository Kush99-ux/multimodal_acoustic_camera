"""
scratch/test_mode4_trace.py
Trace the exact execution of Mode 4 in ProfessorDemoApp.
"""

import os
import sys

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from demo.app import ProfessorDemoApp
import sqlite3

def run_trace():
    print("=== INITIALIZING ProfessorDemoApp (simulate=True) ===")
    app = ProfessorDemoApp(simulate=True, headless_frames=1)
    
    print("\n=== SWITCHING TO MODE 4 ===")
    app.current_mode = 4
    
    print("\n=== CLICKING 'CENTER' TAB THEN START TEST ===")
    # Click CENTER tab
    app.panel_horizontal.handle_click(180, 120)
    print(f"Selected pos: {app.panel_horizontal.selected_pos}")
    
    # Click START 20-FRAME TEST button
    app.panel_horizontal.handle_click(450, 120)
    print(f"Test running: {app.panel_horizontal.test_running}")
    print(f"ExpManager trial_active: {app.experiment_manager.trial_active}")
    print(f"ExpManager test_type: {app.experiment_manager.test_type}")
    print(f"ExpManager target_label: {app.experiment_manager.target_label}")
    
    # Check initial render at 0/20
    canvas0 = app.compose_frame(app.runner.latest_telemetry or {})
    cur_f, tgt_f, prog = app.experiment_manager.get_progress()
    print(f"[UI State Initial] progress: {cur_f}/{tgt_f} ({prog*100:.0f}%)")
    
    print("\n=== STEPPING PIPELINE 25 FRAMES ===")
    completed_rec = None
    for i in range(25):
        telemetry = app.runner.step()
        raw_img = app.runner.latest_camera_frame
        ovl_img = app.runner.latest_overlaid_frame
        hmap = app.runner.latest_heatmap
        
        peak_valid = telemetry.get("peak_valid", False)
        az = telemetry.get("azimuth_deg")
        el = telemetry.get("elevation_deg")
        
        h_res = app.panel_horizontal.update_frame(
            telemetry, raw_image=raw_img, overlaid_image=ovl_img, heatmap=hmap
        )
        
        # Compose UI frame to test render() during active collection
        canvas = app.compose_frame(telemetry)
        
        if app.experiment_manager.is_running():
            cur_f, tgt_f, prog = app.experiment_manager.get_progress()
            print(f"[UI Progress] RECORDING ({cur_f}/{tgt_f}) - {prog*100:.0f}%")
        
        if h_res:
            completed_rec = h_res
            print(f"\n*** TEST COMPLETED at step {i+1}! ***")
            print(f"Result: Run ID={completed_rec.run_id}, Status={completed_rec.status}, Frames={completed_rec.frame_count}")
            break
            
    assert completed_rec is not None, "FAILED: Mode 4 test did not complete within 25 frames!"
    
    # Verify post-completion render of Mode 4
    canvas_done = app.compose_frame(telemetry)
    print("[UI Post-Completion] Mode 4 rendered successfully with completed result card.")
    
    # Verify database persistence
    print("\n=== VERIFYING DATABASE PERSISTENCE ===")
    conn = sqlite3.connect(app.db.db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM experiments WHERE run_id = ?",
        (completed_rec.run_id,)
    ).fetchone()
    conn.close()
    
    assert row is not None, f"FAILED: {completed_rec.run_id} not found in SQLite experiments table!"
    print(f"[DB Verification SUCCESS]: Found {row['run_id']} | Type: {row['test_type']} | Target: {row['target_label']} | Frames: {row['frame_count']} | Status: {row['status']}")
    
    # Verify artifacts on disk
    print("\n=== VERIFYING ARTIFACTS ON DISK ===")
    run_dir = os.path.join("data", "runs", completed_rec.run_id)
    assert os.path.isdir(run_dir), f"FAILED: Run directory {run_dir} does not exist!"
    required_artifacts = [
        "summary.json",
        "representative.jpg",
        "overlay.jpg",
        "telemetry.json",
        "heatmap.npy",
    ]
    for art in required_artifacts:
        art_path = os.path.join(run_dir, art)
        assert os.path.isfile(art_path), f"FAILED: Missing artifact {art_path}"
        sz = os.path.getsize(art_path)
        print(f"  Artifact OK: {art} ({sz:,} bytes)")
        
    # Verify Mode 8 displays the new H-xxx run
    print("\n=== SWITCHING TO MODE 8: VALIDATION DASHBOARD ===")
    app.current_mode = 8
    canvas_mode8 = app.compose_frame(telemetry)
    
    recent_runs = app.repo.get_recent_experiments(limit=11)
    print(f"Mode 8 Top Run: {recent_runs[0].run_id} (Expected: {completed_rec.run_id})")
    assert recent_runs[0].run_id == completed_rec.run_id, f"FAILED: Top run in Mode 8 is {recent_runs[0].run_id}, expected {completed_rec.run_id}"
    print(f"[Mode 8 Verification SUCCESS]: {completed_rec.run_id} is displayed at the top of Mode 8!")
    
    app.runner.close()
    print("\n========================================================")
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("========================================================")

if __name__ == "__main__":
    run_trace()
