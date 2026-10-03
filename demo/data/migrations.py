"""
demo.data.migrations
====================
One-time historical validation data migration into the SQLite database.
Seeds the verified physical validation campaigns (Run 2, Run 4, Run 5, Run 6, Run 7)
without recalculating or modifying the original experimental values.
"""

from __future__ import annotations
import os
from typing import List

from demo.data.models import ExperimentRecord, ArtifactRecord
from demo.data.repository import ExperimentRepository


def migrate_historical_records(repo: ExperimentRepository) -> int:
    """
    Seed the verified historical validation campaigns from the project report into SQLite.
    Idempotent: skips any record whose run_id already exists.
    Returns the count of new records inserted.
    """
    historical_data = [
        {
            "run_id": "Run 2",
            "session_id": "historical-campaign-01",
            "timestamp": "2026-09-14T10:15:00",
            "test_type": "horizontal",
            "target_label": "LEFT",
            "frame_count": 20,
            "median_azimuth": 107.5,
            "median_elevation": -8.75,
            "azimuth_std": 1.62,
            "elevation_std": 0.75,
            "ground_truth_x": 146,
            "ground_truth_y": 216,
            "pixel_error": 77.3,
            "sync_percentage": 100.0,
            "median_timestamp_difference_ms": 28.52,
            "max_timestamp_difference_ms": 49.47,
            "status": "PASS",
            "notes": "Absorber-Shielded LEFT (Towel)",
            "config_metadata": '{"preset": "shielded_left", "source": "final_validation_report"}',
            "artifacts": [
                ("representative_image", "scratch/left_final_validation_representative.jpg"),
                ("raw_image", "scratch/left_final_validation_raw.jpg"),
            ],
        },
        {
            "run_id": "Run 4",
            "session_id": "historical-campaign-01",
            "timestamp": "2026-09-14T11:30:00",
            "test_type": "horizontal",
            "target_label": "CENTER",
            "frame_count": 20,
            "median_azimuth": 86.0,
            "median_elevation": -3.50,
            "azimuth_std": 4.92,
            "elevation_std": 8.88,
            "ground_truth_x": 320,
            "ground_truth_y": 240,
            "pixel_error": 101.7,
            "sync_percentage": 100.0,
            "median_timestamp_difference_ms": 29.10,
            "max_timestamp_difference_ms": 48.15,
            "status": "PASS",
            "notes": "True Physical CENTER",
            "config_metadata": '{"preset": "center", "source": "final_validation_report"}',
            "artifacts": [
                ("representative_image", "scratch/center_final_validation_representative.jpg"),
                ("raw_image", "scratch/center_final_validation_raw.jpg"),
            ],
        },
        {
            "run_id": "Run 5",
            "session_id": "historical-campaign-01",
            "timestamp": "2026-09-14T13:00:00",
            "test_type": "horizontal",
            "target_label": "CENTER",
            "frame_count": 20,
            "median_azimuth": 94.0,
            "median_elevation": -11.00,
            "azimuth_std": 3.04,
            "elevation_std": 2.10,
            "ground_truth_x": 320,
            "ground_truth_y": 300,
            "pixel_error": 75.8,
            "sync_percentage": 100.0,
            "median_timestamp_difference_ms": 27.80,
            "max_timestamp_difference_ms": 46.20,
            "status": "PASS",
            "notes": "Controlled Lower-Center",
            "config_metadata": '{"preset": "lower_center", "source": "final_validation_report"}',
            "artifacts": [],
        },
        {
            "run_id": "Run 6",
            "session_id": "historical-campaign-01",
            "timestamp": "2026-09-14T14:15:00",
            "test_type": "horizontal",
            "target_label": "RIGHT",
            "frame_count": 20,
            "median_azimuth": 86.2,
            "median_elevation": -18.75,
            "azimuth_std": 2.17,
            "elevation_std": 0.73,
            "ground_truth_x": 490,
            "ground_truth_y": 240,
            "pixel_error": 206.4,
            "sync_percentage": 100.0,
            "median_timestamp_difference_ms": 30.40,
            "max_timestamp_difference_ms": 52.10,
            "status": "FLAGGED",
            "notes": "Unshielded True RIGHT (Tabletop multipath reflection)",
            "config_metadata": '{"preset": "unshielded_right", "source": "final_validation_report"}',
            "artifacts": [],
        },
        {
            "run_id": "Run 7",
            "session_id": "historical-campaign-01",
            "timestamp": "2026-09-14T15:45:00",
            "test_type": "horizontal",
            "target_label": "RIGHT",
            "frame_count": 20,
            "median_azimuth": 68.5,
            "median_elevation": -18.75,
            "azimuth_std": 11.53,
            "elevation_std": 4.74,
            "ground_truth_x": 550,
            "ground_truth_y": 360,
            "pixel_error": 135.2,
            "sync_percentage": 100.0,
            "median_timestamp_difference_ms": 28.90,
            "max_timestamp_difference_ms": 49.80,
            "status": "PASS",
            "notes": "Lower-Right Corner (Unshielded)",
            "config_metadata": '{"preset": "lower_right", "source": "final_validation_report"}',
            "artifacts": [],
        },
    ]

    inserted = 0
    project_root = repo.project_root

    for entry in historical_data:
        run_id = entry["run_id"]
        # Skip if already in database
        if repo.get_experiment_by_run_id(run_id) is not None:
            continue

        rec = ExperimentRecord(
            run_id=run_id,
            session_id=entry["session_id"],
            timestamp=entry["timestamp"],
            test_type=entry["test_type"],
            target_label=entry["target_label"],
            frame_count=entry["frame_count"],
            median_azimuth=entry["median_azimuth"],
            median_elevation=entry["median_elevation"],
            azimuth_std=entry["azimuth_std"],
            elevation_std=entry["elevation_std"],
            ground_truth_x=entry["ground_truth_x"],
            ground_truth_y=entry["ground_truth_y"],
            pixel_error=entry["pixel_error"],
            sync_percentage=entry["sync_percentage"],
            median_timestamp_difference_ms=entry["median_timestamp_difference_ms"],
            max_timestamp_difference_ms=entry["max_timestamp_difference_ms"],
            status=entry["status"],
            notes=entry["notes"],
            config_metadata=entry["config_metadata"],
        )

        artifacts: List[ArtifactRecord] = []
        for art_type, rel_path in entry["artifacts"]:
            full_path = os.path.join(project_root, rel_path)
            if os.path.isfile(full_path):
                artifacts.append(
                    ArtifactRecord(
                        run_id=run_id,
                        artifact_type=art_type,
                        file_path=rel_path.replace("\\", "/"),
                        created_at=entry["timestamp"],
                    )
                )

        repo.insert_experiment(rec, artifacts)
        inserted += 1

    return inserted
