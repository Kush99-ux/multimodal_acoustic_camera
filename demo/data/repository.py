"""
demo.data.repository
====================
CRUD operations, queries, statistics, export, and import for the experiment database.
"""

from __future__ import annotations
import csv
import json
import os
import re
import shutil
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from demo.data.database import ExperimentDatabase
from demo.data.models import ExperimentRecord, ArtifactRecord, DatabaseStats


class ExperimentRepository:
    """Repository handling all database reads, writes, statistics, and exports."""

    def __init__(self, db: Optional[ExperimentDatabase] = None) -> None:
        self.db = db if db is not None else ExperimentDatabase()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------
    def register_session(self, session_id: str, notes: Optional[str] = None) -> None:
        """Register a new active working session."""
        now = datetime.now().isoformat()
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, start_time, notes)
                VALUES (?, ?, ?);
                """,
                (session_id, now, notes),
            )

    def start_session(self, session_id: str, mode: str = "LIVE", notes: Optional[str] = None) -> None:
        """Convenience alias for register_session."""
        combined_notes = f"[{mode}] {notes}" if notes else f"[{mode}]"
        self.register_session(session_id, notes=combined_notes)

    def end_session(self, session_id: str) -> None:
        """Record the conclusion of a working session."""
        now = datetime.now().isoformat()
        with self.db.connection() as conn:
            conn.execute(
                """
                UPDATE sessions SET end_time = ? WHERE session_id = ?;
                """,
                (now, session_id),
            )

    # -------------------------------------------------------------------------
    # Run ID Generation
    # -------------------------------------------------------------------------
    def get_next_run_id(self, test_type: str) -> str:
        """
        Generate the next sequential unique Run ID for horizontal ('H-XXX')
        or vertical ('V-XXX') validation trials.
        Survives restarts and never resets or overwrites existing IDs.
        """
        prefix = "H" if test_type.lower().startswith("h") else "V"
        pattern = f"{prefix}-%"

        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT run_id FROM experiments WHERE run_id LIKE ?;",
                (pattern,),
            ).fetchall()

        max_seq = 0
        regex = re.compile(rf"^{prefix}-(\d+)$")
        for r in rows:
            m = regex.match(r["run_id"])
            if m:
                try:
                    num = int(m.group(1))
                    if num > max_seq:
                        max_seq = num
                except ValueError:
                    pass

        next_seq = max_seq + 1
        return f"{prefix}-{next_seq:03d}"

    # -------------------------------------------------------------------------
    # Insertion & Transactions
    # -------------------------------------------------------------------------
    def insert_experiment(
        self,
        record: Union[ExperimentRecord, dict],
        artifacts: Optional[List[ArtifactRecord]] = None,
    ) -> int:
        """
        Insert a completed experiment trial and its associated artifacts inside an atomic transaction.
        Accepts either an ExperimentRecord instance or a dictionary.
        Returns the inserted experiment database row ID.
        """
        if isinstance(record, dict):
            record = ExperimentRecord(
                run_id=record["run_id"],
                session_id=record.get("session_id", "SES-DEFAULT"),
                timestamp=record.get("timestamp", datetime.now().isoformat()),
                test_type=record.get("test_type", "horizontal"),
                target_label=record.get("target_label", record.get("label", "TARGET")),
                frame_count=record.get("frame_count", record.get("frames_collected", 20)),
                median_azimuth=float(record.get("median_azimuth", 0.0)),
                median_elevation=float(record.get("median_elevation", 0.0)),
                azimuth_std=float(record.get("azimuth_std", 0.0)),
                elevation_std=float(record.get("elevation_std", 0.0)),
                ground_truth_x=record.get("ground_truth_x"),
                ground_truth_y=record.get("ground_truth_y"),
                pixel_error=float(record["pixel_error"]) if record.get("pixel_error") is not None else None,
                sync_percentage=float(record.get("sync_percentage", 100.0)),
                median_timestamp_difference_ms=float(record.get("median_timestamp_difference_ms", record.get("mean_sync_offset_ms", 0.0))),
                max_timestamp_difference_ms=float(record.get("max_timestamp_difference_ms", 0.0)),
                status=record.get("status", record.get("outcome_status", "PASS")),
                notes=record.get("notes"),
                config_metadata=record.get("config_metadata"),
            )

        with self.db.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO experiments (
                    run_id, session_id, timestamp, test_type, target_label,
                    ground_truth_x, ground_truth_y, frame_count,
                    median_azimuth, median_elevation, azimuth_std, elevation_std,
                    pixel_error, sync_percentage, median_timestamp_difference_ms,
                    max_timestamp_difference_ms, status, notes, config_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.run_id,
                    record.session_id,
                    record.timestamp,
                    record.test_type,
                    record.target_label,
                    record.ground_truth_x,
                    record.ground_truth_y,
                    record.frame_count,
                    record.median_azimuth,
                    record.median_elevation,
                    record.azimuth_std,
                    record.elevation_std,
                    record.pixel_error,
                    record.sync_percentage,
                    record.median_timestamp_difference_ms,
                    record.max_timestamp_difference_ms,
                    record.status,
                    record.notes,
                    record.config_metadata,
                ),
            )
            inserted_id = cur.lastrowid

            if artifacts:
                for art in artifacts:
                    conn.execute(
                        """
                        INSERT INTO artifacts (run_id, artifact_type, file_path, created_at)
                        VALUES (?, ?, ?, ?);
                        """,
                        (art.run_id, art.artifact_type, art.file_path, art.created_at),
                    )

        return inserted_id

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    def get_recent_experiments(self, limit: int = 11) -> List[ExperimentRecord]:
        """
        Retrieve approximately the top 10-11 most recent trials from the database.
        Ordered by timestamp descending, then id descending.
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, run_id, session_id, timestamp, test_type, target_label,
                       ground_truth_x, ground_truth_y, frame_count,
                       median_azimuth, median_elevation, azimuth_std, elevation_std,
                       pixel_error, sync_percentage, median_timestamp_difference_ms,
                       max_timestamp_difference_ms, status, notes, config_metadata
                FROM experiments
                ORDER BY timestamp DESC, id DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()

        return [ExperimentRecord.from_row(tuple(r)) for r in rows]

    def get_all_experiments(self) -> List[ExperimentRecord]:
        """Retrieve complete experiment history from the database."""
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, run_id, session_id, timestamp, test_type, target_label,
                       ground_truth_x, ground_truth_y, frame_count,
                       median_azimuth, median_elevation, azimuth_std, elevation_std,
                       pixel_error, sync_percentage, median_timestamp_difference_ms,
                       max_timestamp_difference_ms, status, notes, config_metadata
                FROM experiments
                ORDER BY timestamp DESC, id DESC;
                """
            ).fetchall()

        return [ExperimentRecord.from_row(tuple(r)) for r in rows]

    def get_experiment_by_run_id(self, run_id: str) -> Optional[ExperimentRecord]:
        """Look up an experiment by its unique Run ID."""
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT id, run_id, session_id, timestamp, test_type, target_label,
                       ground_truth_x, ground_truth_y, frame_count,
                       median_azimuth, median_elevation, azimuth_std, elevation_std,
                       pixel_error, sync_percentage, median_timestamp_difference_ms,
                       max_timestamp_difference_ms, status, notes, config_metadata
                FROM experiments
                WHERE run_id = ?;
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None
        return ExperimentRecord.from_row(tuple(row))

    def get_artifacts_for_run(self, run_id: str) -> List[ArtifactRecord]:
        """Retrieve all registered artifact file paths for a given trial."""
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, run_id, artifact_type, file_path, created_at
                FROM artifacts
                WHERE run_id = ?
                ORDER BY id ASC;
                """,
                (run_id,),
            ).fetchall()

        return [ArtifactRecord.from_row(tuple(r)) for r in rows]

    def get_experiment_by_id(self, run_id: str) -> Optional[ExperimentRecord]:
        """Convenience alias for get_experiment_by_run_id."""
        return self.get_experiment_by_run_id(run_id)

    def get_artifacts_for_experiment(self, run_id: str) -> List[ArtifactRecord]:
        """Convenience alias for get_artifacts_for_run."""
        return self.get_artifacts_for_run(run_id)

    def get_experiment_count(self) -> int:
        """Return total count of experiments in database."""
        with self.db.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM experiments;").fetchone()[0]

    def get_database_stats(self) -> DatabaseStats:
        """Compute aggregate counts for display on Mode 8."""
        with self.db.connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM experiments;").fetchone()[0]
            horizontal = conn.execute("SELECT COUNT(*) FROM experiments WHERE test_type = 'horizontal';").fetchone()[0]
            vertical = conn.execute("SELECT COUNT(*) FROM experiments WHERE test_type = 'vertical';").fetchone()[0]
            historical = conn.execute("SELECT COUNT(*) FROM experiments WHERE test_type = 'historical';").fetchone()[0]
            completed = total  # All saved trials are completed acquisition runs
            passes = conn.execute("SELECT COUNT(*) FROM experiments WHERE status = 'PASS';").fetchone()[0]
            flagged_failed = conn.execute("SELECT COUNT(*) FROM experiments WHERE status != 'PASS';").fetchone()[0]

        return DatabaseStats(
            total_experiments=total,
            horizontal_count=horizontal,
            vertical_count=vertical,
            historical_count=historical,
            completed_count=completed,
            pass_count=passes,
            flagged_failed_count=flagged_failed,
        )

    # -------------------------------------------------------------------------
    # Export / Import / Backup
    # -------------------------------------------------------------------------
    def export_csv(
        self,
        file_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Export all experiment records to a CSV file.
        If file_path is omitted, saves to 'data/exports/experiments_YYYYMMDD_HHMMSS.csv'.
        """
        if file_path is None:
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = output_dir if output_dir is not None else os.path.join(self.project_root, "data", "exports")
            os.makedirs(export_dir, exist_ok=True)
            file_path = os.path.join(export_dir, f"experiments_{now_str}.csv")
        else:
            file_path = os.path.abspath(file_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        records = self.get_all_experiments()
        # Sort in chronological order for exports
        records = sorted(records, key=lambda r: (r.timestamp, r.run_id))

        fields = [
            "run_id", "session_id", "timestamp", "test_type", "target_label",
            "frame_count", "median_azimuth", "median_elevation", "azimuth_std", "elevation_std",
            "ground_truth_x", "ground_truth_y", "pixel_error", "sync_percentage",
            "median_timestamp_difference_ms", "max_timestamp_difference_ms",
            "status", "notes", "config_metadata",
        ]

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in records:
                row_dict = rec.to_dict()
                filtered = {k: row_dict.get(k) for k in fields}
                writer.writerow(filtered)

        return file_path

    def export_json(
        self,
        file_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Export all experiment records and their associated artifacts to a structured JSON file.
        If file_path is omitted, saves to 'data/exports/experiments_YYYYMMDD_HHMMSS.json'.
        """
        if file_path is None:
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = output_dir if output_dir is not None else os.path.join(self.project_root, "data", "exports")
            os.makedirs(export_dir, exist_ok=True)
            file_path = os.path.join(export_dir, f"experiments_{now_str}.json")
        else:
            file_path = os.path.abspath(file_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        records = self.get_all_experiments()
        # Sort in chronological order for exports
        records = sorted(records, key=lambda r: (r.timestamp, r.run_id))

        payload: List[Dict[str, Any]] = []

        for rec in records:
            d = rec.to_dict()
            arts = self.get_artifacts_for_run(rec.run_id)
            d["artifacts"] = [a.to_dict() for a in arts]
            payload.append(d)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return file_path

    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """
        Create a clean online SQLite backup using the SQLite backup API.
        Guarantees zero corruption even if concurrent operations are active.
        """
        if backup_path is None:
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(self.project_root, "data", "backups")
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, f"experiments_backup_{now_str}.db")
        else:
            backup_path = os.path.abspath(backup_path)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

        src_conn = self.db.get_raw_connection()
        try:
            dst_conn = sqlite3.connect(backup_path)
            with dst_conn:
                src_conn.backup(dst_conn)
            dst_conn.close()
        finally:
            src_conn.close()

        return backup_path

    def import_json(self, file_path: str) -> int:
        """
        Import experiments from a JSON export. Validates fields, skips duplicates without overwriting.
        Returns the count of new records successfully imported.
        """
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Import file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data.get("experiments", []) if isinstance(data, dict) else data
        imported_count = 0

        for item in items:
            run_id = item.get("run_id")
            if not run_id:
                continue

            # Check for existing run_id to avoid duplicate key errors
            if self.get_experiment_by_run_id(run_id) is not None:
                continue

            rec = ExperimentRecord(
                run_id=run_id,
                session_id=item.get("session_id", "imported"),
                timestamp=item.get("timestamp", datetime.now().isoformat()),
                test_type=item.get("test_type", "imported"),
                target_label=item.get("target_label", "UNKNOWN"),
                frame_count=int(item.get("frame_count", 20)),
                median_azimuth=float(item.get("median_azimuth", 90.0)),
                median_elevation=float(item.get("median_elevation", 0.0)),
                azimuth_std=float(item.get("azimuth_std", 0.0)),
                elevation_std=float(item.get("elevation_std", 0.0)),
                ground_truth_x=item.get("ground_truth_x"),
                ground_truth_y=item.get("ground_truth_y"),
                pixel_error=item.get("pixel_error"),
                sync_percentage=float(item.get("sync_percentage", 100.0)),
                median_timestamp_difference_ms=float(item.get("median_timestamp_difference_ms", 0.0)),
                max_timestamp_difference_ms=float(item.get("max_timestamp_difference_ms", 0.0)),
                status=item.get("status", "PASS"),
                notes=item.get("notes"),
                config_metadata=item.get("config_metadata"),
            )

            artifacts: List[ArtifactRecord] = []
            for a_item in item.get("artifacts", []):
                artifacts.append(ArtifactRecord(
                    run_id=run_id,
                    artifact_type=a_item.get("artifact_type", "summary"),
                    file_path=a_item.get("file_path", ""),
                    created_at=a_item.get("created_at", rec.timestamp),
                ))

            self.insert_experiment(rec, artifacts)
            imported_count += 1

        return imported_count
