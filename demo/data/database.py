"""
demo.data.database
==================
SQLite connection management, table initialization, and transaction handling.
"""

from __future__ import annotations
import os
import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional


class ExperimentDatabase:
    """Manages SQLite database connections, schema setup, and transactions."""

    DEFAULT_DB_REL_PATH = os.path.join("data", "experiments.db")

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            # Resolve relative to project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.db_path = os.path.join(project_root, self.DEFAULT_DB_REL_PATH)
        else:
            self.db_path = os.path.abspath(db_path)

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize schema upon construction
        self.init_schema()

    def get_raw_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Enable Write-Ahead Logging for concurrency safety and faster transactions
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Convenience alias for get_raw_connection."""
        return self.get_raw_connection()

    def close(self) -> None:
        """No-op as connections are pooled/opened on-demand, but provided for interface symmetry."""
        pass

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for SQLite operations with automatic commit and rollback."""
        conn = self.get_raw_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create tables and indices if they do not already exist."""
        with self.connection() as conn:
            # Sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    notes TEXT
                );
            """)

            # Experiments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    target_label TEXT NOT NULL,
                    ground_truth_x INTEGER,
                    ground_truth_y INTEGER,
                    frame_count INTEGER NOT NULL,
                    median_azimuth REAL NOT NULL,
                    median_elevation REAL NOT NULL,
                    azimuth_std REAL NOT NULL,
                    elevation_std REAL NOT NULL,
                    pixel_error REAL,
                    sync_percentage REAL DEFAULT 100.0,
                    median_timestamp_difference_ms REAL DEFAULT 0.0,
                    max_timestamp_difference_ms REAL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    notes TEXT,
                    config_metadata TEXT
                );
            """)

            # Artifacts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES experiments(run_id) ON DELETE CASCADE
                );
            """)

            # Performance indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_timestamp ON experiments(timestamp DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_run_id ON experiments(run_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts(run_id);")
