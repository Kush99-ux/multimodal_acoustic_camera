"""
demo.data.models
================
Data models for persistent experiment tracking, validation trials, and artifacts.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import os
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRecord:
    """Represents a single completed validation experiment trial."""

    run_id: str
    session_id: str
    timestamp: str
    test_type: str  # 'horizontal', 'vertical', 'historical'
    target_label: str  # 'LEFT', 'CENTER', 'RIGHT', 'ABOVE', 'LEVEL', 'BELOW'
    frame_count: int
    median_azimuth: float
    median_elevation: float
    azimuth_std: float
    elevation_std: float
    ground_truth_x: Optional[int] = None
    ground_truth_y: Optional[int] = None
    pixel_error: Optional[float] = None
    sync_percentage: float = 100.0
    median_timestamp_difference_ms: float = 0.0
    max_timestamp_difference_ms: float = 0.0
    status: str = "PASS"  # 'PASS', 'OBSERVED', 'FLAGGED', 'FAILED'
    notes: Optional[str] = None
    config_metadata: Optional[str] = None
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        d = asdict(self)
        return d

    @property
    def label(self) -> str:
        """Alias for target_label."""
        return self.target_label

    @property
    def outcome_status(self) -> str:
        """Alias for status."""
        return self.status

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style key access."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Allow dict-style get access."""
        return getattr(self, key, default)

    @classmethod
    def from_row(cls, row: tuple) -> ExperimentRecord:
        """Create record from SQLite SELECT row."""
        return cls(
            id=row[0],
            run_id=row[1],
            session_id=row[2],
            timestamp=row[3],
            test_type=row[4],
            target_label=row[5],
            ground_truth_x=row[6],
            ground_truth_y=row[7],
            frame_count=row[8],
            median_azimuth=row[9],
            median_elevation=row[10],
            azimuth_std=row[11],
            elevation_std=row[12],
            pixel_error=row[13],
            sync_percentage=row[14],
            median_timestamp_difference_ms=row[15],
            max_timestamp_difference_ms=row[16],
            status=row[17],
            notes=row[18],
            config_metadata=row[19],
        )


@dataclass
class ArtifactRecord:
    """Represents a persisted file artifact tied to a specific experiment trial."""

    run_id: str
    artifact_type: str  # 'summary', 'representative_image', 'overlay_image', 'telemetry', 'heatmap'
    file_path: str  # Relative path within project root
    created_at: str
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to JSON-serializable dictionary."""
        return asdict(self)

    @property
    def file_size_bytes(self) -> int:
        """Resolve file size on disk."""
        if os.path.isabs(self.file_path):
            p = self.file_path
        else:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            p = os.path.join(project_root, self.file_path)
        return os.path.getsize(p) if os.path.isfile(p) else 0

    @classmethod
    def from_row(cls, row: tuple) -> ArtifactRecord:
        """Create artifact record from SQLite SELECT row."""
        return cls(
            id=row[0],
            run_id=row[1],
            artifact_type=row[2],
            file_path=row[3],
            created_at=row[4],
        )


@dataclass
class DatabaseStats:
    """Aggregate statistics summarizing the persistent experiment database."""

    total_experiments: int = 0
    horizontal_count: int = 0
    vertical_count: int = 0
    historical_count: int = 0
    completed_count: int = 0
    pass_count: int = 0
    flagged_failed_count: int = 0

    @property
    def total_trials(self) -> int:
        """Alias for total_experiments."""
        return self.total_experiments

    @property
    def success_rate(self) -> float:
        """Percentage of trials that achieved PASS status."""
        if self.total_experiments == 0:
            return 0.0
        return (self.pass_count / self.total_experiments) * 100.0
