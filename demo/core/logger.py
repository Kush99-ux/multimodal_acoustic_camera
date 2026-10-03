"""
demo.core.logger
================
Snapshot and session telemetry data logger for the Professor Demo.
Saves structured JSON and representative images to scratch/demo_results/.
Includes robust serialization for NumPy scalar types and arrays.
"""

from __future__ import annotations
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


def to_json_serializable(obj: Any) -> Any:
    """Recursively convert NumPy data types and collections into standard JSON types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


class NumpyJSONEncoder(json.JSONEncoder):
    """JSONEncoder subclass that handles NumPy scalar types and ndarrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


class DemoLogger:
    """Manages snapshot captures and continuous session recording."""

    def __init__(self, output_dir: str = "scratch/demo_results") -> None:
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.recording_session = False
        self.session_frames: List[Dict[str, Any]] = []
        self.session_start_time: float = 0.0
        self.current_session_file: Optional[str] = None

    def save_snapshot(
        self,
        frame_image: Optional[np.ndarray],
        telemetry: Dict[str, Any],
    ) -> str:
        """Save a single timestamped snapshot with JSON metadata and optional JPG."""
        now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"snapshot_{now_str}"
        json_path = os.path.join(self.output_dir, f"{base_name}.json")
        img_path = os.path.join(self.output_dir, f"{base_name}.jpg")

        saved_img = False
        if frame_image is not None and frame_image.size > 0:
            try:
                cv2.imwrite(img_path, frame_image)
                saved_img = True
            except Exception as e:
                print(f"[DemoLogger] Warning: Failed to save snapshot image: {e}")

        payload = {
            "record_type": "snapshot",
            "saved_at": datetime.now().isoformat(),
            "image_path": img_path if saved_img else None,
            "telemetry": to_json_serializable(telemetry),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, cls=NumpyJSONEncoder)

        print(f"[DemoLogger] Snapshot saved to {json_path}")
        return json_path

    def start_session(self) -> str:
        """Begin a continuous telemetry recording session."""
        self.recording_session = True
        self.session_frames = []
        self.session_start_time = time.perf_counter()
        now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_session_file = os.path.join(self.output_dir, f"session_{now_str}.json")
        return self.current_session_file

    def record_frame(self, telemetry: Dict[str, Any]) -> None:
        """Record one frame's telemetry if session recording is active."""
        if not self.recording_session:
            return
        frame_record = {
            "elapsed_s": round(time.perf_counter() - self.session_start_time, 4),
            "telemetry": to_json_serializable(telemetry),
        }
        self.session_frames.append(frame_record)

    def stop_session(self) -> Optional[str]:
        """Stop session recording and write accumulated frames to JSON."""
        if not self.recording_session or not self.current_session_file:
            self.recording_session = False
            return None

        self.recording_session = False
        duration = round(time.perf_counter() - self.session_start_time, 2)
        payload = {
            "record_type": "session",
            "recorded_at": datetime.now().isoformat(),
            "duration_seconds": duration,
            "frame_count": len(self.session_frames),
            "frames": self.session_frames,
        }

        with open(self.current_session_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, cls=NumpyJSONEncoder)

        path = self.current_session_file
        print(f"[DemoLogger] Session recording saved to {path}")
        self.current_session_file = None
        self.session_frames = []
        return path
