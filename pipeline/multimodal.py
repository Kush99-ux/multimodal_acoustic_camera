"""
pipeline.multimodal
===================

Milestone 20 real-time multimodal orchestration.

This module composes the existing camera, vision, audio, acoustic,
synchronization, and fusion components. It deliberately performs no
spatial projection or object-to-sound association.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

from acoustic.audio import AudioAcquisition, AudioFrame
from acoustic.models import AcousticFrame
from acoustic.music import MusicLocalizer
from vision.camera import CameraAcquisition, CameraFrame
from config.settings import (
    MULTIMODAL_ACOUSTIC_BUFFER_SIZE,
    MULTIMODAL_AUDIO_READ_TIMEOUT_SECONDS,
)
from fusion.engine import FusionEngine
from fusion.models import MultimodalResult
from fusion.synchronization import TimestampSynchronizer
from vision.models import VisionFrame
from vision.segmentation import YOLOSegmenter


class MultimodalPipeline:
    """Bounded live orchestration of the existing M13--M19 components.

    Optional component injection keeps the orchestration testable without
    ESP32-CAM or UMA-16 hardware. Production callers normally use defaults.
    """

    def __init__(
        self,
        *,
        camera: Optional[CameraAcquisition] = None,
        segmenter: Optional[YOLOSegmenter] = None,
        audio: Optional[AudioAcquisition] = None,
        localizer: Optional[MusicLocalizer] = None,
        synchronizer: Optional[TimestampSynchronizer] = None,
        fusion_engine: Optional[FusionEngine] = None,
        acoustic_buffer_size: int = MULTIMODAL_ACOUSTIC_BUFFER_SIZE,
        audio_read_timeout: float = MULTIMODAL_AUDIO_READ_TIMEOUT_SECONDS,
    ) -> None:
        if acoustic_buffer_size <= 0:
            raise ValueError("acoustic_buffer_size must be positive.")
        if audio_read_timeout < 0:
            raise ValueError("audio_read_timeout cannot be negative.")

        self.camera = camera if camera is not None else CameraAcquisition()
        self.segmenter = segmenter if segmenter is not None else YOLOSegmenter()
        self.audio = audio if audio is not None else AudioAcquisition()
        self.localizer = localizer if localizer is not None else MusicLocalizer()
        self.synchronizer = (
            synchronizer
            if synchronizer is not None
            else TimestampSynchronizer()
        )
        self.fusion_engine = (
            fusion_engine
            if fusion_engine is not None
            else FusionEngine(synchronizer=self.synchronizer)
        )

        self.audio_read_timeout = float(audio_read_timeout)
        self.acoustic_frames: deque[AcousticFrame] = deque(
            maxlen=int(acoustic_buffer_size)
        )
        self.started = False
        self._statistics = {
            "camera_frames_acquired": 0,
            "vision_frames_processed": 0,
            "audio_frames_acquired": 0,
            "acoustic_frames_localized": 0,
            "synchronization_attempts": 0,
            "successful_synchronized_pairs": 0,
            "fusion_results": 0,
            "rejected_unsynchronized_frames": 0,
        }

    def start(self) -> None:
        """Start the camera and audio acquisition resources."""
        if self.started:
            return
        self.camera.start()
        try:
            self.audio.start()
        except Exception:
            self.camera.stop()
            raise
        self.started = True

    def stop(self) -> None:
        """Release audio and camera resources; safe to call repeatedly."""
        try:
            self.audio.stop()
        finally:
            self.camera.stop()
            self.started = False

    def __enter__(self) -> "MultimodalPipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def build_vision_frame(self, camera_frame: CameraFrame) -> VisionFrame:
        """Run segmentation and preserve camera acquisition identity/time."""
        image = camera_frame.frame
        height, width = image.shape[:2]
        start_time = time.perf_counter()
        detections = self.segmenter.detect(image)
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        self._statistics["vision_frames_processed"] += 1
        return VisionFrame(
            timestamp=camera_frame.timestamp,
            frame_id=camera_frame.frame_id,
            detections=detections,
            width=int(width),
            height=int(height),
            inference_time_ms=inference_time_ms,
        )

    def ingest_audio_frame(self, audio_frame: AudioFrame) -> AcousticFrame:
        """Localize an acquired frame and retain its bounded result."""
        acoustic_frame = self.localizer.localize_audio_frame(audio_frame)
        self.acoustic_frames.append(acoustic_frame)
        self._statistics["acoustic_frames_localized"] += 1
        return acoustic_frame

    def process_vision_frame(
        self,
        vision_frame: VisionFrame,
    ) -> Optional[MultimodalResult]:
        """Match one VisionFrame against buffered acoustic frames and fuse."""
        self._statistics["synchronization_attempts"] += 1
        pair = self.synchronizer.match(vision_frame, self.acoustic_frames)
        if pair is None:
            self._statistics["rejected_unsynchronized_frames"] += 1
            return None

        self._statistics["successful_synchronized_pairs"] += 1
        result = self.fusion_engine.fuse(pair.vision, pair.acoustic)
        if result is None:
            self._statistics["rejected_unsynchronized_frames"] += 1
            return None

        self._statistics["fusion_results"] += 1
        return result

    def run_once(self) -> Optional[MultimodalResult]:
        """Perform one finite live acquisition and fusion iteration.

        Audio is acquired before the camera frame and localized afterwards.
        This keeps the two acquisition timestamps close while avoiding an
        additional copy of either the image or audio data.
        """
        if not self.started:
            raise RuntimeError("Pipeline is not started. Call start() first.")

        audio_frame = self.audio.read(timeout=self.audio_read_timeout)
        if audio_frame is not None:
            self._statistics["audio_frames_acquired"] += 1

        camera_frame = self.camera.read()
        if camera_frame is None:
            return None
        self._statistics["camera_frames_acquired"] += 1
        vision_frame = self.build_vision_frame(camera_frame)

        if audio_frame is not None:
            self.ingest_audio_frame(audio_frame)

        return self.process_vision_frame(vision_frame)

    def run_forever(self) -> None:
        """Run until Ctrl+C, always releasing camera and audio resources."""
        self.start()
        try:
            while True:
                result = self.run_once()
                if result is not None:
                    self.log_result(result)
        except KeyboardInterrupt:
            print("[Pipeline] Stopped by user")
        finally:
            self.stop()

    @staticmethod
    def log_result(result: MultimodalResult) -> None:
        """Print the M20 structured-result summary for live observation."""
        print("[Pipeline]")
        print(f"Vision frame ID: {result.vision.frame_id}")
        print(f"Vision timestamp: {result.vision.timestamp:.6f}")
        print(f"Number of detections: {result.vision_detection_count}")
        print(f"Acoustic frame ID: {result.acoustic.frame_id}")
        print(f"Acoustic timestamp: {result.acoustic.timestamp:.6f}")
        print(f"Timestamp difference: {result.timestamp_difference_ms:.3f} ms")
        print(f"Acoustic localization available: {result.acoustic_available}")
        print(f"Fusion confidence: {result.fusion_confidence:.3f}")
        print(f"Fusion status: {result.status}")

    def get_statistics(self) -> dict[str, int]:
        """Return a snapshot of M20 orchestration statistics."""
        return dict(self._statistics)
