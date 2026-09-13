"""Hardware-free Milestone 20 orchestration integration test."""

from __future__ import annotations

import numpy as np

from acoustic.audio import AudioFrame
from acoustic.models import AcousticFrame, AcousticLocalization
from vision.camera import CameraFrame
from pipeline.multimodal import MultimodalPipeline
from vision.models import VisionDetection, VisionFrame


class FakeCamera:
    def __init__(self) -> None:
        self.started = False
        self.frame = CameraFrame(
            frame=np.zeros((240, 320, 3), dtype=np.uint8),
            timestamp=100.050,
            frame_id=7,
        )

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def read(self) -> CameraFrame:
        return self.frame


class FakeSegmenter:
    def detect(self, image: np.ndarray) -> list[VisionDetection]:
        return [
            VisionDetection(
                class_id=0,
                class_name="object",
                confidence=0.9,
                bbox=(1.0, 2.0, 30.0, 40.0),
                bbox_centroid=(15.5, 21.0),
            )
        ]


class FakeAudio:
    def __init__(self) -> None:
        self.started = False
        self.frame = AudioFrame(
            data=np.zeros((4, 16), dtype=np.float32),
            timestamp=100.000,
            frame_id=3,
        )

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def read(self, timeout: float | None = None) -> AudioFrame:
        return self.frame


class FakeLocalizer:
    def localize_audio_frame(self, audio_frame: AudioFrame) -> AcousticFrame:
        return AcousticFrame(
            frame_id=audio_frame.frame_id,
            timestamp=audio_frame.timestamp,
            localization=AcousticLocalization(
                x=0.0,
                y=0.0,
                z=1.0,
                response=0.8,
                confidence=0.8,
            ),
            sample_rate=48000.0,
            frame_size=4,
        )


def main() -> None:
    camera = FakeCamera()
    audio = FakeAudio()
    pipeline = MultimodalPipeline(
        camera=camera,
        segmenter=FakeSegmenter(),
        audio=audio,
        localizer=FakeLocalizer(),
        acoustic_buffer_size=2,
        audio_read_timeout=0.0,
    )

    pipeline.start()
    assert camera.started and audio.started

    result = pipeline.run_once()
    assert result is not None
    assert result.vision.frame_id == 7
    assert result.vision.detection_count == 1
    assert result.acoustic.frame_id == 3
    assert np.isclose(result.timestamp_difference_ms, 50.0)
    assert len(pipeline.acoustic_frames) == 1

    pipeline.ingest_audio_frame(AudioFrame(np.zeros((4, 16)), 100.060, 4))
    pipeline.ingest_audio_frame(AudioFrame(np.zeros((4, 16)), 100.070, 5))
    assert len(pipeline.acoustic_frames) == 2
    assert pipeline.acoustic_frames[0].frame_id == 4

    closest_result = pipeline.process_vision_frame(
        VisionFrame(100.058, 8, [], 320, 240)
    )
    assert closest_result is not None
    assert closest_result.acoustic.frame_id == 4

    rejected = pipeline.process_vision_frame(
        VisionFrame(200.0, 9, [], 320, 240)
    )
    assert rejected is None

    statistics = pipeline.get_statistics()
    assert statistics["camera_frames_acquired"] == 1
    assert statistics["vision_frames_processed"] == 1
    assert statistics["audio_frames_acquired"] == 1
    assert statistics["acoustic_frames_localized"] == 3
    assert statistics["synchronization_attempts"] == 3
    assert statistics["successful_synchronized_pairs"] == 2
    assert statistics["fusion_results"] == 2
    assert statistics["rejected_unsynchronized_frames"] == 1

    pipeline.stop()
    assert not camera.started and not audio.started
    print("MILESTONE 20 SOFTWARE TEST PASSED")


if __name__ == "__main__":
    main()
