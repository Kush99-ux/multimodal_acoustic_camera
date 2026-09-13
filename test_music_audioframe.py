"""
test_music_audioframe.py
========================

Milestone 19 integration test.

Purpose
-------
Verify the complete acoustic processing chain:

    AudioAcquisition
          ↓
      AudioFrame
          ↓
    MusicLocalizer
          ↓
  AcousticLocalization
          ↓
    AcousticFrame

This test verifies that the existing MUSIC implementation can
consume the project-level AudioFrame produced by the audio
acquisition subsystem and return the project-level AcousticFrame
required by the fusion engine.

This test does NOT:
- modify microphone geometry
- modify steering vectors
- perform camera processing
- perform YOLO inference
- perform synchronization
- perform multimodal fusion
- evaluate localization accuracy

The purpose is software integration, not acoustic accuracy validation.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import numpy as np

from acoustic.audio import AudioAcquisition
from acoustic.models import AcousticFrame
from acoustic.music import MusicLocalizer


# ============================================================
# Configuration
# ============================================================

TEST_FRAMES = 1


# ============================================================
# Main test
# ============================================================

def main() -> None:

    print()
    print("=" * 72)
    print("MILESTONE 19 - MUSIC AUDIOFRAME INTEGRATION TEST")
    print("=" * 72)

    print()
    print("Pipeline:")
    print("  UMA-16")
    print("      ↓")
    print("  AudioAcquisition")
    print("      ↓")
    print("  AudioFrame")
    print("      ↓")
    print("  MusicLocalizer")
    print("      ↓")
    print("  AcousticFrame")

    print()
    print("This test does NOT evaluate localization accuracy.")
    print("It verifies the software integration boundary.")

    # --------------------------------------------------------
    # Initialize MUSIC
    # --------------------------------------------------------

    print()
    print("[System] Initializing MUSIC localizer...")

    localizer = MusicLocalizer()

    print("[PASS] MusicLocalizer initialized")

    # --------------------------------------------------------
    # Initialize audio acquisition
    # --------------------------------------------------------

    print()
    print("[Audio] Initializing UMA-16 acquisition...")

    audio = AudioAcquisition()

    # --------------------------------------------------------
    # Start audio stream
    # --------------------------------------------------------

    try:

        audio.start()

        print("[PASS] AudioAcquisition started")

        # ----------------------------------------------------
        # Acquire one AudioFrame
        # ----------------------------------------------------

        print()
        print("[Measurement] Acquiring AudioFrame...")

        audio_frame = None

        for _ in range(TEST_FRAMES):

            audio_frame = audio.read()

            if audio_frame is None:
                raise RuntimeError(
                    "AudioAcquisition returned no AudioFrame."
                )

        # ----------------------------------------------------
        # Validate AudioFrame
        # ----------------------------------------------------

        assert audio_frame is not None

        print()
        print("[PASS] AudioFrame acquired")

        print(
            f"       Frame ID    : "
            f"{audio_frame.frame_id}"
        )

        print(
            f"       Timestamp   : "
            f"{audio_frame.timestamp:.6f}"
        )

        print(
            f"       Shape       : "
            f"{audio_frame.data.shape}"
        )

        print(
            f"       Sample rate : "
            f"{audio_frame.data.shape[0]}"
        )

        print(
            f"       Channels    : "
            f"{audio_frame.data.shape[1]}"
        )

        # ----------------------------------------------------
        # Basic AudioFrame validation
        # ----------------------------------------------------

        assert isinstance(
            audio_frame.data,
            np.ndarray,
        )

        assert audio_frame.data.ndim == 2

        assert (
            audio_frame.data.shape[0]
            == 4096
        )

        assert (
            audio_frame.data.shape[1]
            == 16
        )

        assert np.isfinite(
            audio_frame.data
        ).all()

        print("[PASS] AudioFrame structure validated")
        print("[PASS] 16-channel audio data validated")

        # ----------------------------------------------------
        # Run MUSIC through new integration method
        # ----------------------------------------------------

        print()
        print(
            "[MUSIC] Processing AudioFrame..."
        )

        acoustic_frame = (
            localizer.localize_audio_frame(
                audio_frame
            )
        )

        # ----------------------------------------------------
        # Validate AcousticFrame
        # ----------------------------------------------------

        assert isinstance(
            acoustic_frame,
            AcousticFrame,
        )

        print()
        print(
            "[PASS] MusicLocalizer returned "
            "AcousticFrame"
        )

        print(
            f"       Frame ID    : "
            f"{acoustic_frame.frame_id}"
        )

        print(
            f"       Timestamp   : "
            f"{acoustic_frame.timestamp:.6f}"
        )

        print(
            f"       Sample rate : "
            f"{acoustic_frame.sample_rate} Hz"
        )

        print(
            f"       Frame size  : "
            f"{acoustic_frame.frame_size}"
        )

        # ----------------------------------------------------
        # Verify timestamp propagation
        # ----------------------------------------------------

        assert (
            acoustic_frame.timestamp
            == audio_frame.timestamp
        )

        assert (
            acoustic_frame.frame_id
            == audio_frame.frame_id
        )

        print(
            "[PASS] AudioFrame timestamp preserved"
        )

        print(
            "[PASS] AudioFrame ID preserved"
        )

        # ----------------------------------------------------
        # Verify localization
        # ----------------------------------------------------

        assert (
            acoustic_frame.localization
            is not None
        )

        localization = (
            acoustic_frame.localization
        )

        print()
        print(
            "[PASS] AcousticLocalization generated"
        )

        print(
            f"       X           : "
            f"{localization.x:+.4f} m"
        )

        print(
            f"       Y           : "
            f"{localization.y:+.4f} m"
        )

        print(
            f"       Z           : "
            f"{localization.z:+.4f} m"
        )

        print(
            f"       Response    : "
            f"{localization.response:.4f}"
        )

        print(
            f"       Confidence  : "
            f"{localization.confidence:.4f}"
        )

        # ----------------------------------------------------
        # Validate localization fields
        # ----------------------------------------------------

        assert np.isfinite(
            localization.x
        )

        assert np.isfinite(
            localization.y
        )

        assert np.isfinite(
            localization.z
        )

        assert np.isfinite(
            localization.response
        )

        assert (
            0.0
            <= localization.confidence
            <= 1.0
        )

        print(
            "[PASS] Localization values are valid"
        )

        # ----------------------------------------------------
        # Verify processing metadata
        # ----------------------------------------------------

        print()
        print(
            "[PASS] Processing metadata:"
        )

        print(
            f"       Processing time : "
            f"{acoustic_frame.processing_time:.4f} s"
        )

        if acoustic_frame.metadata:

            for key, value in (
                acoustic_frame.metadata.items()
            ):

                print(
                    f"       {key} : {value}"
                )

        print()
        print(
            "[PASS] AcousticFrame metadata available"
        )

    finally:

        # ----------------------------------------------------
        # Stop audio
        # ----------------------------------------------------

        audio.stop()

        print()
        print("[Audio] Stream stopped")

    # ========================================================
    # Complete
    # ========================================================

    print()
    print("=" * 72)
    print("MILESTONE 19 - MUSIC AUDIOFRAME INTEGRATION TEST PASSED")
    print("=" * 72)

    print()
    print("Acoustic pipeline verified:")
    print()
    print("  UMA-16")
    print("     ↓")
    print("  AudioAcquisition")
    print("     ↓")
    print("  AudioFrame")
    print("     ↓")
    print("  MusicLocalizer")
    print("     ↓")
    print("  AcousticLocalization")
    print("     ↓")
    print("  AcousticFrame")
    print()


if __name__ == "__main__":
    main()