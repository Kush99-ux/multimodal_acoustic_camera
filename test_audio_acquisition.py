"""
test_audio_acquisition.py
=========================

Milestone 18 test for the UMA-16 audio acquisition interface.

This test verifies:

- UMA-16 stream startup
- multichannel acquisition
- frame dimensions
- timestamps
- sequential frame IDs
- basic signal statistics
- clean stream shutdown

This test does NOT use:

- MUSIC
- steering matrices
- localization
- camera
- YOLO
- synchronization
- fusion

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import time

import numpy as np

from acoustic.audio import AudioAcquisition
from config.settings import (
    SAMPLE_RATE,
    NUM_CHANNELS,
    FRAME_SIZE,
    HOP_SIZE,
)


# ============================================================
# Configuration
# ============================================================

AUDIO_DEVICE = 28
TEST_FRAMES = 5


# ============================================================
# Main
# ============================================================


def main() -> None:

    print()
    print("=" * 72)
    print("MILESTONE 18 - UMA-16 AUDIO ACQUISITION TEST")
    print("=" * 72)

    print()
    print("[Configuration]")
    print(f"  Device       : {AUDIO_DEVICE}")
    print(f"  Sample rate  : {SAMPLE_RATE} Hz")
    print(f"  Channels     : {NUM_CHANNELS}")
    print(f"  Frame size   : {FRAME_SIZE}")
    print(f"  Hop size     : {HOP_SIZE}")
    print(f"  Test frames  : {TEST_FRAMES}")

    audio = AudioAcquisition(
        device=AUDIO_DEVICE,
        sample_rate=SAMPLE_RATE,
        channels=NUM_CHANNELS,
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
    )

    frames = []

    try:

        # ----------------------------------------------------
        # Start
        # ----------------------------------------------------

        audio.start()

        print()
        print("[PASS] UMA-16 stream started")

        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------

        print()
        print("[Audio] Warming up for 1.0 seconds...")

        time.sleep(1.0)

        # ----------------------------------------------------
        # Acquire frames
        # ----------------------------------------------------

        print()
        print("[Measurement] Acquiring audio frames...")

        for index in range(TEST_FRAMES):

            packet = audio.read(
                timeout=5.0
            )

            if packet is None:

                raise RuntimeError(
                    "Audio frame acquisition timed out."
                )

            frames.append(packet)

            print()
            print(
                f"  Frame {index + 1:02d}/{TEST_FRAMES}"
            )

            print(
                f"    ID        : {packet.frame_id}"
            )

            print(
                f"    Timestamp : {packet.timestamp:.6f}"
            )

            print(
                f"    Shape     : {packet.data.shape}"
            )

            print(
                f"    RMS       : "
                f"{np.sqrt(np.mean(packet.data ** 2)):.6f}"
            )

    finally:

        audio.stop()

    # ========================================================
    # Validation
    # ========================================================

    print()
    print("=" * 72)
    print("VALIDATION")
    print("=" * 72)

    assert len(frames) == TEST_FRAMES

    print(
        f"[PASS] Acquired {len(frames)} frames"
    )

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    for packet in frames:

        assert isinstance(
            packet.data,
            np.ndarray,
        )

        assert packet.data.shape == (
            FRAME_SIZE,
            NUM_CHANNELS,
        )

    print(
        "[PASS] All frames have correct shape "
        f"({FRAME_SIZE}, {NUM_CHANNELS})"
    )

    # --------------------------------------------------------
    # Frame IDs
    # --------------------------------------------------------

    frame_ids = [
        packet.frame_id
        for packet in frames
    ]

    assert frame_ids == list(
        range(
            frame_ids[0],
            frame_ids[0] + TEST_FRAMES,
        )
    )

    print("[PASS] Frame IDs are sequential")

    # --------------------------------------------------------
    # Timestamps
    # --------------------------------------------------------

    timestamps = [
        packet.timestamp
        for packet in frames
    ]

    for i in range(1, len(timestamps)):

        assert timestamps[i] > timestamps[i - 1]

    print("[PASS] Timestamps are monotonic")

    # --------------------------------------------------------
    # Data validity
    # --------------------------------------------------------

    for packet in frames:

        assert np.all(
            np.isfinite(packet.data)
        )

    print("[PASS] Audio samples are finite")

    # --------------------------------------------------------
    # Channel count
    # --------------------------------------------------------

    for packet in frames:

        assert packet.channel_count == NUM_CHANNELS

    print(
        f"[PASS] {NUM_CHANNELS}-channel acquisition verified"
    )

    # --------------------------------------------------------
    # Sample count
    # --------------------------------------------------------

    for packet in frames:

        assert packet.sample_count == FRAME_SIZE

    print(
        f"[PASS] {FRAME_SIZE}-sample frame size verified"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    elapsed = audio.get_elapsed_time()
    fps = audio.get_fps()

    print()
    print("[Statistics]")
    print(
        f"  Frames acquired : "
        f"{audio.get_frame_count()}"
    )

    print(
        f"  Elapsed time    : "
        f"{elapsed:.3f} s"
    )

    print(
        f"  Average FPS     : "
        f"{fps:.2f}"
    )

    # ========================================================
    # Complete
    # ========================================================

    print()
    print("=" * 72)
    print("MILESTONE 18 - AUDIO ACQUISITION TEST PASSED")
    print("=" * 72)

    print()
    print("Acoustic pipeline now has:")

    print(
        """
  UMA-16
     ↓
  AudioAcquisition
     ↓
  AudioFrame
     ↓
  MusicLocalizer
     ↓
  AcousticLocalization
     ↓
  AcousticFrame
"""
    )


if __name__ == "__main__":
    main()