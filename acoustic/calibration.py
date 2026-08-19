"""
acoustic.calibration
====================

Impulse-based UMA-16 channel calibration.

Tap directly above the requested physical microphone label.
The script captures a short impulse window and reports the strongest channels.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import time
import numpy as np

from acoustic.acquisition import AudioAcquisition


PHYSICAL_MIC_ORDER = [
    "MIC1", "MIC2", "MIC3", "MIC4",
    "MIC5", "MIC6", "MIC7", "MIC8",
    "MIC9", "MIC10", "MIC11", "MIC12",
    "MIC13", "MIC14", "MIC15", "MIC16",
]


class ChannelCalibrator:
    """Impulse-based physical-microphone to audio-channel calibration."""

    def __init__(self):
        self.audio = AudioAcquisition()
        self.mapping = {}

    def _wait_for_impulse(
        self,
        threshold: float = 0.08,
        timeout: float = 10.0,
    ) -> np.ndarray:
        """
        Wait until a sharp impulse is detected.

        Returns one raw audio frame containing the impulse.
        """

        start = time.time()

        while time.time() - start < timeout:
            frame = self.audio.get_frame(timeout=1.0)

            if frame is None:
                continue

            peak = np.max(np.abs(frame))

            if peak > threshold:
                return frame

        raise RuntimeError("Impulse detection timed out.")

    def _analyze_impulse(self, frame: np.ndarray):
        """
        Analyze a captured impulse frame.

        Returns
        -------
        strongest_channel : int
            Zero-based channel index.

        ranking : list[tuple[int, float]]
            Channels ranked by peak amplitude.
        """

        peaks = np.max(np.abs(frame), axis=0)

        order = np.argsort(peaks)[::-1]

        ranking = [(int(ch), float(peaks[ch])) for ch in order]

        strongest = int(order[0])

        return strongest, ranking

    def run(self):
        """Run the interactive calibration procedure."""

        self.audio.start()

        try:
            print("\\n=== UMA-16 Impulse Calibration ===\\n")
            print("Place the board on a soft surface.")
            print("Tap directly above the requested microphone label.")
            print("Use a fingernail or small plastic object for a sharp impulse.\\n")

            for mic in PHYSICAL_MIC_ORDER:

                input(f"Press ENTER when ready for {mic}...")

                print(f"Waiting for impulse on {mic}...")

                frame = self._wait_for_impulse()

                strongest, ranking = self._analyze_impulse(frame)

                self.mapping[mic] = strongest + 1

                print(f"Strongest channel: CH{strongest + 1:02d}\\n")
                print("Top 5 channels:")

                for ch, amp in ranking[:5]:
                    print(f"  CH{ch + 1:02d}: {amp:.4f}")

                print()

            print("=== Calibration Complete ===\\n")

            print("Physical microphone -> Audio channel")
            print("-" * 40)

            for mic in PHYSICAL_MIC_ORDER:
                print(f"{mic:5s} -> CH{self.mapping[mic]:02d}")

            print("-" * 40)

        finally:
            self.audio.stop()

        return self.mapping