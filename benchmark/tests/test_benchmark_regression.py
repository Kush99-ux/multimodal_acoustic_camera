"""
Regression test for the 6 golden synthetic directions using standard unittest.
"""

from __future__ import annotations
import unittest
import numpy as np

from benchmark.variants.baseline import BaselineProductionVariant
from benchmark.core.generator import SyntheticAudioGenerator
from benchmark.models.source import SoundSource, WaveformType
from benchmark.models.scenario import BenchmarkScenario
from benchmark.core.metrics import calculate_angular_error
from benchmark.config.benchmark_config import GOLDEN_DIRECTIONS


class TestGoldenDirectionsRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.localizer = BaselineProductionVariant()
        cls.generator = SyntheticAudioGenerator()

    def test_all_golden_directions(self):
        for az, el in GOLDEN_DIRECTIONS:
            with self.subTest(az=az, el=el):
                scenario = BenchmarkScenario(
                    scenario_id=f"regression_{int(az)}_{int(el)}",
                    category="regression",
                    description=f"Direction Az={az}, El={el}",
                    sources=[SoundSource(azimuth_deg=az, elevation_deg=el, waveform=WaveformType.MULTI_TONE, frequency_hz=1500.0)],
                    snr_db=35.0,
                )
                signal = self.generator.generate(scenario)
                self.localizer.reset_state()
                heatmap, peak = self.localizer.localize(signal.data)

                est_az = float(peak["azimuth"])
                est_el = float(peak["elevation"])
                err = calculate_angular_error(az, el, est_az, est_el)

                self.assertLessEqual(
                    err,
                    1.0,
                    f"Angular error {err:.2f} deg exceeded 1.0 deg threshold (True: {az},{el} -> Est: {est_az},{est_el})",
                )
                self.assertTrue(peak["valid"])


if __name__ == "__main__":
    unittest.main()
