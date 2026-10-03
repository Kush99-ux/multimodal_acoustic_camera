"""
Unit tests for the synthetic 16-channel audio generator using standard unittest.
"""

from __future__ import annotations
import unittest
import numpy as np

from benchmark.core.generator import SyntheticAudioGenerator
from benchmark.models.source import SoundSource, WaveformType
from benchmark.models.scenario import BenchmarkScenario


class TestSyntheticAudioGenerator(unittest.TestCase):

    def test_generator_output_shape(self):
        gen = SyntheticAudioGenerator(sample_rate=48000, frame_size=4096)
        scenario = BenchmarkScenario(
            scenario_id="unit_test_tone",
            category="test",
            description="test tone",
            sources=[SoundSource(azimuth_deg=90.0, elevation_deg=0.0, waveform=WaveformType.TONE, frequency_hz=1000.0)],
            snr_db=30.0,
        )
        signal = gen.generate(scenario)
        self.assertEqual(signal.data.shape, (4096, 16))
        self.assertEqual(signal.sample_rate, 48000)
        self.assertGreater(signal.overall_rms, 0.0)

    def test_generator_broadside_delay_symmetry(self):
        """At broadside (Az=90, El=0), symmetric microphone pairs should have zero relative delay."""
        gen = SyntheticAudioGenerator()
        src = SoundSource(azimuth_deg=90.0, elevation_deg=0.0)
        u = src.unit_vector
        self.assertTrue(np.isclose(u[0], 0.0))
        self.assertTrue(np.isclose(u[1], 1.0))
        self.assertTrue(np.isclose(u[2], 0.0))

        proj_dist = gen.mic_positions.T @ u
        self.assertTrue(np.allclose(proj_dist, 0.0))


if __name__ == "__main__":
    unittest.main()
