"""
test_hardware_devices.py
========================

Unit tests for the dynamic hardware device selection and discovery layer.

Validates:
- UMA-16 identification heuristics and confidence hierarchy
- Safety guard: strict blocking of 2-channel and 4-channel fallback
- Host API ranking when resolving multiple endpoints of the same physical device
- Explicit user selection preservation (no silent host API replacement)
- Disambiguation requirement when multiple generic 16-channel devices exist
- Camera stream URL resolution priority (explicit > config > candidates)
- Non-blocking offline stream probe behavior
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from hardware.devices import (
    AudioDeviceInfo,
    CameraStreamInfo,
    find_uma16_device,
    hostapi_rank,
    is_uma16_name,
    list_audio_input_devices,
    probe_all_hardware,
    resolve_camera_stream,
    test_stream_url,
    validate_audio_config,
)


class TestHardwareDeviceDiscovery(unittest.TestCase):
    """Test suite for hardware discovery layer."""

    def test_uma16_name_heuristics(self) -> None:
        """Verify UMA-16 name pattern matching."""
        self.assertTrue(is_uma16_name("miniDSP UMA-16"))
        self.assertTrue(is_uma16_name("miniDSP UMA-16 v2"))
        self.assertTrue(is_uma16_name("UMA16v2"))
        self.assertTrue(is_uma16_name("nanoSHARC micArray UMA-16"))
        self.assertTrue(is_uma16_name("minidsp"))

        self.assertFalse(is_uma16_name("Microphone Array (Intel Smart Sound)"))
        self.assertFalse(is_uma16_name("Realtek HD Audio Mic"))
        self.assertFalse(is_uma16_name("Primary Sound Capture Driver"))

    def test_hostapi_ranking(self) -> None:
        """Verify WASAPI > WDM-KS > DirectSound > MME ranking."""
        rank_wasapi = hostapi_rank("Windows WASAPI")
        rank_wdm = hostapi_rank("Windows WDM-KS")
        rank_ds = hostapi_rank("Windows DirectSound")
        rank_mme = hostapi_rank("MME")
        rank_unknown = hostapi_rank("OtherHostAPI")

        self.assertGreater(rank_wasapi, rank_wdm)
        self.assertGreater(rank_wdm, rank_ds)
        self.assertGreater(rank_ds, rank_mme)
        self.assertGreater(rank_mme, rank_unknown)

    def test_live_list_audio_devices(self) -> None:
        """Verify live enumeration returns structured AudioDeviceInfo list."""
        devices = list_audio_input_devices()
        self.assertIsInstance(devices, list)
        for d in devices:
            self.assertIsInstance(d, AudioDeviceInfo)
            self.assertGreater(d.max_input_channels, 0)
            self.assertIsInstance(d.name, str)
            self.assertIsInstance(d.hostapi_name, str)
            self.assertIsInstance(d.is_uma16, bool)
            self.assertIsInstance(d.supports_production_config, bool)

    def test_safety_guard_blocks_laptop_mics(self) -> None:
        """Ensure system NEVER falls back to 2ch or 4ch laptop microphones."""
        mock_devices = [
            AudioDeviceInfo(
                index=0,
                name="Microphone (Realtek HD Audio)",
                max_input_channels=2,
                default_samplerate=44100.0,
                hostapi_index=0,
                hostapi_name="MME",
                is_uma16=False,
                supports_production_config=False,
            ),
            AudioDeviceInfo(
                index=1,
                name="Microphone Array (Intel Smart Sound)",
                max_input_channels=4,
                default_samplerate=48000.0,
                hostapi_index=1,
                hostapi_name="Windows DirectSound",
                is_uma16=False,
                supports_production_config=False,
            ),
        ]
        with patch("hardware.devices.list_audio_input_devices", return_value=mock_devices):
            selected, reason = find_uma16_device()
            self.assertIsNone(selected)
            self.assertIn("blocked", reason.lower())

    def test_auto_detect_uma16_with_hostapi_selection(self) -> None:
        """When UMA-16 exists across MME and WASAPI, choose WASAPI."""
        mock_devices = [
            AudioDeviceInfo(
                index=5,
                name="miniDSP UMA-16 v2",
                max_input_channels=16,
                default_samplerate=48000.0,
                hostapi_index=0,
                hostapi_name="MME",
                is_uma16=True,
                supports_production_config=True,
            ),
            AudioDeviceInfo(
                index=12,
                name="miniDSP UMA-16 v2",
                max_input_channels=16,
                default_samplerate=48000.0,
                hostapi_index=2,
                hostapi_name="Windows WASAPI",
                is_uma16=True,
                supports_production_config=True,
            ),
        ]
        with patch("hardware.devices.list_audio_input_devices", return_value=mock_devices):
            selected, reason = find_uma16_device()
            self.assertIsNotNone(selected)
            self.assertEqual(selected.index, 12)
            self.assertEqual(selected.hostapi_name, "Windows WASAPI")
            self.assertIn("Auto-detected UMA-16", reason)

    def test_explicit_user_selection_by_index_honored(self) -> None:
        """Explicit index must NOT be silently replaced by another host API."""
        mock_devices = [
            AudioDeviceInfo(
                index=5,
                name="miniDSP UMA-16 v2",
                max_input_channels=16,
                default_samplerate=48000.0,
                hostapi_index=0,
                hostapi_name="MME",
                is_uma16=True,
                supports_production_config=True,
            ),
            AudioDeviceInfo(
                index=12,
                name="miniDSP UMA-16 v2",
                max_input_channels=16,
                default_samplerate=48000.0,
                hostapi_index=2,
                hostapi_name="Windows WASAPI",
                is_uma16=True,
                supports_production_config=True,
            ),
        ]
        with patch("hardware.devices.list_audio_input_devices", return_value=mock_devices):
            # User explicitly requested index 5 (MME endpoint)
            selected, reason = find_uma16_device(preference=5)
            self.assertIsNotNone(selected)
            self.assertEqual(selected.index, 5)
            self.assertEqual(selected.hostapi_name, "MME")

    def test_explicit_user_selection_rejects_incompatible(self) -> None:
        """Explicit user selection of a 2-channel device must be rejected with error."""
        mock_devices = [
            AudioDeviceInfo(
                index=1,
                name="Microphone (Realtek)",
                max_input_channels=2,
                default_samplerate=44100.0,
                hostapi_index=0,
                hostapi_name="MME",
                is_uma16=False,
                supports_production_config=False,
                validation_message="Insufficient input channels: 2",
            ),
        ]
        with patch("hardware.devices.list_audio_input_devices", return_value=mock_devices):
            selected, reason = find_uma16_device(preference=1)
            self.assertIsNone(selected)
            self.assertIn("does NOT support", reason)

    def test_disambiguation_required_for_multiple_unknown_16ch_devices(self) -> None:
        """If multiple different 16ch devices exist and none has UMA-16 name, require selection."""
        mock_devices = [
            AudioDeviceInfo(
                index=10,
                name="Studio Interface A 16ch",
                max_input_channels=16,
                default_samplerate=48000.0,
                hostapi_index=2,
                hostapi_name="Windows WASAPI",
                is_uma16=False,
                supports_production_config=True,
            ),
            AudioDeviceInfo(
                index=15,
                name="Stage Box B 16ch",
                max_input_channels=16,
                default_samplerate=48000.0,
                hostapi_index=2,
                hostapi_name="Windows WASAPI",
                is_uma16=False,
                supports_production_config=True,
            ),
        ]
        with patch("hardware.devices.list_audio_input_devices", return_value=mock_devices):
            selected, reason = find_uma16_device()
            self.assertIsNone(selected)
            self.assertIn("Multiple 16-channel devices found", reason)

    def test_camera_stream_priority(self) -> None:
        """ESP32-CAM priority: explicit --stream-url > config.settings.STREAM_URL > candidates."""
        # 1. When explicit URL succeeds, it is returned
        with patch("hardware.devices.test_stream_url") as mock_test:
            mock_test.side_effect = lambda url: CameraStreamInfo(
                url=url,
                is_online=(url == "http://explicit.local:81/stream"),
                resolution=(320, 240),
            )
            url, res = resolve_camera_stream(explicit_url="http://explicit.local:81/stream")
            self.assertEqual(url, "http://explicit.local:81/stream")
            self.assertTrue(res.is_online)

        # 2. When explicit fails, config STREAM_URL is tested next
        with patch("hardware.devices.test_stream_url") as mock_test:
            mock_test.side_effect = lambda url: CameraStreamInfo(
                url=url,
                is_online=(url == "http://10.243.98.165:81/stream"),
                resolution=(320, 240),
            )
            url, res = resolve_camera_stream(
                explicit_url="http://offline.local:81/stream",
                candidate_urls=["http://candidate.local:81/stream"],
            )
            self.assertEqual(url, "http://10.243.98.165:81/stream")
            self.assertTrue(res.is_online)

    def test_nonblocking_offline_stream_probe(self) -> None:
        """Verify test_stream_url fails gracefully on unreachable IP in under 2 seconds."""
        info = test_stream_url("http://192.0.2.1:81/stream", timeout=0.5)
        self.assertFalse(info.is_online)
        self.assertIsNotNone(info.error_message)


if __name__ == "__main__":
    unittest.main()
