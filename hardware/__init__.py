"""
hardware
========

Hardware Device Discovery and Management for the
Real-Time Multimodal Acoustic Camera.
"""

from hardware.devices import (
    AudioDeviceInfo,
    CameraStreamInfo,
    HardwareReport,
    find_uma16_device,
    list_audio_input_devices,
    probe_all_hardware,
    resolve_camera_stream,
    test_stream_url,
    validate_audio_config,
)

__all__ = [
    "AudioDeviceInfo",
    "CameraStreamInfo",
    "HardwareReport",
    "find_uma16_device",
    "list_audio_input_devices",
    "probe_all_hardware",
    "resolve_camera_stream",
    "test_stream_url",
    "validate_audio_config",
]

