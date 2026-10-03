"""
hardware.devices
================

Dynamic Hardware Device Selection and Discovery Layer for the
Real-Time Multimodal Acoustic Camera.

Provides discovery, configuration validation, and stream resolution for:
1. miniDSP UMA-16 16-channel USB microphone array
2. ESP32-CAM MJPEG network video stream

Safety and Architectural Rules:
- NEVER fall back to 2-channel or 4-channel PC audio devices for MUSIC localization.
- Strict confidence hierarchy:
    Exact/strong UMA-16 name match > explicit user selection > 16+ channel candidate
- Verification of 16-channel / 48000 Hz configuration before stream activation.
- Host API preference (WASAPI > WDM-KS > DirectSound > MME) used ONLY when resolving
  multiple endpoints for the same physical candidate; explicit selection always honored.
- ESP32-CAM selection priority: explicit --stream-url > config.settings.STREAM_URL > user candidates.
  Obsolete IP addresses are never automatically probed.
- Zero modification to acoustic MUSIC, steering geometry, CSM/EMA, or latest-frame buffering.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import socket
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import urllib.parse

# Ensure repository root is on sys.path when executed directly
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import sounddevice as sd

from config.settings import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    NUM_CHANNELS,
    SAMPLE_RATE,
    STREAM_URL,
)


# Known name patterns identifying miniDSP UMA-16 hardware
UMA16_NAME_PATTERNS = ("uma-16", "uma16", "minidsp", "nanosharc")


@dataclass
class AudioDeviceInfo:
    """Detailed record of an enumerated audio input device."""
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float
    hostapi_index: int
    hostapi_name: str
    is_uma16: bool
    supports_production_config: bool
    validation_message: str = ""

    def summary_line(self) -> str:
        """Format a single-line readable summary for console output."""
        uma_badge = " [UMA-16]" if self.is_uma16 else ""
        prod_badge = " [16ch/48k OK]" if self.supports_production_config else " [INCOMPATIBLE]"
        return (
            f"[{self.index:2d}] {self.name:<45} | "
            f"{self.max_input_channels:2d} in | "
            f"{self.default_samplerate:5.0f} Hz | "
            f"{self.hostapi_name:<18}{uma_badge}{prod_badge}"
        )


@dataclass
class CameraStreamInfo:
    """Record of a tested network camera stream endpoint."""
    url: str
    is_online: bool
    resolution: Optional[Tuple[int, int]] = None
    latency_ms: float = 0.0
    error_message: Optional[str] = None

    def summary_line(self) -> str:
        """Format a readable status line for console output."""
        if self.is_online:
            res_str = f"{self.resolution[0]}x{self.resolution[1]}" if self.resolution else "unknown"
            return f"ONLINE  ({res_str}, {self.latency_ms:.1f}ms) -> {self.url}"
        reason = f" ({self.error_message})" if self.error_message else ""
        return f"OFFLINE{reason} -> {self.url}"


@dataclass
class HardwareReport:
    """Unified pre-flight hardware status report."""
    audio_devices: List[AudioDeviceInfo]
    selected_audio: Optional[AudioDeviceInfo]
    audio_status_message: str
    camera_stream: CameraStreamInfo
    all_hardware_ready: bool

    def formatted_banner(self) -> str:
        """Return an academic/engineering pre-flight status block."""
        lines = [
            "=" * 78,
            "REAL-TIME MULTIMODAL ACOUSTIC CAMERA -- HARDWARE PRE-FLIGHT REPORT",
            "=" * 78,
            "",
            "--- AUDIO ACQUISITION: miniDSP UMA-16 (Required: 16 ch @ 48000 Hz) ---",
        ]

        if self.selected_audio:
            dev = self.selected_audio
            lines.extend([
                f"  Status        : DETECTED & VALIDATED",
                f"  Device Name   : {dev.name}",
                f"  Device Index  : {dev.index}",
                f"  Input Channels: {dev.max_input_channels}",
                f"  Sample Rate   : {dev.default_samplerate:.0f} Hz",
                f"  Host API      : {dev.hostapi_name}",
                f"  Heuristic     : {'Strong UMA-16 signature' if dev.is_uma16 else '16-channel audio interface'}",
            ])
        else:
            lines.extend([
                f"  Status        : NOT CONNECTED / DEFERRED",
                f"  Reason        : {self.audio_status_message}",
                f"  Safety Policy : Laptop/stereo 2-4ch fallback STRICTLY BLOCKED for MUSIC safety.",
                f"  Detected Input Devices ({len(self.audio_devices)}):",
            ])
            for d in self.audio_devices:
                lines.append(f"    {d.summary_line()}")

        lines.extend([
            "",
            "--- VISION ACQUISITION: ESP32-CAM MJPEG Stream ---",
            f"  Endpoint      : {self.camera_stream.url}",
            f"  Stream Status : {'ONLINE' if self.camera_stream.is_online else 'OFFLINE'}",
        ])
        if self.camera_stream.is_online and self.camera_stream.resolution:
            lines.append(f"  Resolution    : {self.camera_stream.resolution[0]}x{self.camera_stream.resolution[1]}")
            lines.append(f"  Probe Latency : {self.camera_stream.latency_ms:.1f} ms")
        elif not self.camera_stream.is_online:
            lines.append(f"  Failure Cause : {self.camera_stream.error_message or 'No response'}")

        lines.extend([
            "",
            f"System State   : {'READY FOR MULTIMODAL DEMO' if self.all_hardware_ready else 'PARTIAL / SIMULATION RECOMMENDED'}",
            "=" * 78,
        ])
        return "\n".join(lines)


# ============================================================================
# Heuristic & Host API Helpers
# ============================================================================


def is_uma16_name(name: str) -> bool:
    """Return True if the device name contains a miniDSP UMA-16 signature."""
    lower = name.lower()
    return any(p in lower for p in UMA16_NAME_PATTERNS)


def hostapi_rank(hostapi_name: str) -> int:
    """
    Rank host APIs for Windows latency and stability when multiple endpoints
    exist for the same physical device.

    WASAPI > WDM-KS > DirectSound > MME
    """
    n = hostapi_name.lower()
    if "wasapi" in n:
        return 4
    if "wdm-ks" in n or "wdm" in n:
        return 3
    if "directsound" in n:
        return 2
    if "mme" in n:
        return 1
    return 0


# ============================================================================
# Audio Device Discovery & Validation
# ============================================================================


def validate_audio_config(
    device_index: int,
    channels: int = NUM_CHANNELS,
    sample_rate: int = SAMPLE_RATE,
) -> Tuple[bool, str]:
    """
    Validate whether a PortAudio device supports the required multichannel configuration.

    Parameters
    ----------
    device_index : int
        The PortAudio device index.
    channels : int
        Required channels (default 16).
    sample_rate : int
        Required sample rate in Hz (default 48000).

    Returns
    -------
    (bool, str)
        Tuple of (is_supported, validation_message).
    """
    try:
        sd.check_input_settings(
            device=device_index,
            channels=channels,
            samplerate=sample_rate,
        )
        return True, f"Device supports {channels} channels at {sample_rate} Hz."
    except Exception as exc:
        return False, f"Unsupported config ({channels}ch @ {sample_rate}Hz): {exc}"


def list_audio_input_devices() -> List[AudioDeviceInfo]:
    """
    Enumerate all physical and virtual audio input devices on the system.

    Returns
    -------
    List[AudioDeviceInfo]
        All input devices with their capabilities, Host APIs, and UMA-16 heuristics.
    """
    devices = []
    try:
        raw_devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as e:
        print(f"[HardwareDiscovery] Warning: Failed to query sounddevice: {e}")
        return []

    for idx, d in enumerate(raw_devices):
        max_in = d.get("max_input_channels", 0)
        if max_in <= 0:
            continue

        dev_name = d.get("name", f"Device-{idx}")
        h_idx = d.get("hostapi", 0)
        h_name = "Unknown"
        if 0 <= h_idx < len(hostapis):
            h_name = hostapis[h_idx].get("name", f"HostAPI-{h_idx}")

        default_sr = float(d.get("default_samplerate", 44100.0))
        is_uma = is_uma16_name(dev_name)

        # Validate against production requirements (16 channels, 48000 Hz)
        if max_in >= NUM_CHANNELS:
            supported, msg = validate_audio_config(idx, NUM_CHANNELS, SAMPLE_RATE)
        else:
            supported, msg = False, f"Insufficient input channels: {max_in} (requires {NUM_CHANNELS})"

        info = AudioDeviceInfo(
            index=idx,
            name=dev_name,
            max_input_channels=max_in,
            default_samplerate=default_sr,
            hostapi_index=h_idx,
            hostapi_name=h_name,
            is_uma16=is_uma,
            supports_production_config=supported,
            validation_message=msg,
        )
        devices.append(info)

    return devices


def find_uma16_device(
    preference: Optional[int | str] = None,
) -> Tuple[Optional[AudioDeviceInfo], str]:
    """
    Discover and select the miniDSP UMA-16 audio device according to the strict hierarchy:
        1. Explicit user selection (index or name)
        2. Exact/strong UMA-16 name match
        3. Generic 16+ channel candidate (only if unambiguous)

    Safety Guarantees:
    - Never falls back to 2-channel or 4-channel devices.
    - Resolves multiple host API endpoints only for the same candidate.
    - Disallows ambiguous choice if multiple unknown 16-channel devices exist.

    Parameters
    ----------
    preference : Optional[int | str]
        User-specified device index or name substring.

    Returns
    -------
    (Optional[AudioDeviceInfo], str)
        Tuple of (selected_device_or_None, status_or_reason).
    """
    all_inputs = list_audio_input_devices()
    if not all_inputs:
        return None, "No audio input devices detected on the host system."

    # ------------------------------------------------------------------------
    # 1. EXPLICIT USER SELECTION
    # ------------------------------------------------------------------------
    if preference is not None:
        pref_str = str(preference).strip()

        # A: Integer index specified
        if pref_str.isdigit():
            target_idx = int(pref_str)
            match = next((d for d in all_inputs if d.index == target_idx), None)
            if match is None:
                return None, f"User-selected device index [{target_idx}] not found."
            if not match.supports_production_config:
                return (
                    None,
                    f"User-selected device [{target_idx}] '{match.name}' does NOT support "
                    f"{NUM_CHANNELS}-channel / {SAMPLE_RATE} Hz production audio: {match.validation_message}",
                )
            # Explicit selection honored without silent host API replacement
            return match, f"Explicitly selected device index [{match.index}] '{match.name}' validated."

        # B: Name substring specified
        pref_lower = pref_str.lower()
        name_matches = [
            d for d in all_inputs
            if pref_lower in d.name.lower() and d.supports_production_config
        ]
        if not name_matches:
            return (
                None,
                f"No 16ch/48k audio device matching user preference '{pref_str}' was found.",
            )
        # If multiple endpoints share this candidate name, pick highest Host API rank
        name_matches.sort(key=lambda d: hostapi_rank(d.hostapi_name), reverse=True)
        selected = name_matches[0]
        return selected, f"Selected device '{selected.name}' matching preference '{pref_str}' via {selected.hostapi_name}."

    # ------------------------------------------------------------------------
    # 2. STRONG UMA-16 NAME MATCH
    # ------------------------------------------------------------------------
    uma16_candidates = [
        d for d in all_inputs
        if d.is_uma16 and d.supports_production_config
    ]
    if uma16_candidates:
        # Group by device name or pick highest Host API ranking
        uma16_candidates.sort(key=lambda d: hostapi_rank(d.hostapi_name), reverse=True)
        selected = uma16_candidates[0]
        return selected, f"Auto-detected UMA-16: [{selected.index}] '{selected.name}' via {selected.hostapi_name}."

    # ------------------------------------------------------------------------
    # 3. GENERIC 16+ CHANNEL CANDIDATES
    # ------------------------------------------------------------------------
    sixteen_ch_candidates = [
        d for d in all_inputs
        if d.supports_production_config
    ]

    if not sixteen_ch_candidates:
        return (
            None,
            f"No 16-channel / {SAMPLE_RATE} Hz audio device detected. "
            f"Fallback to 2/4-channel laptop mic blocked for MUSIC array safety.",
        )

    # Check distinct physical device names
    distinct_names = {d.name for d in sixteen_ch_candidates}
    if len(distinct_names) == 1:
        # Multiple endpoints of the same physical device across host APIs
        sixteen_ch_candidates.sort(key=lambda d: hostapi_rank(d.hostapi_name), reverse=True)
        selected = sixteen_ch_candidates[0]
        return selected, f"Auto-detected 16-channel device: [{selected.index}] '{selected.name}' via {selected.hostapi_name}."

    # Multiple DIFFERENT 16-channel devices exist without clear UMA-16 heuristic
    cand_list = ", ".join(f"[{d.index}] {d.name} ({d.hostapi_name})" for d in sixteen_ch_candidates)
    return (
        None,
        f"Multiple 16-channel devices found with no definitive UMA-16 match ({cand_list}). "
        f"Please explicitly specify --audio-device <index>.",
    )


# ============================================================================
# Camera Stream Probing & Resolution
# ============================================================================


def test_stream_url(url: str, timeout: float = 1.5) -> CameraStreamInfo:
    """
    Non-blocking network probe for an ESP32-CAM MJPEG stream endpoint.

    Uses a fast socket pre-check to prevent blocking before opening OpenCV.

    Parameters
    ----------
    url : str
        Stream URL to validate.
    timeout : float
        Maximum network timeout in seconds (default: 1.5).

    Returns
    -------
    CameraStreamInfo
        Detailed connectivity and resolution report.
    """
    t_start = time.perf_counter()

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    port = parsed.port or (80 if parsed.scheme == "http" else 443)

    # 1. Fast socket connectivity check to avoid long OpenCV timeouts
    if host:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
        except socket.timeout:
            return CameraStreamInfo(
                url=url,
                is_online=False,
                error_message=f"Connection timed out ({timeout:.1f}s) to {host}:{port}",
            )
        except OSError as e:
            return CameraStreamInfo(
                url=url,
                is_online=False,
                error_message=f"Socket error ({e}) to {host}:{port}",
            )

    # 2. OpenCV stream grab
    cap = None
    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            return CameraStreamInfo(
                url=url,
                is_online=False,
                error_message="VideoCapture could not open stream",
            )

        ret, frame = cap.read()
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        if ret and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            return CameraStreamInfo(
                url=url,
                is_online=True,
                resolution=(w, h),
                latency_ms=elapsed_ms,
            )
        else:
            return CameraStreamInfo(
                url=url,
                is_online=False,
                error_message="Stream opened but failed to decode valid video frame",
                latency_ms=elapsed_ms,
            )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return CameraStreamInfo(
            url=url,
            is_online=False,
            error_message=str(exc),
            latency_ms=elapsed_ms,
        )
    finally:
        if cap is not None:
            cap.release()


def resolve_camera_stream(
    explicit_url: Optional[str] = None,
    candidate_urls: Optional[Sequence[str]] = None,
) -> Tuple[str, CameraStreamInfo]:
    """
    Resolve and validate the active ESP32-CAM stream URL according to priority:
        1. Explicit --stream-url
        2. config.settings.STREAM_URL
        3. User-provided candidate URLs (if any)

    Obsolete IP addresses are never automatically injected.

    Parameters
    ----------
    explicit_url : Optional[str]
        Explicit URL passed by caller or CLI.
    candidate_urls : Optional[Sequence[str]]
        Additional user-specified candidate endpoints.

    Returns
    -------
    (str, CameraStreamInfo)
        Tuple of (resolved_url, test_result).
    """
    urls_to_test: List[str] = []

    if explicit_url:
        urls_to_test.append(explicit_url)
    if STREAM_URL and STREAM_URL not in urls_to_test:
        urls_to_test.append(STREAM_URL)

    if candidate_urls:
        for c in candidate_urls:
            if c not in urls_to_test:
                urls_to_test.append(c)


    if not urls_to_test:
        urls_to_test.append(STREAM_URL)

    last_result: Optional[CameraStreamInfo] = None
    for url in urls_to_test:
        res = test_stream_url(url)
        if res.is_online:
            return url, res
        last_result = res

    # If none succeeded, return the primary candidate with its failure info
    primary = explicit_url or STREAM_URL
    return primary, (last_result or CameraStreamInfo(url=primary, is_online=False, error_message="No stream responsive"))


# ============================================================================
# Unified Pre-Flight Probing
# ============================================================================


def probe_all_hardware(
    audio_preference: Optional[int | str] = None,
    camera_url: Optional[str] = None,
    candidate_urls: Optional[Sequence[str]] = None,
) -> HardwareReport:
    """
    Perform a complete non-destructive probe of audio and camera subsystems.

    Returns
    -------
    HardwareReport
        Complete diagnostic report suitable for display or programmatic consumption.
    """
    all_inputs = list_audio_input_devices()
    selected_audio, audio_msg = find_uma16_device(preference=audio_preference)
    cam_url, cam_info = resolve_camera_stream(
        explicit_url=camera_url,
        candidate_urls=candidate_urls,
    )

    all_ready = (selected_audio is not None) and cam_info.is_online

    return HardwareReport(
        audio_devices=all_inputs,
        selected_audio=selected_audio,
        audio_status_message=audio_msg,
        camera_stream=cam_info,
        all_hardware_ready=all_ready,
    )


# ============================================================================
# CLI Standalone Pre-Flight Diagnostic
# ============================================================================


def main() -> None:
    """Command-line utility for pre-flight hardware inspection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-flight hardware inspection for Real-Time Multimodal Acoustic Camera"
    )
    parser.add_argument(
        "--audio-device",
        type=str,
        default=None,
        help="Device index or name substring for miniDSP UMA-16",
    )
    parser.add_argument(
        "--stream-url",
        type=str,
        default=None,
        help=f"ESP32-CAM MJPEG stream URL (default: {STREAM_URL})",
    )
    parser.add_argument(
        "--list-audio",
        action="store_true",
        help="List all enumerated audio input devices and exit",
    )
    args = parser.parse_args()

    if args.list_audio:
        devices = list_audio_input_devices()
        print(f"\nEnumerated Audio Input Devices ({len(devices)}):")
        print("-" * 88)
        for d in devices:
            print(d.summary_line())
        print("-" * 88)
        return

    print("\nProbing hardware components (ESP32-CAM & miniDSP UMA-16)...")
    report = probe_all_hardware(
        audio_preference=args.audio_device,
        camera_url=args.stream_url,
    )
    print()
    print(report.formatted_banner())


if __name__ == "__main__":
    main()
