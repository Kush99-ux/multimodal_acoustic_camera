"""
demo.core.runner
================
Robust multimodal pipeline driver with hardware connection management,
graceful error handling, and simulation fallback.
"""

from __future__ import annotations
import os
import time
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from config.settings import (
    STREAM_URL,
    CAMERA_FOV_H,
    CAMERA_FOV_V,
    SAMPLE_RATE,
    FRAME_SIZE,
    SYNC_TOLERANCE_MS,
    NOISE_THRESHOLD_DB,
)
from acoustic.audio import AudioAcquisition, AudioFrame
from acoustic.music import MusicLocalizer
from acoustic.models import AcousticFrame
from vision.camera import CameraAcquisition, CameraFrame
from vision.segmentation import YOLOSegmenter
from vision.models import VisionFrame
from fusion.synchronization import TimestampSynchronizer
from fusion.visualization import AcousticHeatmapVisualizer
from demo.core.diagnostics import StreamDiagnostics
from demo.core.logger import DemoLogger
from hardware.devices import AudioDeviceInfo, find_uma16_device, resolve_camera_stream


class DemoPipelineRunner:

    """
    Orchestrates live acquisition from ESP32-CAM and UMA-16, executes
    MUSIC and YOLO, and produces unified telemetry for the UI panels.
    """

    def __init__(
        self,
        stream_url: str = "http://10.243.98.165:81/stream",
        simulate: bool = False,
        log_dir: str = "scratch/demo_results",
        audio_device: Optional[int | str] = None,
    ) -> None:
        self.stream_url = stream_url
        self.simulate = simulate
        self.log_dir = log_dir
        self.audio_device = audio_device
        self.selected_audio_info: Optional[AudioDeviceInfo] = None

        # Subsystems

        self.camera: Optional[CameraAcquisition] = None
        self.audio: Optional[AudioAcquisition] = None
        self.localizer: Optional[MusicLocalizer] = None
        self.segmenter: Optional[YOLOSegmenter] = None
        self.synchronizer = TimestampSynchronizer(tolerance_ms=SYNC_TOLERANCE_MS)
        self.visualizer = AcousticHeatmapVisualizer(alpha=0.45)
        self.diagnostics = StreamDiagnostics()
        self.logger = DemoLogger(output_dir=self.log_dir)

        # Connection & operational states
        self.camera_connected = False
        self.audio_connected = False
        self.music_running = False
        self.yolo_running = False
        self.yolo_enabled = True
        self.is_synchronized = False

        # Simulation assets if simulate=True
        self._sim_audio_frames: Optional[np.ndarray] = None
        self._sim_image: Optional[np.ndarray] = None
        self._sim_frame_idx = 0

        # Ground truth click target
        self.gt_click: Optional[Tuple[int, int]] = None
        self.gt_mode_enabled = False

        # Most recent telemetry snapshot
        self.latest_telemetry: Dict[str, Any] = self._init_empty_telemetry()
        self.latest_camera_frame: Optional[np.ndarray] = None
        self.latest_overlaid_frame: Optional[np.ndarray] = None
        self.latest_heatmap: Optional[np.ndarray] = None
        self.latest_raw_audio: Optional[np.ndarray] = None

        self._init_subsystems()

    def _init_empty_telemetry(self) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "camera_connected": False,
            "camera_frame_id": 0,
            "camera_timestamp": 0.0,
            "audio_connected": False,
            "audio_frame_id": 0,
            "audio_timestamp": 0.0,
            "music_running": False,
            "yolo_running": False,
            "yolo_enabled": True,
            "synchronized": False,
            "sync_delta_ms": 0.0,
            "azimuth_deg": 90.0,
            "elevation_deg": -6.0,
            "peak_db": 0.0,
            "peak_valid": False,
            "overlay_u": 320,
            "overlay_v": 240,
            "yolo_object": "NONE",
            "yolo_confidence": 0.0,
            "yolo_box": None,
            "yolo_center": None,
            "yolo_latency_ms": 0.0,
            "gt_click": None,
            "pixel_error_gt": None,
            "mic_rms": np.zeros(16, dtype=np.float32),
            "band_energies": [(f, 0.0, -60.0) for f in [500, 1000, 1500, 2000, 2500, 3000]],
            "sync_stats": self.diagnostics.get_sync_stats(),
        }

    def _init_subsystems(self) -> None:
        """Initialize models and hardware connections with graceful fallbacks."""
        print("[DemoRunner] Loading YOLO11n-seg model...")
        try:
            self.segmenter = YOLOSegmenter()
            self.yolo_running = True
            print("[DemoRunner] YOLO11n-seg initialized.")
        except Exception as e:
            print(f"[DemoRunner] Warning: YOLO initialization failed: {e}")
            self.yolo_running = False

        print("[DemoRunner] Initializing 6-band MUSIC localizer...")
        try:
            self.localizer = MusicLocalizer()
            self.music_running = True
            print("[DemoRunner] MUSIC localizer initialized.")
        except Exception as e:
            print(f"[DemoRunner] Warning: MUSIC localizer initialization failed: {e}")
            self.music_running = False

        if self.simulate:
            self._init_simulation_assets()
        else:
            self._connect_hardware()

    def _init_simulation_assets(self) -> None:
        """Load genuine recorded experimental files for hardware-independent demonstration."""
        print("[DemoRunner] Operating in SIMULATION / REPLAY mode.")
        npy_path = "scratch/moderate_left_audio_frames.npy"
        img_path = "scratch/left_representative.jpg"

        if os.path.exists(npy_path):
            self._sim_audio_frames = np.load(npy_path)
            print(f"[DemoRunner] Loaded {len(self._sim_audio_frames)} simulation audio frames from {npy_path}.")
            self.audio_connected = True
        else:
            print(f"[DemoRunner] Warning: Simulation audio file {npy_path} not found.")

        if os.path.exists(img_path):
            self._sim_image = cv2.imread(img_path)
            print(f"[DemoRunner] Loaded simulation image from {img_path}.")
            self.camera_connected = True
        else:
            self._sim_image = np.zeros((480, 640, 3), dtype=np.uint8)

    def _connect_hardware(self) -> None:
        """Attempt connections to ESP32-CAM and miniDSP UMA-16."""
        # 1. Resolve and validate ESP32-CAM stream
        print(f"[DemoRunner] Resolving ESP32-CAM stream URL: {self.stream_url}...")
        resolved_url, stream_info = resolve_camera_stream(explicit_url=self.stream_url)
        self.stream_url = resolved_url

        if stream_info.is_online:
            res_str = f"{stream_info.resolution[0]}x{stream_info.resolution[1]}" if stream_info.resolution else "unknown"
            print(f"[DemoRunner] ESP32-CAM stream verified online ({res_str}, {stream_info.latency_ms:.1f}ms).")
            try:
                self.camera = CameraAcquisition(stream_url=self.stream_url)
                self.camera.start()
                self.camera_connected = True
                print("[DemoRunner] Camera reader thread started.")
            except Exception as e:
                print(f"[DemoRunner] ESP32-CAM connection error: {e}")
                self.camera_connected = False
        else:
            print(f"[DemoRunner] ESP32-CAM offline: {stream_info.error_message}. Deferring connection.")
            self.camera_connected = False

        # 2. Discover and validate miniDSP UMA-16 audio device
        print(f"[DemoRunner] Discovering 16-channel UMA-16 audio device (preference: {self.audio_device})...")
        dev, status_msg = find_uma16_device(preference=self.audio_device)
        self.selected_audio_info = dev

        if dev is not None:
            print(f"[DemoRunner] {status_msg}")
            print(
                f"[DemoRunner] Connecting AudioAcquisition on device [{dev.index}] "
                f"'{dev.name}' ({dev.max_input_channels} ch @ 48000 Hz, {dev.hostapi_name})..."
            )
            try:
                self.audio = AudioAcquisition(device=dev.index)
                self.audio.start()
                self.audio_connected = True
                print("[DemoRunner] UMA-16 audio stream started.")
            except Exception as e:
                print(f"[DemoRunner] UMA-16 audio stream deferred: {e}")
                self.audio_connected = False
        else:
            print(f"[DemoRunner] UMA-16 connection deferred: {status_msg}")
            self.audio_connected = False


    def step(self) -> Dict[str, Any]:
        """
        Execute one acquisition, processing, and fusion iteration.
        Returns the complete updated live telemetry dictionary.
        """
        now = time.time()
        telemetry = self._init_empty_telemetry()
        telemetry["timestamp"] = now

        # ----------------------------------------------------
        # 1. ACQUIRE AUDIO
        # ----------------------------------------------------
        audio_frame: Optional[AudioFrame] = None
        raw_audio_data: Optional[np.ndarray] = None

        if self.simulate and self._sim_audio_frames is not None:
            raw_audio_data = self._sim_audio_frames[self._sim_frame_idx % len(self._sim_audio_frames)]
            self._sim_frame_idx += 1
            audio_frame = AudioFrame(data=raw_audio_data, timestamp=now, frame_id=self._sim_frame_idx)
            self.audio_connected = True
        elif self.audio is not None and self.audio_connected:
            try:
                audio_frame = self.audio.read(timeout=0.04)
                if audio_frame is not None:
                    raw_audio_data = audio_frame.data
            except Exception:
                self.audio_connected = False

        self.latest_raw_audio = raw_audio_data

        # ----------------------------------------------------
        # 2. LOCALIZATION (MUSIC)
        # ----------------------------------------------------
        acoustic_frame: Optional[AcousticFrame] = None
        if audio_frame is not None and self.localizer is not None:
            try:
                acoustic_frame = self.localizer.localize_audio_frame(audio_frame)
                meta = acoustic_frame.metadata
                self.latest_heatmap = meta.get("heatmap")
                telemetry["azimuth_deg"] = float(meta.get("azimuth_deg", 90.0))
                telemetry["elevation_deg"] = float(meta.get("elevation_deg", -6.0))
                telemetry["peak_db"] = float(meta.get("db_level", 0.0))
                telemetry["peak_valid"] = bool(meta.get("valid", False))
                telemetry["audio_frame_id"] = int(audio_frame.frame_id)
                telemetry["audio_timestamp"] = float(audio_frame.timestamp)
            except Exception as e:
                pass

        # Calculate live audio diagnostics
        if raw_audio_data is not None:
            telemetry["mic_rms"] = self.diagnostics.compute_mic_rms(raw_audio_data)
            telemetry["band_energies"] = self.diagnostics.compute_band_energies(raw_audio_data)

        # ----------------------------------------------------
        # 3. ACQUIRE CAMERA
        # ----------------------------------------------------
        cam_frame_obj: Optional[CameraFrame] = None
        raw_cam_img: Optional[np.ndarray] = None

        if self.simulate and self._sim_image is not None:
            raw_cam_img = self._sim_image.copy()
            cam_frame_obj = CameraFrame(frame=raw_cam_img, timestamp=now, frame_id=self._sim_frame_idx)
            self.camera_connected = True
        elif self.camera is not None and self.camera_connected:
            try:
                cam_frame_obj = self.camera.read()
                if cam_frame_obj is not None:
                    raw_cam_img = cam_frame_obj.frame
            except Exception:
                self.camera_connected = False

        if raw_cam_img is None:
            # Fallback placeholder frame
            raw_cam_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                raw_cam_img,
                "[!] CAMERA DISCONNECTED / RETRYING",
                (80, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (60, 60, 220),
                2,
                cv2.LINE_AA,
            )

        self.latest_camera_frame = raw_cam_img
        h_img, w_img = raw_cam_img.shape[:2]

        if cam_frame_obj is not None:
            telemetry["camera_frame_id"] = int(cam_frame_obj.frame_id)
            telemetry["camera_timestamp"] = float(cam_frame_obj.timestamp)

        # ----------------------------------------------------
        # 4. VISION (YOLO11n-seg)
        # ----------------------------------------------------
        detections = []
        if self.yolo_enabled and self.segmenter is not None and raw_cam_img is not None and self.camera_connected:
            t_start = time.perf_counter()
            detections = self.segmenter.detect(raw_cam_img)
            telemetry["yolo_latency_ms"] = float((time.perf_counter() - t_start) * 1000.0)

            if detections:
                best = max(detections, key=lambda d: d.confidence)
                telemetry["yolo_object"] = str(best.class_name)
                telemetry["yolo_confidence"] = float(best.confidence)
                telemetry["yolo_box"] = best.bbox
                telemetry["yolo_center"] = best.bbox_centroid
            else:
                telemetry["yolo_object"] = "NONE"
                telemetry["yolo_confidence"] = 0.0
        else:
            telemetry["yolo_latency_ms"] = 0.0
            if not self.yolo_enabled:
                telemetry["yolo_object"] = "DISABLED"
                telemetry["yolo_confidence"] = 0.0
                telemetry["yolo_box"] = None
                telemetry["yolo_center"] = None

        # ----------------------------------------------------
        # 5. SYNCHRONIZATION
        # ----------------------------------------------------
        if cam_frame_obj is not None and audio_frame is not None:
            delta_ms = abs(cam_frame_obj.timestamp - audio_frame.timestamp) * 1000.0
            telemetry["sync_delta_ms"] = float(delta_ms)
            telemetry["synchronized"] = bool(delta_ms <= SYNC_TOLERANCE_MS)
            self.diagnostics.record_sync_delta(delta_ms)
        telemetry["sync_stats"] = self.diagnostics.get_sync_stats()

        # ----------------------------------------------------
        # 6. ACOUSTIC PROJECTION & VISUAL OVERLAY
        # ----------------------------------------------------
        proj_u, proj_v = self.visualizer.angle_to_image(
            telemetry["azimuth_deg"],
            telemetry["elevation_deg"],
            w_img,
            h_img,
        )
        telemetry["overlay_u"] = int(proj_u)
        telemetry["overlay_v"] = int(proj_v)

        overlaid = raw_cam_img.copy()

        # Render YOLO masks & boxes
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(overlaid, (x1, y1), (x2, y2), (215, 85, 185), 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                overlaid,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.52,
                (215, 85, 185),
                1,
                cv2.LINE_AA,
            )

        # Render acoustic heatmap if valid
        if self.latest_heatmap is not None and telemetry["peak_valid"]:
            try:
                # Resize and blend heatmap
                h_norm = cv2.resize(self.latest_heatmap, (w_img, h_img))
                h_norm = np.fliplr(np.flipud(h_norm))  # Coordinate alignment
                h_u8 = (np.clip(h_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
                colored = cv2.applyColorMap(h_u8, cv2.COLORMAP_INFERNO)
                cv2.addWeighted(colored, 0.40, overlaid, 0.60, 0, overlaid)
            except Exception:
                pass

        # Draw optical center crosshair (reference)
        cx, cy = w_img // 2, h_img // 2
        cv2.line(overlaid, (cx - 10, cy), (cx + 10, cy), (140, 140, 140), 1, cv2.LINE_AA)
        cv2.line(overlaid, (cx, cy - 10), (cx, cy + 10), (140, 140, 140), 1, cv2.LINE_AA)

        # Draw acoustic peak marker
        if telemetry["peak_valid"]:
            cv2.drawMarker(
                overlaid,
                (proj_u, proj_v),
                (0, 240, 255),  # Yellow/amber peak
                cv2.MARKER_CROSS,
                32,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(overlaid, (proj_u, proj_v), 16, (0, 240, 255), 1, cv2.LINE_AA)
            cv2.putText(
                overlaid,
                f"ACOUSTIC ({proj_u}, {proj_v})",
                (proj_u + 14, proj_v - 12),
                cv2.FONT_HERSHEY_DUPLEX,
                0.48,
                (0, 240, 255),
                1,
                cv2.LINE_AA,
            )

        # Ground truth click and Euclidean error calculation
        if self.gt_click is not None:
            gx, gy = self.gt_click
            telemetry["gt_click"] = (gx, gy)
            # Distance error
            dist_px = float(np.sqrt((proj_u - gx) ** 2 + (proj_v - gy) ** 2))
            telemetry["pixel_error_gt"] = dist_px

            # Draw GT click marker
            cv2.circle(overlaid, (gx, gy), 7, (40, 130, 240), -1, cv2.LINE_AA)
            cv2.putText(
                overlaid,
                f"GT ({gx}, {gy})",
                (gx + 12, gy + 16),
                cv2.FONT_HERSHEY_DUPLEX,
                0.48,
                (40, 130, 240),
                1,
                cv2.LINE_AA,
            )

            # Draw connector line showing error
            if telemetry["peak_valid"]:
                cv2.line(overlaid, (proj_u, proj_v), (gx, gy), (40, 130, 240), 2, cv2.LINE_AA)

        self.latest_overlaid_frame = overlaid

        # Update telemetry state flags
        telemetry["camera_connected"] = self.camera_connected
        telemetry["audio_connected"] = self.audio_connected
        telemetry["music_running"] = self.music_running
        telemetry["yolo_running"] = self.yolo_running and self.yolo_enabled
        telemetry["yolo_enabled"] = self.yolo_enabled
        self.latest_telemetry = telemetry

        # Record frame if active session
        self.logger.record_frame(telemetry)

        return telemetry

    def toggle_yolo(self) -> bool:
        """Toggle YOLO inference ON/OFF without stopping camera, audio, or MUSIC."""
        self.yolo_enabled = not self.yolo_enabled
        return self.yolo_enabled

    def set_ground_truth(self, u: int, v: int) -> None:
        """Set ground truth target pixel coordinates."""
        self.gt_click = (int(u), int(v))

    def clear_ground_truth(self) -> None:
        """Clear ground truth target pixel coordinates."""
        self.gt_click = None

    def save_snapshot(self, custom_image: Optional[np.ndarray] = None) -> str:
        """Save snapshot of current telemetry and canvas/camera frame."""
        img = custom_image if custom_image is not None else self.latest_overlaid_frame
        return self.logger.save_snapshot(img, self.latest_telemetry)

    def toggle_session_recording(self) -> Tuple[bool, Optional[str]]:
        """Toggle continuous session telemetry recording on/off."""
        if self.logger.recording_session:
            saved_path = self.logger.stop_session()
            return (False, saved_path)
        else:
            file_path = self.logger.start_session()
            return (True, file_path)

    def close(self) -> None:
        """Safely release camera and audio resources."""
        if self.camera is not None and self.camera_connected:
            try:
                self.camera.stop()
            except Exception:
                pass
        if self.audio is not None and self.audio_connected:
            try:
                self.audio.stop()
            except Exception:
                pass
        print("[DemoRunner] Resources safely stopped.")

