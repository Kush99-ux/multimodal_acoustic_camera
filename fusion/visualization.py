"""
fusion.visualization
====================

Angular acoustic-to-camera visualization.

Projection:
    MUSIC direction (azimuth, elevation)
             ↓
        camera FOV (centered at az=90, el=0)
             ↓
          pixels

This is angular projection only. It is NOT physical camera-acoustic
calibration.

IMPORTANT — mirroring
----------------------
The angle->pixel formula below is copied from the original
uma_live_camera.py prototype, which explicitly mirrored the camera
image (`cv2.flip(frame, 1)`) before drawing on it. The formula was
tuned against that mirrored frame.

The ESP32-CAM feed is NOT guaranteed to have the same physical
orientation as the original webcam setup, so `mirror_camera` is
exposed as a constructor flag rather than hard-coded. Use the
Phase 10 hardware test to determine the correct value for your rig.
"""

from __future__ import annotations

import cv2
import numpy as np

from fusion.models import MultimodalResult


class AcousticHeatmapVisualizer:
    def __init__(
        self,
        heatmap_radius: int = 55,
        alpha: float = 0.45,
        horizontal_fov_deg: float = 50.0,
        vertical_fov_deg: float = 37.5,
        marker_size: int = 40,
        mirror_camera: bool = True,   # matches original cv2.flip(frame, 1)
        diagnostic: bool = False,
    ) -> None:

        if heatmap_radius <= 0:
            raise ValueError("heatmap_radius must be positive.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        if horizontal_fov_deg <= 0 or vertical_fov_deg <= 0:
            raise ValueError("Camera FOV must be positive.")

        self.heatmap_radius = int(heatmap_radius)
        self.alpha = float(alpha)
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.vertical_fov_deg = float(vertical_fov_deg)
        self.marker_size = int(marker_size)
        self.mirror_camera = bool(mirror_camera)
        self.diagnostic = bool(diagnostic)

        self.azimuth_center = 90.0
        self.elevation_center = 0.0

    # --------------------------------------------------------
    # Angles -> camera pixels
    # --------------------------------------------------------

    def angle_to_image(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        width: int,
        height: int,
    ) -> tuple[int, int]:

        az_min = self.azimuth_center - (self.horizontal_fov_deg / 2.0)
        az_max = self.azimuth_center + (self.horizontal_fov_deg / 2.0)

        el_min = self.elevation_center - (self.vertical_fov_deg / 2.0)
        el_max = self.elevation_center + (self.vertical_fov_deg / 2.0)

        x_norm = (azimuth_deg - az_min) / (az_max - az_min)
        y_norm = (elevation_deg - el_min) / (el_max - el_min)

        u = int(x_norm * width)
        v = int((1.0 - y_norm) * height)

        return (
            int(np.clip(u, 0, width - 1)),
            int(np.clip(v, 0, height - 1)),
        )

    # --------------------------------------------------------
    # Smooth Gaussian heatmap
    # --------------------------------------------------------

    def create_heatmap(
        self,
        shape: tuple[int, int],
        center: tuple[int, int],
        intensity: float,
    ) -> np.ndarray:

        height, width = shape
        u, v = center

        scale = min(width, height) / 480.0
        radius = max(45.0, self.heatmap_radius * scale)
        sigma = radius / 2.0

        yy, xx = np.mgrid[0:height, 0:width]

        heatmap = np.exp(
            -((xx - u) ** 2 + (yy - v) ** 2) / (2.0 * sigma * sigma)
        ).astype(np.float32)

        return heatmap * float(np.clip(intensity, 0.0, 1.0))

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    def render(
        self,
        image: np.ndarray,
        result: MultimodalResult,
    ) -> np.ndarray:

        if image is None:
            raise ValueError("image cannot be None.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be BGR with shape (H, W, 3).")

        # ------------------------------------------------
        # Restore the original mirroring step.
        #
        # This MUST match the physical camera orientation.
        # Toggle `mirror_camera` and re-run Phase 10 if the
        # marker lands on the wrong side.
        # ------------------------------------------------
        output = cv2.flip(image, 1) if self.mirror_camera else image.copy()

        localization = result.acoustic.localization

        if localization is None or not localization.valid:
            return output

        height, width = output.shape[:2]

        azimuth = float(result.acoustic.metadata.get("azimuth_deg", 90.0))
        elevation = float(result.acoustic.metadata.get("elevation_deg", 0.0))

        center = self.angle_to_image(azimuth, elevation, width, height)

        confidence = float(np.clip(localization.confidence, 0.0, 1.0))

        if self.diagnostic:
            print("MUSIC PEAK")
            print("-----------")
            print(f"azimuth   : {azimuth:+.1f} deg")
            print(f"elevation : {elevation:+.1f} deg")
            print(f"confidence: {confidence:.3f}")
            print("CAMERA PROJECTION")
            print("-----------------")
            print(f"pixel x   : {center[0]}")
            print(f"pixel y   : {center[1]}")
            print(f"mirrored  : {self.mirror_camera}")

        if confidence <= 0.0:
            return output

        heatmap = self.create_heatmap((height, width), center, confidence)

        heatmap_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

        mask = heatmap[:, :, None] * self.alpha

        output = (
            output.astype(np.float32) * (1.0 - mask)
            + colored.astype(np.float32) * mask
        ).clip(0, 255).astype(np.uint8)

        cv2.drawMarker(
            output,
            center,
            (0, 0, 255),
            cv2.MARKER_CROSS,
            self.marker_size,
            3,
            cv2.LINE_AA,
        )

        return output