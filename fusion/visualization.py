"""
fusion.visualization
====================

Acoustic-to-camera visualization.

Projection:
    MUSIC direction (azimuth, elevation)
             ↓
        camera FOV (centered at az=90, el=0)
             ↓
           pixels

This renders the REAL 2D MUSIC pseudo-spectrum heatmap overlaid on the camera
feed along with the acoustic peak crosshair marker.

Camera Orientation:
The ESP32-CAM is outward-facing. By default, no horizontal mirror flip is applied
(`mirror_camera=False`), maintaining 1:1 physical consistency between visual
and acoustic azimuth:
    - Physical Left:  azimuth < 90° -> image Left (u < width/2)
    - Physical Right: azimuth > 90° -> image Right (u > width/2)
    - Physical Above: elevation > 0° -> image Top (v < height/2)
    - Physical Below: elevation < 0° -> image Bottom (v > height/2)

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import cv2
import numpy as np

from config.settings import (
    CAMERA_FOV_H,
    CAMERA_FOV_V,
    MIRROR_CAMERA,
    HEATMAP_ALPHA,
)
from fusion.models import MultimodalResult


class AcousticHeatmapVisualizer:
    """
    Overlays the true 2D MUSIC acoustic heatmap and peak marker onto camera frames.
    """

    def __init__(
        self,
        alpha: float = HEATMAP_ALPHA,
        horizontal_fov_deg: float = CAMERA_FOV_H,
        vertical_fov_deg: float = CAMERA_FOV_V,
        marker_size: int = 40,
        mirror_camera: bool = MIRROR_CAMERA,
        colormap: int = cv2.COLORMAP_INFERNO,
        diagnostic: bool = False,
        azimuth_center: float = 90.0,
        elevation_center: float = -6.0,
    ) -> None:

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        if horizontal_fov_deg <= 0 or vertical_fov_deg <= 0:
            raise ValueError("Camera FOV must be positive.")

        self.alpha = float(alpha)
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.vertical_fov_deg = float(vertical_fov_deg)
        self.marker_size = int(marker_size)
        self.mirror_camera = bool(mirror_camera)
        self.colormap = int(colormap)
        self.diagnostic = bool(diagnostic)

        self.azimuth_center = float(azimuth_center)
        self.elevation_center = float(elevation_center)

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
        """
        Convert angular direction (azimuth, elevation in degrees) to pixel coordinates.

        Physical alignment (outward-facing camera, no mirror):
          - Azimuth > 90° (-X) -> Camera Left (x_norm < 0.5 -> u < width/2)
          - Azimuth < 90° (+X) -> Camera Right (x_norm > 0.5 -> u > width/2)
          - Elevation > 0° -> Up (y_norm > 0.5 -> v < height/2)
          - Elevation < 0° -> Down (y_norm < 0.5 -> v > height/2)
        """
        az_min = self.azimuth_center - (self.horizontal_fov_deg / 2.0)
        az_max = self.azimuth_center + (self.horizontal_fov_deg / 2.0)

        el_min = self.elevation_center - (self.vertical_fov_deg / 2.0)
        el_max = self.elevation_center + (self.vertical_fov_deg / 2.0)

        x_norm = (az_max - azimuth_deg) / (az_max - az_min)
        y_norm = (elevation_deg - el_min) / (el_max - el_min)

        u = int(x_norm * width)
        v = int((1.0 - y_norm) * height)

        return (
            int(np.clip(u, 0, width - 1)),
            int(np.clip(v, 0, height - 1)),
        )

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    def render(
        self,
        image: np.ndarray,
        result: MultimodalResult,
    ) -> np.ndarray:
        """
        Render the real 2-D MUSIC acoustic heatmap and peak crosshair onto the camera image.
        """
        if image is None:
            raise ValueError("image cannot be None.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be BGR with shape (H, W, 3).")

        # ESP32-CAM is outward-facing; keep unflipped unless explicitly requested.
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

        # --- Overlay Real 2D MUSIC Heatmap ---
        raw_heatmap = result.acoustic_heatmap
        if raw_heatmap is None and "heatmap" in result.acoustic.metadata:
            raw_heatmap = result.acoustic.metadata["heatmap"]

        if raw_heatmap is not None and np.any(raw_heatmap > 0):
            # In the Acoular scanning grid:
            #   row 0 is el_min (-18.75 deg, bottom)
            #   row -1 is el_max (+18.75 deg, top)
            # In OpenCV image space:
            #   y = 0 is top, y = H-1 is bottom
            #   x = 0 is left, x = W-1 is right
            # Vertically flip (higher elevation to top) and horizontally flip (azimuth > 90° to left):
            heatmap_oriented = np.fliplr(np.flipud(raw_heatmap))

            heatmap_u8 = (np.clip(heatmap_oriented, 0.0, 1.0) * 255.0).astype(np.uint8)
            heatmap_resized = cv2.resize(
                heatmap_u8,
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )
            colored_heatmap = cv2.applyColorMap(heatmap_resized, self.colormap)

            # Blend with camera image
            output = cv2.addWeighted(
                output,
                1.0 - self.alpha,
                colored_heatmap,
                self.alpha,
                0.0,
            )

        # --- Draw Peak Direction Crosshair ---
        if confidence > 0.0:
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
