"""
visualization.heatmap
=====================

Real-time acoustic heatmap visualization for MUSIC localization.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from __future__ import annotations

import cv2
import numpy as np


class AcousticHeatmapVisualizer:
    """
    OpenCV-based real-time heatmap visualizer.
    """

    def __init__(
        self,
        window_name: str = "Acoustic Heatmap",
        display_size: int = 700,
    ) -> None:
        self.window_name = window_name
        self.display_size = display_size

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.window_name,
            self.display_size,
            self.display_size,
        )

    def render(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Convert a normalized heatmap into a colored visualization.
        """

        img = np.clip(heatmap * 255, 0, 255).astype(np.uint8)

        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        img = cv2.resize(
            img,
            (self.display_size, self.display_size),
            interpolation=cv2.INTER_CUBIC,
        )

        return img

    def show(self, heatmap: np.ndarray) -> bool:
        """
        Display the heatmap.

        Returns
        -------
        bool
            False when the user presses Q.
        """

        image = self.render(heatmap)

        cv2.imshow(self.window_name, image)

        key = cv2.waitKey(1) & 0xFF

        return key != ord("q")

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)