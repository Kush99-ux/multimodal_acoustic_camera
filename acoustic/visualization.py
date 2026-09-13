"""
acoustic.visualization
======================

Visualization utilities for MUSIC acoustic localization.

This module converts a normalized MUSIC heatmap into a displayable
image without modifying the underlying acoustic localization data.
"""

from __future__ import annotations

import cv2
import numpy as np


def heatmap_to_image(
    heatmap: np.ndarray,
    *,
    output_width: int,
    output_height: int,
    alpha: float = 0.65,
) -> np.ndarray:
    """
    Convert a normalized MUSIC heatmap into a displayable image.

    Parameters
    ----------
    heatmap:
        2-D MUSIC heatmap with values in the range 0..1.

    output_width:
        Width of the resulting image.

    output_height:
        Height of the resulting image.

    alpha:
        Heatmap intensity used for visualization.

    Returns
    -------
    np.ndarray
        BGR image suitable for OpenCV display.
    """

    if heatmap.ndim != 2:
        raise ValueError(
            "Heatmap must be a 2-dimensional array."
        )

    if output_width <= 0 or output_height <= 0:
        raise ValueError(
            "Output dimensions must be positive."
        )

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0,
    )

    heatmap_uint8 = (
        heatmap * 255.0
    ).astype(np.uint8)

    resized = cv2.resize(
        heatmap_uint8,
        (output_width, output_height),
        interpolation=cv2.INTER_CUBIC,
    )

    colored = cv2.applyColorMap(
        resized,
        cv2.COLORMAP_JET,
    )

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    return colored


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Overlay a MUSIC heatmap on top of a camera image.

    Parameters
    ----------
    image:
        Camera image in BGR format.

    heatmap:
        Normalized 2-D MUSIC heatmap.

    alpha:
        Heatmap opacity.

    Returns
    -------
    np.ndarray
        Camera image with acoustic heatmap overlay.
    """

    if image.ndim != 3:
        raise ValueError(
            "Image must be a 3-dimensional BGR image."
        )

    height, width = image.shape[:2]

    colored_heatmap = heatmap_to_image(
        heatmap,
        output_width=width,
        output_height=height,
        alpha=alpha,
    )

    return cv2.addWeighted(
        image,
        1.0 - alpha,
        colored_heatmap,
        alpha,
        0.0,
    )