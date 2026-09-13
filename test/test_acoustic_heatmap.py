"""
Software test for acoustic heatmap visualization.
"""

from __future__ import annotations

import cv2
import numpy as np

from acoustic.visualization import overlay_heatmap


def main() -> None:

    image = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    heatmap = np.zeros(
        (20, 20),
        dtype=np.float32,
    )

    # Artificial acoustic source near the centre.
    heatmap[10, 10] = 1.0

    # Create a small surrounding response.
    heatmap[9, 10] = 0.8
    heatmap[11, 10] = 0.8
    heatmap[10, 9] = 0.8
    heatmap[10, 11] = 0.8

    result = overlay_heatmap(
        image,
        heatmap,
        alpha=0.6,
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8

    # The result should not remain completely black.
    assert np.any(result != 0)

    cv2.imwrite(
        "test_heatmap_output.jpg",
        result,
    )

    print(
        "ACOUSTIC HEATMAP VISUALIZATION TEST PASSED"
    )


if __name__ == "__main__":
    main()