import cv2
import numpy as np

from acoustic.acquisition import AudioAcquisition
from acoustic.music import MusicLocalizer
from visualization.heatmap import AcousticHeatmapVisualizer


def main():

    localizer = MusicLocalizer()

    visualizer = AcousticHeatmapVisualizer

    audio = AudioAcquisition()

    audio.start()

    try:

        while True:

            frame = audio.read()

            heatmap = localizer.localize(frame)

            display = visualizer.render(heatmap)

            cv2.imshow("MUSIC Localization", display)

            max_index = np.unravel_index(np.argmax(heatmap), heatmap.shape)

            x = (
                localizer.factory.grid.x_min
                + max_index[1] * localizer.factory.grid.increment
            )

            y = (
                localizer.factory.grid.y_min
                + max_index[0] * localizer.factory.grid.increment
            )

            print(
                f"Estimated source: x={x:.3f} m, y={y:.3f} m",
                end="\\r",
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        audio.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()