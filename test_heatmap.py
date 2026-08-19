import time

from acoustic.acquisition import AudioAcquisition
from acoustic.music import MusicLocalizer
from visualization.heatmap import AcousticHeatmapVisualizer

audio = AudioAcquisition()
localizer = MusicLocalizer()
visualizer = AcousticHeatmapVisualizer()

audio.start()

print("Real-time acoustic heatmap running.")
print("Press Q to exit.")

try:
    while True:
        frame = audio.read()

        heatmap = localizer.localize(frame)

        if not visualizer.show(heatmap):
            break

        time.sleep(0.02)

finally:
    audio.stop()
    visualizer.close()

print("Done.")