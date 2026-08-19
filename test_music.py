import time

from acoustic.acquisition import AudioAcquisition
from acoustic.music import MusicLocalizer

audio = AudioAcquisition()
localizer = MusicLocalizer()

audio.start()

print("Running MUSIC localization for 5 iterations...\n")

try:
    for i in range(5):
        frame = audio.read()

        heatmap = localizer.localize(frame)

        print(
            f"Iteration {i + 1}: "
            f"heatmap shape = {heatmap.shape}, "
            f"max = {heatmap.max():.3f}, "
            f"min = {heatmap.min():.3f}"
        )

        time.sleep(0.2)

finally:
    audio.stop()

print("Done.")