from acoustic.calibration import ChannelCalibrator


if __name__ == "__main__":

    calibrator = ChannelCalibrator()

    mapping = calibrator.run()

    print("\\nFinal mapping:")

    for mic, ch in mapping.items():
        print(f"{mic} -> CH{ch}")