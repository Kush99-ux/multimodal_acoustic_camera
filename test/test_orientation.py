from acoustic.geometry import UMA16_GEOMETRY
from acoustic.orientation import ArrayOrientation


if __name__ == "__main__":

    coords = UMA16_GEOMETRY

    print("Original first microphone:", coords[0])

    for angle in [0, 90, 180, 270]:
        transformed = ArrayOrientation.transform(coords, rotation=angle)
        print(f"Rotation {angle:3d}: first microphone -> {transformed[0]}")