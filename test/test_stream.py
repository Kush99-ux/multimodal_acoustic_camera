import cv2

from vision.stream import ESP32Camera

camera = ESP32Camera()

if not camera.connect():
    raise RuntimeError("Failed to connect to ESP32 camera")

while True:
    frame = camera.read()

    if frame is None:
        continue

    cv2.imshow("ESP32 Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()