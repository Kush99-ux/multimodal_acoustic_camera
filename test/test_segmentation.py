import cv2

from vision.stream import ESP32Camera
from vision.segmentation import YOLOSegmenter

camera = ESP32Camera()
segmenter = YOLOSegmenter()

if not camera.connect():
    raise RuntimeError("Failed to connect to ESP32 camera")

while True:
    frame = camera.read()

    if frame is None:
        continue

    results = segmenter.segment(frame)
    detections = segmenter.get_detections(results)
    print(detections)
    annotated = segmenter.annotate(frame, results)

    cv2.imshow("YOLO11 Segmentation", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()