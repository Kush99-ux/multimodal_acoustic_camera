import cv2

from vision.pipeline import VisionPipeline

vision = VisionPipeline()

if not vision.connect():
    raise RuntimeError("Failed to connect to ESP32 camera")

while True:
    frame, annotated, detections = vision.process_frame()

    if annotated is None:
        continue

    cv2.imshow("Vision Pipeline", annotated)

    # Print detected object names
    if detections:
        names = [d["class_name"] for d in detections]
        print(names)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vision.release()
cv2.destroyAllWindows()