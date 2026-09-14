import cv2
from ultralytics import YOLO

# Load YOLO11 segmentation model
model = YOLO("yolo11n-seg.pt")

# ESP32 stream URL
url = "http://10.158.108.165:81/stream"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Cannot open stream")
    exit()

cv2.namedWindow("ESP32 YOLO11 Segmentation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ESP32 YOLO11 Segmentation", 1000, 700)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO11 segmentation
    results = model(frame)

    # Draw masks + boxes + labels
    annotated_frame = results[0].plot()

    cv2.imshow("ESP32 YOLO11 Segmentation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()