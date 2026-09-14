import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# ESP32 stream URL
url = "http://10.158.108.165:81/stream"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Cannot open stream")
    exit()

# Create resizable window
cv2.namedWindow("ESP32 YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ESP32 YOLO Detection", 1000, 700)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("ESP32 YOLO Detection", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()