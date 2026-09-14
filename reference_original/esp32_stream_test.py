import cv2

# Replace with your ESP32 IP
url = "http://10.158.108.165:81/stream"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Cannot open stream")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("ESP32 Stream", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()