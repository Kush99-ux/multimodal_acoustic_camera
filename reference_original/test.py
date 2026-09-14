import sounddevice as sd
import cv2
print("--- AUDIO DEVICES ---")
print(sd.query_devices())

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is AVAILABLE")
        cap.release()