"""
============================================================
Configuration Module
Project: Real-Time Multimodal Acoustic Camera
Repository: multimodal_acoustic_camera

This file contains all centralized configuration parameters used by
the project. No hardware-specific values should be hardcoded in other
modules; all modules should import settings from this file.

Author: Kush Sahu
============================================================
"""

# ============================================================
# Camera Configuration (ESP32-CAM AI Thinker + OV2640)
# ============================================================

# Current ESP32-CAM IP address on the local Wi-Fi network.
# Update this if the ESP32 receives a new IP from the router.
ESP32_IP = "172.27.8.165"

# MJPEG stream endpoint exposed by the ESP32 camera server.
STREAM_URL = f"http://{ESP32_IP}:81/stream"

# Streaming resolution currently configured in the firmware
# (FRAMESIZE_QVGA -> 320x240).
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Optional camera metadata
CAMERA_MODEL = "ESP32-CAM AI Thinker"
CAMERA_SENSOR = "OV2640"


# ============================================================
# Audio Configuration (miniDSP UMA-16 v2)
# ============================================================

# Device name detected by sounddevice / PortAudio.
AUDIO_DEVICE_NAME = "UMA16v2"

# Audio capture parameters.
SAMPLE_RATE = 48000
NUM_CHANNELS = 16
FRAME_SIZE = 1024
BIT_DEPTH = 24

# Speed of sound in air (m/s).
SPEED_OF_SOUND = 343.0


# ============================================================
# Array Geometry Configuration
# ============================================================

# Microphone array type used by the project.
ARRAY_TYPE = "URA"  # Uniform Rectangular Array

# Physical dimensions of the UMA-16 v2 board (meters).
ARRAY_WIDTH = 0.202
ARRAY_HEIGHT = 0.132

# Placeholder for microphone spacing.
# The exact geometry will be populated later from the UMA-16 CAD data.
MIC_SPACING = None


# ============================================================
# YOLO11 Segmentation Configuration
# ============================================================

# Path to the segmentation model.
MODEL_PATH = "models/yolo11n-seg.pt"

# Detection confidence threshold.
CONFIDENCE_THRESHOLD = 0.25

# Intersection-over-Union threshold.
IOU_THRESHOLD = 0.45

# Device selection: "cpu", "cuda", or "auto".
YOLO_DEVICE = "cpu"

# Verbose inference output.
YOLO_VERBOSE = False


# ============================================================
# MUSIC Localization Configuration
# ============================================================

# Angular search resolution (degrees).
GRID_RESOLUTION = 1.0

# Frequency range used for localization (Hz).
MIN_FREQUENCY = 500
MAX_FREQUENCY = 4000

# Number of dominant sound sources to estimate.
NUM_SOURCES = 1


# ============================================================
# Synchronization Configuration
# ============================================================

# Maximum allowable timestamp difference between audio and video
# before a fusion update is considered stale.
SYNC_TOLERANCE_MS = 100


# ============================================================
# Visualization Configuration
# ============================================================

WINDOW_NAME = "Multimodal Acoustic Camera"

DISPLAY_FPS = True
DISPLAY_SEGMENTATION = True
DISPLAY_ACOUSTIC_OVERLAY = True
DISPLAY_BOUNDING_BOXES = False

# Heatmap overlay transparency.
HEATMAP_ALPHA = 0.45


# ============================================================
# Logging Configuration
# ============================================================

LOG_LEVEL = "INFO"
SAVE_DEBUG_FRAMES = False
SAVE_AUDIO_RECORDINGS = False


# ============================================================
# Future Deployment Configuration
# ============================================================

# Reserved for Raspberry Pi / Jetson deployment.
TARGET_PLATFORM = "windows"