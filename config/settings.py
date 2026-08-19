"""
config.settings
===============

Centralized configuration for the Multimodal Acoustic Camera project.

All hardware-specific and algorithm-specific configuration values should
be defined here rather than hardcoded inside individual modules.

Project
-------
Real-Time Multimodal Acoustic Camera

Author
------
Kush Sahu
"""

# ============================================================
# CAMERA CONFIGURATION
# ESP32-CAM AI Thinker + OV2640
# ============================================================

# Current IP address assigned to the ESP32-CAM.
#
# IMPORTANT:
# This address may change if the ESP32 receives a new DHCP lease.
# Update this value whenever the ESP32 IP changes.
ESP32_IP = "172.27.8.165"

# ESP32-CAM MJPEG stream endpoint.
STREAM_URL = f"http://{ESP32_IP}:81/stream"

# ------------------------------------------------------------
# Camera stream resolution
# ------------------------------------------------------------

# The current ESP32 firmware uses:
#
#     FRAMESIZE_QVGA
#
# Therefore the actual streamed image is currently 320 × 240.
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Camera hardware identification.
CAMERA_MODEL = "ESP32-CAM AI Thinker"
CAMERA_SENSOR = "OV2640"


# ============================================================
# AUDIO CONFIGURATION
# miniDSP UMA-16 v2
# ============================================================

# Device name used for automatic sounddevice/PortAudio discovery.
AUDIO_DEVICE_NAME = "UMA16v2"

# ------------------------------------------------------------
# Audio stream parameters
# ------------------------------------------------------------

# UMA-16 sample rate.
SAMPLE_RATE = 48000

# Number of simultaneous microphone channels.
NUM_CHANNELS = 16

# Audio sample resolution.
#
# The UMA-16 provides 24-bit audio, although the sounddevice
# stream is handled as floating-point samples by the Python
# acquisition layer.
BIT_DEPTH = 24

# ------------------------------------------------------------
# Analysis buffer
# ------------------------------------------------------------

# Number of samples in one MUSIC analysis frame.
#
# 4096 samples at 48 kHz corresponds to approximately:
#
#     4096 / 48000 = 85.33 ms
#
# This is the value currently used by AudioAcquisition.
FRAME_SIZE = 4096

# Number of new samples received between consecutive analysis
# windows.
#
# 2048 samples corresponds to approximately 42.67 ms at 48 kHz.
#
# Therefore the analysis windows overlap by 50%.
HOP_SIZE = 2048


# ============================================================
# ACOUSTIC ARRAY CONFIGURATION
# miniDSP UMA-16 v2
# ============================================================

# Array topology.
ARRAY_TYPE = "URA"

# Number of physical microphones.
NUM_MICROPHONES = 16

# Physical microphone arrangement.
ARRAY_ROWS = 4
ARRAY_COLS = 4

# ------------------------------------------------------------
# Physical board dimensions
# ------------------------------------------------------------

# These describe the approximate physical PCB dimensions.
#
# They are metadata and are NOT used directly to generate the
# microphone coordinates.
ARRAY_WIDTH = 0.202
ARRAY_HEIGHT = 0.132

# ------------------------------------------------------------
# Microphone spacing
# ------------------------------------------------------------

# Center-to-center spacing between adjacent microphone elements.
#
# The actual geometry module currently uses 23.5 mm:
#
#     acoustic.geometry.MIC_SPACING
#
# Keep this value here as centralized configuration as well,
# but geometry.py remains the authoritative coordinate generator
# until the geometry architecture is refactored.
MIC_SPACING = 0.0235


# ============================================================
# SPEED OF SOUND
# ============================================================

# Speed of sound used by the steering-vector calculations.
#
# Units: metres / second
SPEED_OF_SOUND = 343.0


# ============================================================
# MUSIC LOCALIZATION CONFIGURATION
# ============================================================

# Number of acoustic sources currently estimated by MUSIC.
NUM_SOURCES = 1

# Temporal smoothing factor for the CSM.
#
# Used by MusicLocalizer:
#
#     new_CSM =
#         smoothing * current_CSM
#         + (1 - smoothing) * previous_CSM
#
# Higher values respond faster.
# Lower values provide more temporal smoothing.
MUSIC_SMOOTHING = 0.3

# ------------------------------------------------------------
# Acoustic frequency range
# ------------------------------------------------------------

# Frequency range used by the steering/MUSIC pipeline.
#
# The current SteeringFactory uses these values to generate:
#
#     300, 400, 500, ..., 4000 Hz
#
# resulting in 38 frequency bins.
FREQ_MIN = 300
FREQ_MAX = 4000
FREQ_STEP = 100

# These aliases describe the same frequency range at the
# higher-level configuration layer.
MIN_FREQUENCY = FREQ_MIN
MAX_FREQUENCY = FREQ_MAX


# ============================================================
# ACOUSTIC LOCALIZATION GRID
# ============================================================

# The MUSIC search is performed on a horizontal plane located
# in front of the microphone array.
#
# Coordinates are expressed in metres.

# Distance of the localization plane from the microphone array.
GRID_Z = 1.0

# X-axis search range.
GRID_X_MIN = -1.0
GRID_X_MAX = 1.0

# Y-axis search range.
GRID_Y_MIN = -1.0
GRID_Y_MAX = 1.0

# Spatial resolution of the search grid.
#
# 0.02 m = 2 cm.
#
# With the current ±1 m range this produces:
#
#     101 × 101 = 10,201 spatial points.
GRID_INCREMENT = 0.02

# Human-readable equivalent of grid resolution.
GRID_RESOLUTION = GRID_INCREMENT


# ============================================================
# VISION / YOLO11 SEGMENTATION CONFIGURATION
# ============================================================

# YOLO11 instance-segmentation model.
MODEL_PATH = "models/yolo11n-seg.pt"

# Minimum detection confidence.
CONFIDENCE_THRESHOLD = 0.25

# Non-maximum suppression IoU threshold.
IOU_THRESHOLD = 0.45

# Inference device.
#
# Current development target is the Windows laptop CPU.
#
# Supported conceptual values:
#
#     "cpu"
#     "cuda"
#     "auto"
YOLO_DEVICE = "cpu"

# Suppress verbose Ultralytics output during normal operation.
YOLO_VERBOSE = False


# ============================================================
# MULTIMODAL SYNCHRONIZATION
# ============================================================

# Maximum allowed audio/video timestamp difference before a
# fusion result is considered stale.
#
# Units: milliseconds.
SYNC_TOLERANCE_MS = 100


# ============================================================
# VISUALIZATION CONFIGURATION
# ============================================================

WINDOW_NAME = "Multimodal Acoustic Camera"

# Display FPS information.
DISPLAY_FPS = True

# Display YOLO segmentation results.
DISPLAY_SEGMENTATION = True

# Display acoustic localization overlay.
DISPLAY_ACOUSTIC_OVERLAY = True

# Display conventional bounding boxes.
#
# Currently disabled because segmentation masks are the primary
# vision representation.
DISPLAY_BOUNDING_BOXES = False

# Transparency of the acoustic heatmap when overlaid onto video.
#
# Range: 0.0 → 1.0
HEATMAP_ALPHA = 0.45


# ============================================================
# LOGGING / DEBUG CONFIGURATION
# ============================================================

# Application logging level.
LOG_LEVEL = "INFO"

# Save individual debug camera frames.
SAVE_DEBUG_FRAMES = False

# Save raw audio recordings.
SAVE_AUDIO_RECORDINGS = False


# ============================================================
# ORIENTATION CONFIGURATION
# ============================================================

# Current physical orientation of the microphone coordinate model.
#
# IMPORTANT:
# This is intentionally kept at the neutral state.
#
# We have implemented the mathematical rotation utility, but
# the physical orientation of the acoustic coordinate system has
# NOT yet been experimentally validated.
ARRAY_ROTATION_DEGREES = 0

# Horizontal mirroring.
#
# Keep disabled until experimentally validated.
ARRAY_MIRROR_X = False

# Vertical mirroring.
#
# Keep disabled until experimentally validated.
ARRAY_MIRROR_Y = False


# ============================================================
# CHANNEL MAPPING CONFIGURATION
# ============================================================

# IMPORTANT:
#
# We have performed microphone/channel calibration experiments,
# but the resulting mappings were physically inconsistent and
# therefore have NOT been accepted as a validated mapping.
#
# Consequently, no channel remapping is currently applied.
#
# The acquisition stream is therefore treated as:
#
#     audio channel 1  -> geometry index 1
#     audio channel 2  -> geometry index 2
#     ...
#     audio channel 16 -> geometry index 16
#
# This is a temporary baseline and MUST NOT be interpreted as
# experimentally validated physical microphone/channel mapping.
APPLY_CHANNEL_MAPPING = False


# ============================================================
# PLATFORM / DEPLOYMENT
# ============================================================

# Current development platform.
TARGET_PLATFORM = "windows"

# Future deployment targets.
#
# These are reserved for later development and are NOT active
# in the current implementation.
SUPPORTED_FUTURE_PLATFORMS = (
    "raspberry_pi",
    "jetson",
)


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Multimodal Acoustic Camera Configuration")
    print("=" * 60)

    print("\n[Camera]")
    print(f"Model       : {CAMERA_MODEL}")
    print(f"Sensor      : {CAMERA_SENSOR}")
    print(f"ESP32 IP    : {ESP32_IP}")
    print(f"Stream      : {STREAM_URL}")
    print(f"Resolution  : {FRAME_WIDTH} × {FRAME_HEIGHT}")

    print("\n[Audio]")
    print(f"Device      : {AUDIO_DEVICE_NAME}")
    print(f"Sample rate : {SAMPLE_RATE} Hz")
    print(f"Channels    : {NUM_CHANNELS}")
    print(f"Frame size  : {FRAME_SIZE}")
    print(f"Hop size    : {HOP_SIZE}")

    print("\n[Array]")
    print(f"Type        : {ARRAY_TYPE}")
    print(f"Microphones : {NUM_MICROPHONES}")
    print(f"Layout      : {ARRAY_ROWS} × {ARRAY_COLS}")
    print(f"Spacing     : {MIC_SPACING * 1000:.1f} mm")

    print("\n[MUSIC]")
    print(f"Frequency   : {FREQ_MIN}–{FREQ_MAX} Hz")
    print(f"Step        : {FREQ_STEP} Hz")
    print(f"Sources     : {NUM_SOURCES}")
    print(f"Smoothing   : {MUSIC_SMOOTHING}")

    print("\n[Grid]")
    print(f"X range     : {GRID_X_MIN} → {GRID_X_MAX} m")
    print(f"Y range     : {GRID_Y_MIN} → {GRID_Y_MAX} m")
    print(f"Z plane     : {GRID_Z} m")
    print(f"Increment   : {GRID_INCREMENT} m")

    print("\n[Vision]")
    print(f"Model       : {MODEL_PATH}")
    print(f"Confidence  : {CONFIDENCE_THRESHOLD}")
    print(f"IoU         : {IOU_THRESHOLD}")
    print(f"Device      : {YOLO_DEVICE}")

    print("\n[Calibration]")
    print(f"Rotation    : {ARRAY_ROTATION_DEGREES}°")
    print(f"Mirror X    : {ARRAY_MIRROR_X}")
    print(f"Mirror Y    : {ARRAY_MIRROR_Y}")
    print(f"Channel map : {APPLY_CHANNEL_MAPPING}")

    print("\n[Platform]")
    print(f"Target      : {TARGET_PLATFORM}")

    print("=" * 60)