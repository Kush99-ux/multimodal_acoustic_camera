# Real-Time Multimodal Acoustic Camera

A research-oriented multimodal perception system that combines **16-channel acoustic localization (MUSIC)** with **real-time visual instance segmentation (YOLO11)** using an **ESP32-CAM** and **miniDSP UMA-16 v2** microphone array.

## Project Overview

This project aims to identify **which object in a scene is producing sound** by fusing:

* **ESP32-CAM wireless video streaming**
* **YOLO11 instance segmentation**
* **16-channel acoustic localization using the MUSIC algorithm**
* **Real-time multimodal sensor fusion**

The system is being developed as a research prototype under the guidance of **Prof. Sumit Shekhar**.

## Repository Structure

```text
acoustic/     # Audio acquisition, array geometry, MUSIC localization
vision/       # ESP32 video streaming and YOLO11 segmentation
fusion/       # Sensor synchronization and multimodal association
config/       # Centralized configuration
models/       # YOLO model weights
docs/         # Architecture and hardware documentation
tests/        # Unit and integration tests
```

## Hardware

* **ESP32-CAM (AI Thinker, OV2640)**
* **miniDSP UMA-16 v2 USB Microphone Array**
* **Windows development environment**
* **Python + OpenCV + Acoular + Ultralytics YOLO11**

## Development Status

Repository initialized and architecture under active development.

## License

MIT License
