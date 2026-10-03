"""
benchmark.config.benchmark_config
=================================
Centralized benchmark configuration constants, versioning, and tolerances.
"""

from __future__ import annotations
import os

BENCHMARK_VERSION = "1.0.0"

# Target search angular boundaries (matching production)
AZIMUTH_MIN_DEG = 65.0
AZIMUTH_MAX_DEG = 115.0
AZIMUTH_CENTER_DEG = 90.0

ELEVATION_MIN_DEG = -18.75
ELEVATION_MAX_DEG = 18.75
ELEVATION_CENTER_DEG = 0.0

ANGULAR_STEP_DEG = 0.5

# Speed of sound (m/s)
SPEED_OF_SOUND = 343.0

# Frame parameters (matching production)
SAMPLE_RATE = 48000
FRAME_SIZE = 4096
HOP_SIZE = 2048
NUM_CHANNELS = 16

# Sub-band frequencies
FREQ_CENTERS = [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0]
BAND_WIDTH = 500.0

# Success tolerance criteria
TOLERANCE_EXCELLENT_DEG = 1.0   # <= 1.0 deg error is excellent
TOLERANCE_PASS_DEG = 3.0        # <= 3.0 deg error is a pass
TOLERANCE_FAILURE_DEG = 5.0     # > 5.0 deg error is a localization failure

# Golden regression directions: (Azimuth, Elevation)
GOLDEN_DIRECTIONS = [
    (90.0, 0.0),    # Broadside center
    (103.0, 0.0),   # Moderate left
    (103.0, -5.0),  # Moderate left, down
    (103.0, -10.0), # Moderate left, further down
    (103.0, 10.0),  # Moderate left, up
    (75.0, 0.0),    # Moderate right
]

# Standard suite paths
BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BENCHMARK_ROOT, "results")
BASELINE_DIR = os.path.join(RESULTS_DIR, "baseline")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
COMPARISONS_DIR = os.path.join(RESULTS_DIR, "comparisons")
REGISTRY_FILE = os.path.join(BENCHMARK_ROOT, "registry", "benchmark_registry.json")
