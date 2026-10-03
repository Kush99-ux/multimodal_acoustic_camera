"""
demo.ui.theme
=============
Academic Dark theme colors (BGR) and typography constants for OpenCV GUI rendering.
Configured for high legibility and sharp, non-aliased presentation at 1600x900.
"""

import cv2

# Color Palette (BGR format for OpenCV) - High-contrast academic dark
BG_CANVAS = (24, 18, 15)          # #0F1218 Deep slate
BG_PANEL = (36, 28, 22)           # #161C24 Card slate
BG_CARD = (48, 38, 30)            # #1E2630 Elevated card
BG_HOVER = (68, 56, 46)           # Interactive hover
BORDER = (75, 62, 50)             # Subtle borders
BORDER_ACTIVE = (170, 135, 80)    # Active container border

# Text Palette - High-contrast academic dark (pure bright white & light silver)
TEXT_PRIMARY = (252, 252, 252)    # High-contrast pure white
TEXT_SECONDARY = (210, 215, 222)  # Bright silver / light gray (clean, highly legible)
TEXT_MUTED = (165, 172, 182)      # Clearly legible medium-light gray (notes/hints, never dark)
TEXT_ACCENT = (250, 215, 75)      # Highlight gold/amber
TEXT_SUCCESS = (85, 235, 95)      # High-contrast bright green

# Semantic Accent Colors (High Saturation & Contrast)
ACCENT_CYAN = (245, 205, 55)      # Primary action & DoA angles
ACCENT_GREEN = (85, 235, 95)      # Success / Connected / Sync OK
ACCENT_RED = (75, 75, 250)        # Disconnected / Jitter alert
ACCENT_AMBER = (45, 210, 255)     # Warning / Threshold
ACCENT_BLUE = (245, 165, 55)      # Acoustic MUSIC marker
ACCENT_PURPLE = (230, 115, 210)   # YOLO Vision detection
ACCENT_ORANGE = (45, 150, 255)    # Ground Truth Target

# Typography Constants - Using FONT_HERSHEY_SIMPLEX as the clean, non-pixelated primary font
FONT_PRIMARY = cv2.FONT_HERSHEY_SIMPLEX
FONT_TITLE = cv2.FONT_HERSHEY_SIMPLEX
FONT_HEADER = cv2.FONT_HERSHEY_SIMPLEX
FONT_BODY = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX
FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

# Typography Scale Hierarchy (Calibrated specifically for 1600x900 native resolution)
SCALE_APP_TITLE = 0.82      # Main application title (thickness 2)
SCALE_MODE_TITLE = 0.65     # Panel / Mode main title (thickness 2)
SCALE_SECTION = 0.56        # Section / Card header (thickness 1 or 2)
SCALE_LABEL = 0.48          # Metric / Field label (thickness 1)
SCALE_VALUE = 0.50          # Telemetry readout / data value (thickness 1)
SCALE_VALUE_BOLD = 0.54     # Important numerical value (thickness 2)
SCALE_TABLE_HEADER = 0.46   # Table column headers (thickness 1)
SCALE_TABLE_VALUE = 0.48    # Table cell values (thickness 1)
SCALE_TAB = 0.46            # Top navigation tab labels (thickness 1)
SCALE_SMALL = 0.40          # Explanatory notes / footnotes (thickness 1)

# Backward-compatibility aliases
SCALE_TITLE = SCALE_APP_TITLE
SCALE_SUBTITLE = 0.52
SCALE_PANEL_TITLE = SCALE_MODE_TITLE

# Canvas Dimensions
CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 900
HEADER_HEIGHT = 64
FOOTER_HEIGHT = 44
SIDEBAR_WIDTH = 560
MAIN_VIEW_WIDTH = CANVAS_WIDTH - SIDEBAR_WIDTH
MAIN_VIEW_HEIGHT = CANVAS_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT
