"""
demo.ui.theme
=============
Ivory & Navy theme colors (BGR) and typography constants for OpenCV GUI rendering.
Configured for high legibility and sharp, non-aliased presentation at 1600x900.
"""

import cv2

# Color Palette (BGR format for OpenCV)
BG_CANVAS = (233, 241, 244)       # #F4F1E9 Warm ivory
BG_PANEL = (217, 226, 229)        # #E5E2D9 Light stone
BG_CARD = (255, 255, 255)         # #FFFFFF White card
BG_HOVER = (222, 231, 236)        # Muted blue hover tint
BG_VISUALIZATION = (32, 24, 17)  # #111820 Dark navy-black
BORDER = (197, 206, 209)          # #D1CEC5 Muted warm grey
BORDER_ACTIVE = (119, 87, 57)     # #395777 Muted blue
BORDER_ON_DARK = (86, 74, 64)     # Subtle border on visualization surfaces

# Text Palette
TEXT_PRIMARY = (64, 43, 25)       # #192B40 Deep navy
TEXT_SECONDARY = (139, 116, 100)  # #64748B Readable secondary text on light surfaces
TEXT_MUTED = (120, 110, 100)     # Muted text on light surfaces
TEXT_ACCENT = (64, 43, 25)        # Deep navy headings
TEXT_ON_DARK = (240, 245, 245)    # #F5F5F0 Soft white
TEXT_SECONDARY_ON_DARK = (197, 187, 177)  # #B1BBC5 Muted grey-blue
TEXT_MUTED_ON_DARK = (168, 147, 120)     # #7893A8 Steel blue
TEXT_SUCCESS = (72, 104, 55)      # Darkened success text for light surfaces
TEXT_WARNING = (37, 94, 143)      # Darkened warning text for light surfaces
TEXT_ERROR = (69, 69, 153)        # Darkened error text for light surfaces

# Semantic and visualization accents (BGR format)
ACCENT_UI = (119, 87, 57)         # #395777 Muted blue
ACCENT_PRIMARY = TEXT_PRIMARY      # #192B40 Primary interaction
ACCENT_SECONDARY = ACCENT_UI       # #395777 Secondary interaction
ACCENT_CYAN = (245, 205, 55)      # Retained for scientific visualization accents
ACCENT_GREEN = (104, 133, 67)     # #438568 Restrained green
ACCENT_RED = (83, 83, 185)        # #B95353 Restrained red
ACCENT_AMBER = (61, 131, 183)     # #B7833D Muted amber
ACCENT_BLUE = (245, 165, 55)      # Retained acoustic visualization accent
ACCENT_PURPLE = (230, 115, 210)   # Retained YOLO visualization accent
ACCENT_ORANGE = (45, 150, 255)    # Retained ground-truth visualization accent

# Typography Constants - Using FONT_HERSHEY_SIMPLEX as the clean, non-pixelated primary font
FONT_PRIMARY = cv2.FONT_HERSHEY_SIMPLEX
FONT_TITLE = cv2.FONT_HERSHEY_SIMPLEX
FONT_HEADER = cv2.FONT_HERSHEY_SIMPLEX
FONT_BODY = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX
FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

# Typography Scale Hierarchy (Calibrated specifically for 1600x900 native resolution)
SCALE_APP_TITLE = 0.72      # Main application title (thickness 2)
SCALE_MODE_TITLE = 0.65     # Panel / Mode main title (thickness 2)
SCALE_SECTION = 0.56        # Section / Card header (thickness 1 or 2)
SCALE_LABEL = 0.48          # Metric / Field label (thickness 1)
SCALE_VALUE = 0.50          # Telemetry readout / data value (thickness 1)
SCALE_VALUE_BOLD = 0.54     # Important numerical value (thickness 2)
SCALE_TABLE_HEADER = 0.46   # Table column headers (thickness 1)
SCALE_TABLE_VALUE = 0.48    # Table cell values (thickness 1)
SCALE_TAB = 0.46            # Top navigation tab labels (thickness 1)
SCALE_SMALL = 0.40          # Explanatory notes / footnotes (thickness 1)

# Shared geometry tokens for cards, controls, and the application shell.
RADIUS_CARD = 12
RADIUS_BUTTON = 9
RADIUS_BADGE = 7
SPACE_XS = 6
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16
SPACE_XL = 24

# Backward-compatibility aliases
SCALE_TITLE = SCALE_APP_TITLE
SCALE_SUBTITLE = 0.52
SCALE_PANEL_TITLE = SCALE_MODE_TITLE

# Canvas Dimensions
CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 900
HEADER_HEIGHT = 68
FOOTER_HEIGHT = 44
SIDEBAR_WIDTH = 560
MAIN_VIEW_WIDTH = CANVAS_WIDTH - SIDEBAR_WIDTH
MAIN_VIEW_HEIGHT = CANVAS_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT
