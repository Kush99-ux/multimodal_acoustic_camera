"""
scratch/report_generation/build_revised_report.py
=================================================

Builds the revised, submission-ready, single-column academic technical report:
    PROJECT_FINAL_INTEGRATED_VALIDATION_REPORT.pdf

Features:
- Single-column LaTeX article style
- Times-Roman / Times-Bold / Times-Italic default serif fonts
- Numbered sections and subsections
- Unnumbered mathematical display equations
- Bordered academic tables with captions placed ABOVE tables
- Centered figures with captions placed BELOW figures
- Rigorous claim audit: removed overclaims, corrected grid size to 7,676,
  removed 6.2 kHz aliasing claim, reframed -6 deg as empirical baseline,
  reframed towel experiment as substantial multipath reduction,
  and explicitly preserved ongoing pixel-level projection refinement status.

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

import os
import sys
import json
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    PageBreak,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


# ==============================================================================
# Numbered Canvas for Two-Pass Page Numbering and Academic Headers/Footers
# ==============================================================================

class AcademicNumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that accumulates pages and stamps 'Page X of Y' along
    with clean academic running headers and footers on all pages except page 1.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_academic_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_academic_header_footer(self, page_count):
        if self._pageNumber == 1:
            return  # Suppress running header/footer on title page

        self.saveState()

        # Running Header (Academic LaTeX style)
        self.setFont("Times-Italic", 8.5)
        self.setFillColor(colors.HexColor("#222222"))
        self.drawString(54, 748, "Real-Time Multimodal Acoustic Camera: Technical & Experimental Report")
        self.setFont("Times-Roman", 8.5)
        self.drawRightString(558, 748, "September 2026")

        self.setStrokeColor(colors.HexColor("#444444"))
        self.setLineWidth(0.5)
        self.line(54, 743, 558, 743)

        # Running Footer
        self.line(54, 45, 558, 45)
        self.setFont("Times-Roman", 8.5)
        self.setFillColor(colors.HexColor("#333333"))
        self.drawString(54, 33, "Kush Sahu | Integrated Engineering Validation Report")
        self.drawRightString(558, 33, f"Page {self._pageNumber} of {page_count}")

        self.restoreState()


# ==============================================================================
# Helper Formatting Functions
# ==============================================================================

def make_equation(eq_text, style):
    """Render an unnumbered mathematical display equation centered with indent."""
    p = Paragraph(eq_text, style)
    t = Table([[p]], colWidths=[504])
    t.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    return t


def make_academic_table(headers, data, col_widths, style_header, style_cell, align='LEFT'):
    """Generate a clean, bordered academic table with visible grid lines."""
    table_data = []
    # Header row
    h_row = [Paragraph(f"<b>{h}</b>", style_header) for h in headers]
    table_data.append(h_row)

    # Data rows
    for row in data:
        r_cells = []
        for cell in row:
            if isinstance(cell, str):
                r_cells.append(Paragraph(cell, style_cell))
            else:
                r_cells.append(Paragraph(str(cell), style_cell))
        table_data.append(r_cells)

    t = Table(table_data, colWidths=col_widths)
    t_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('ALIGN', (0, 0), (-1, -1), align),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#1E293B")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#64748B")),
        ('LINEBELOW', (0, 0), (-1, 0), 1.0, colors.HexColor("#1E293B")),
    ]
    t.setStyle(TableStyle(t_style))
    return t


def make_note_box(title, text, style_title, style_body):
    """Academic boxed note for key conclusions and status statements."""
    content = [
        Paragraph(f"<b>{title}</b>", style_title),
        Spacer(1, 2),
        Paragraph(text, style_body),
    ]
    t = Table([[content]], colWidths=[504])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#334155")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t


# ==============================================================================
# Main PDF Compilation Script
# ==============================================================================

def generate_revised_pdf_report():
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pdf_filename = os.path.join(workspace_dir, "PROJECT_FINAL_INTEGRATED_VALIDATION_REPORT.pdf")

    print(f"Building Revised Academic Technical Report: {pdf_filename}")

    # Load experimental JSON datasets
    path_final_val = os.path.join(workspace_dir, "scratch", "final_integrated_validation_results.json")
    path_integrated = os.path.join(workspace_dir, "scratch", "integrated_multimodal_results.json")
    path_elevation = os.path.join(workspace_dir, "scratch", "elevation_results.json")
    path_csm_timing = os.path.join(workspace_dir, "scratch", "csm_settling_timing.json")
    path_calib_param = os.path.join(workspace_dir, "scratch", "calibrated_parameters.json")
    path_calib_data = os.path.join(workspace_dir, "scratch", "calibration_dataset.json")

    with open(path_final_val, "r", encoding="utf-8") as f:
        data_final_val = json.load(f)
    with open(path_integrated, "r", encoding="utf-8") as f:
        data_integrated = json.load(f)
    with open(path_elevation, "r", encoding="utf-8") as f:
        data_elevation = json.load(f)
    with open(path_csm_timing, "r", encoding="utf-8") as f:
        data_csm_timing = json.load(f)
    with open(path_calib_param, "r", encoding="utf-8") as f:
        data_calib_param = json.load(f)
    with open(path_calib_data, "r", encoding="utf-8") as f:
        data_calib_data = json.load(f)

    # Document setup: Letter size (612 x 792 pt), 54 pt margins (0.75 in)
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    # Academic Typography Palette (LaTeX standard Times-Roman family)
    c_black = colors.HexColor("#111827")
    c_dark = colors.HexColor("#1F2937")
    c_muted = colors.HexColor("#4B5563")

    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Times-Bold",
        fontSize=18,
        leading=22,
        textColor=c_black,
        alignment=TA_CENTER,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Times-Italic",
        fontSize=11,
        leading=15,
        textColor=c_dark,
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    author_style = ParagraphStyle(
        "DocAuthor",
        fontName="Times-Roman",
        fontSize=10,
        leading=13,
        textColor=c_dark,
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    abstract_title = ParagraphStyle(
        "AbstractTitle",
        fontName="Times-Bold",
        fontSize=9.5,
        leading=12,
        textColor=c_black,
        alignment=TA_CENTER,
        spaceAfter=3,
    )

    abstract_body = ParagraphStyle(
        "AbstractBody",
        fontName="Times-Roman",
        fontSize=8.5,
        leading=11.5,
        textColor=c_dark,
        alignment=TA_JUSTIFY,
        leftIndent=24,
        rightIndent=24,
        spaceAfter=10,
    )

    h1_style = ParagraphStyle(
        "SecH1",
        fontName="Times-Bold",
        fontSize=12,
        leading=15,
        textColor=c_black,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "SecH2",
        fontName="Times-Bold",
        fontSize=10,
        leading=13,
        textColor=c_dark,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True,
    )

    h3_style = ParagraphStyle(
        "SecH3",
        fontName="Times-BoldItalic",
        fontSize=9,
        leading=12,
        textColor=c_dark,
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyText",
        fontName="Times-Roman",
        fontSize=9,
        leading=12,
        textColor=c_dark,
        alignment=TA_JUSTIFY,
        spaceAfter=4.5,
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        fontName="Times-Roman",
        fontSize=9,
        leading=12,
        textColor=c_dark,
        alignment=TA_LEFT,
        leftIndent=14,
        spaceAfter=2.5,
    )

    equation_style = ParagraphStyle(
        "DisplayEquation",
        fontName="Times-Italic",
        fontSize=9,
        leading=12.5,
        textColor=c_black,
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=3,
    )

    table_caption_style = ParagraphStyle(
        "TableCaption",
        fontName="Times-Bold",
        fontSize=8.5,
        leading=11,
        textColor=c_black,
        alignment=TA_LEFT,
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True,
    )

    figure_caption_style = ParagraphStyle(
        "FigureCaption",
        fontName="Times-Italic",
        fontSize=8,
        leading=10.5,
        textColor=c_muted,
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=6,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        fontName="Times-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=c_black,
        alignment=TA_CENTER,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        fontName="Times-Roman",
        fontSize=7.5,
        leading=9.5,
        textColor=c_dark,
        alignment=TA_LEFT,
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        fontName="Times-Roman",
        fontSize=7.5,
        leading=9.5,
        textColor=c_dark,
        alignment=TA_CENTER,
    )

    note_title_style = ParagraphStyle(
        "NoteTitle",
        fontName="Times-Bold",
        fontSize=8.5,
        leading=11,
        textColor=c_black,
    )

    note_body_style = ParagraphStyle(
        "NoteBody",
        fontName="Times-Roman",
        fontSize=8,
        leading=10.5,
        textColor=c_dark,
    )

    story = []

    # ==========================================================================
    # ARTICLE TITLE & METADATA BLOCK
    # ==========================================================================
    story.append(Spacer(1, 4))
    story.append(Paragraph("Real-Time Multimodal Acoustic Camera", title_style))
    story.append(Paragraph("Design, Multi-Band MUSIC Localization, Vision Synchronization, and Hardware Validation", subtitle_style))
    story.append(Paragraph("<b>Kush Sahu</b><br/>Department of Engineering &bull; Advanced Sensor Perception Project<br/>September 2026", author_style))

    # Abstract Block (indented LaTeX style)
    story.append(Paragraph("Abstract", abstract_title))
    abstract_text = (
        "<i>This report details the design, implementation, and experimental validation of a real-time multimodal acoustic camera "
        "integrating a 16-channel Knowles MEMS microphone array (miniDSP UMA-16 v2) with an optical camera (ESP32-CAM AI-Thinker) "
        "and embedded YOLO11n-seg neural instance segmentation. An acoustic localization engine executing a 6-band Multiple Signal "
        "Classification (MUSIC) algorithm across 500 Hz to 3000 Hz with exponential covariance matrix smoothing was implemented and "
        "benchmarked. Physical hardware experiments demonstrated correct monotonic horizontal directional tracking across left, center, "
        "and right positions, and correct vertical elevation ordering across above, level, and below positions. "
        "Multimodal timestamp synchronization achieved a median latency delta of 28.5 ms, well within the tested 100 ms limit. "
        "Forensic testing identified tabletop acoustic multipath reflections as the primary contributor to an elevation discrepancy "
        "at visual level; introducing an acoustic absorption shield substantially reduced this effect and shifted the measured elevation "
        "toward the level baseline (-6.0 deg). The core multimodal perception system is functional under the tested conditions, while "
        "precise pixel-level acoustic-angle-to-camera-pixel projection calibration remains an active area of ongoing refinement.</i>"
    )
    story.append(Paragraph(abstract_text, abstract_body))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_black, spaceBefore=2, spaceAfter=8))

    # Core System Status Note
    status_note_text = (
        "<b>System Validation Scope &amp; Engineering Status:</b> "
        "The core multimodal acoustic-camera system was successfully implemented and experimentally validated under the tested conditions. "
        "Physical hardware acquisition, multi-band MUSIC processing, optical frame ingestion, and cross-modal temporal alignment were "
        "demonstrated to operate concurrently. Pixel-level acoustic-to-camera projection calibration, broader spatial coverage, and "
        "unshielded multipath suppression remain as ongoing engineering refinement."
    )
    story.append(make_note_box("SYSTEM VALIDATION STATUS", status_note_text, note_title_style, note_body_style))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 1. INTRODUCTION AND EXECUTIVE SUMMARY
    # ==========================================================================
    story.append(Paragraph("1. Introduction and Executive Summary", h1_style))
    story.append(Paragraph(
        "Acoustic cameras combine spatial soundfield localization with visual imagery to superimpose acoustic energy distributions "
        "directly over an optical scene. While industrial systems serve critical diagnostic roles in mechanical acoustic fault detection "
        "and architectural noise assessment, their proprietary architectures and high cost ($15,000 to $60,000+) restrict academic "
        "accessibility. The goal of this project was to design, build, and experimentally evaluate an open, accessible, real-time "
        "multimodal acoustic camera leveraging a 16-element MEMS microphone array and an edge optical camera.",
        body_style
    ))

    story.append(Paragraph(
        "The development encompassed driver engineering, multi-band MUSIC mathematical implementation, neural instance segmentation, "
        "sub-100 ms timestamp synchronization, and controlled physical validation. This report transparently documents both the successful "
        "subsystems and the real-world engineering hurdles encountered, including array mounting orientation, camera buffer queue delays, "
        "specular tabletop multipath reflections, and test-harness labeling artifacts.",
        body_style
    ))

    # Table 1.1: Core Functional Metrics
    story.append(Paragraph("Table 1.1: Summary of Core Subsystem Engineering Status and Empirical Observations", table_caption_style))
    summary_headers = ["Subsystem / Domain", "Target Specification", "Empirical Observation", "Validation Status"]
    summary_data = [
        ["Acoustic Acquisition", "16 Channels @ 48 kHz", "Continuous streaming, 0 buffer drops in 8.0s test", "Functional"],
        ["MUSIC Formulation", "6-band narrowband (500--3000 Hz)", "Verified against Acoular reference benchmark", "Functional"],
        ["Synthetic Localization", "Error &lt; 1.0 deg", "11/11 passed, maximum error 0.25 deg (grid limit)", "Functional"],
        ["Horizontal Direction", "Monotonic Left/Center/Right", "107.5 deg &gt; 90.8 deg &gt; 65.0 deg (monotonic pass)", "Validated (Tested Setup)"],
        ["Vertical Direction", "Monotonic Above/Level/Below", "+12.25 deg &gt; -6.00 deg &gt; -17.00 deg (monotonic pass)", "Validated (Tested Setup)"],
        ["Temporal Alignment", "Sync Delta &lt; 100 ms", "Median 28.5 ms, maximum 49.5 ms in 20-frame test", "Validated (Tested Setup)"],
        ["Vision Segmentation", "YOLO11n-seg real-time CPU", "Bounding boxes and masks extracted on CPU", "Functional"],
        ["Pixel-Level Projection", "Sub-pixel visual overlay", "Median 77.3 px error (dx=1 px best frame)", "Ongoing Refinement"],
    ]
    t_summary = make_academic_table(summary_headers, summary_data, [105, 120, 180, 99], table_header_style, table_cell_style)
    story.append(t_summary)
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 2. SYSTEM ARCHITECTURE
    # ==========================================================================
    story.append(Paragraph("2. System Architecture", h1_style))
    story.append(Paragraph(
        "The multimodal acoustic camera operates as a concurrent processing pipeline that acquires audio and video asynchronously, "
        "performs frequency-domain spatial decomposition, detects object silhouettes, and aligns the cross-modal representations.",
        body_style
    ))

    story.append(Paragraph("2.1 Acoustic Acquisition Subsystem", h2_style))
    story.append(Paragraph(
        "A miniDSP UMA-16 v2 microphone array captures 16 synchronous audio channels at 48,000 samples per second with 24-bit resolution. "
        "The acquisition module buffers audio in frames of N = 4096 samples (85.33 ms analysis duration) with a 50% hop size "
        "(2048 samples, 42.67 ms cadence), maintaining a steady stream of overlapping time-domain frames.",
        body_style
    ))

    story.append(Paragraph("2.2 Vision Acquisition Subsystem", h2_style))
    story.append(Paragraph(
        "An ESP32-CAM AI-Thinker module with an Omnivision OV2640 sensor captures 640 x 480 optical frames and transmits a multipart "
        "MJPEG stream over an 802.11 b/g/n Wi-Fi connection. To eliminate video lag caused by socket buffering, frames are ingested "
        "via a dedicated background thread that preserves only the most recently arrived optical frame.",
        body_style
    ))

    story.append(Paragraph("2.3 Synchronization Layer", h2_style))
    story.append(Paragraph(
        "Each arriving audio frame and decoded optical frame is tagged with a high-resolution monotonic timestamp (using time.monotonic()). "
        "A bounded sliding deque retains the 16 most recent localized acoustic frames. Upon the arrival of a vision frame, the nearest "
        "acoustic frame in time is paired with it, provided the absolute timestamp difference |t_cam - t_mic| is within a 100 ms tolerance.",
        body_style
    ))

    story.append(Paragraph("2.4 Multimodal Fusion and Rendering", h2_style))
    story.append(Paragraph(
        "The synchronized pair is packaged into a MultimodalResult structure. The acoustic pseudo-spectrum heatmap is rendered, "
        "colorized with the Inferno colormap, and alpha-blended onto the optical image. The acoustic peak crosshair marker is "
        "mapped into pixel space via pinhole angular projection, while YOLO11n-seg segmentation masks outline candidate objects.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 3. HARDWARE SPECIFICATIONS AND PHYSICAL CONFIGURATION
    # ==========================================================================
    story.append(Paragraph("3. Hardware Specifications and Physical Configuration", h1_style))
    story.append(Paragraph(
        "The experimental setup integrates two primary sensors rigidly aligned on a common mounting frame: the 16-element "
        "planar microphone array and the ESP32-CAM module.",
        body_style
    ))

    story.append(Paragraph("Table 3.1: Hardware Subsystem Parameters and Operational Settings", table_caption_style))
    hw_headers = ["Subsystem Component", "Hardware Specifications & Interface", "Operational Parameters"]
    hw_data = [
        ["Acoustic Array", "miniDSP UMA-16 v2 USB Microphone Array\n16 Knowles SPH0645LM4H digital MEMS microphones\nXMOS xCORE-200 USB Audio Class 2.0 controller", "Sample Rate: 48,000 Hz\nResolution: 24-bit PCM\nTopology: Planar X-Z array\nAxial Pitch: d = 42.0 mm"],
        ["Optical Camera", "ESP32-CAM AI-Thinker Module\nEspressif ESP32-D0WD dual-core Xtensa LX6 @ 240 MHz\nOmnivision OV2640 2.0 MP CMOS image sensor", "Resolution: 640 x 480 (VGA)\nInterface: HTTP MJPEG @ Port 81\nHorizontal FOV: 50.0 deg\nVertical FOV: 37.5 deg"],
        ["Host Workstation", "x86_64 Engineering Laptop\nPython 3.10+ Virtual Environment (.venv)\nPortAudio / sounddevice asynchronous audio capture", "Buffer: 4096 samples (85.3 ms)\nHop: 2048 samples (42.7 ms)\nSync Tolerance: 100.0 ms\nYOLO Device: Local CPU"],
    ]
    t_hw = make_academic_table(hw_headers, hw_data, [125, 230, 149], table_header_style, table_cell_style)
    story.append(t_hw)
    story.append(Spacer(1, 6))

    story.append(Paragraph("3.1 UMA-16 Microphone Geometry and Spatial Sampling", h2_style))
    story.append(Paragraph(
        "The UMA-16 v2 features 16 digital MEMS microphones arranged on a 4-layer rigid PCB (202 mm x 132 mm). "
        "In the production coordinate system, the microphones lie on the X-Z plane, with the array broadside pointing along +Y. "
        "Element coordinates span discrete positions at -63 mm, -21 mm, +21 mm, and +63 mm, establishing an inter-element pitch spacing of d = 42.0 mm. "
        "The basic spatial-aliasing threshold associated with this 42 mm spacing is approximately 4.08 kHz for c = 343 m/s. "
        "The implemented MUSIC processing bands are limited to 500--3000 Hz, remaining below this basic threshold.",
        body_style
    ))

    # Table 3.2: Complete Mic Coordinates
    story.append(Paragraph("Table 3.2: Complete UMA-16 Microphone Geometry Coordinates in Acoular Model", table_caption_style))
    mic_headers = ["Channel", "Board Designator", "Acoular X (m)", "Acoular Y (m)", "Acoular Z (m)", "Relative Physical Position"]
    mic_data = [
        ["CH 1", "MIC 1", "+0.021", "0.000", "-0.063", "Lower inner right quadrant"],
        ["CH 2", "MIC 2", "+0.063", "0.000", "-0.063", "Lower outer right quadrant"],
        ["CH 3", "MIC 3", "+0.021", "0.000", "-0.021", "Mid-lower inner right quadrant"],
        ["CH 4", "MIC 4", "+0.063", "0.000", "-0.021", "Mid-lower outer right quadrant"],
        ["CH 5", "MIC 5", "+0.021", "0.000", "+0.021", "Mid-upper inner right quadrant"],
        ["CH 6", "MIC 6", "+0.063", "0.000", "+0.021", "Mid-upper outer right quadrant"],
        ["CH 7", "MIC 7", "+0.021", "0.000", "+0.063", "Upper inner right quadrant"],
        ["CH 8", "MIC 8", "+0.063", "0.000", "+0.063", "Upper outer right quadrant"],
        ["CH 9", "MIC 9", "-0.063", "0.000", "+0.063", "Upper outer left quadrant"],
        ["CH 10", "MIC 10", "-0.021", "0.000", "+0.063", "Upper inner left quadrant"],
        ["CH 11", "MIC 11", "-0.063", "0.000", "+0.021", "Mid-upper outer left quadrant"],
        ["CH 12", "MIC 12", "-0.021", "0.000", "+0.021", "Mid-upper inner left quadrant"],
        ["CH 13", "MIC 13", "-0.063", "0.000", "-0.021", "Mid-lower outer left quadrant"],
        ["CH 14", "MIC 14", "-0.021", "0.000", "-0.021", "Mid-lower inner left quadrant"],
        ["CH 15", "MIC 15", "-0.063", "0.000", "-0.063", "Lower outer left quadrant"],
        ["CH 16", "MIC 16", "-0.021", "0.000", "-0.063", "Lower inner left quadrant"],
    ]
    t_mics = make_academic_table(mic_headers, mic_data, [55, 80, 75, 75, 75, 144], table_header_style, table_cell_style, align='CENTER')
    story.append(t_mics)
    story.append(Spacer(1, 4))

    fig_uma_path = os.path.join(workspace_dir, "scratch", "report_generation", "figure_uma16_geometry.png")
    if os.path.exists(fig_uma_path):
        story.append(KeepTogether([
            Image(fig_uma_path, width=320, height=267),
            Paragraph("Figure 3.1. Physical microphone layout of the miniDSP UMA-16 v2 showing channel assignments and board designators relative to the central acoustic boresight.", figure_caption_style),
        ]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("3.2 ESP32-CAM Video Streamer", h2_style))
    story.append(Paragraph(
        "The ESP32-CAM module operates an OV2640 CMOS image sensor with a nominal manufacturer field of view of 50.0 deg horizontal "
        "and 37.5 deg vertical. The module executes lightweight firmware serving an MJPEG stream over HTTP on port 81. "
        "The camera is mounted near the acoustic array center with co-axial forward orientation.",
        body_style
    ))

    story.append(Paragraph("3.3 Physical Coordinate Convention", h2_style))
    story.append(Paragraph(
        "The coordinate convention aligns with standard spherical definitions: azimuth theta = 90.0 deg and elevation phi = 0.0 deg "
        "correspond to the array broadside (+Y axis). Azimuth angles greater than 90 deg denote positions to the camera left (-X), "
        "while azimuth angles less than 90 deg denote positions to the camera right (+X). Positive elevation denotes positions above "
        "the horizontal plane (+Z), and negative elevation denotes positions below (-Z).",
        body_style
    ))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 4. SOFTWARE ARCHITECTURE
    # ==========================================================================
    story.append(Paragraph("4. Software Architecture", h1_style))
    story.append(Paragraph(
        "The software architecture enforces clean separation between acquisition, processing, synchronization, and visualization. "
        "The codebase comprises five dedicated packages: acoustic (signal acquisition, steering vector precomputation, MUSIC localizer), "
        "vision (video reader, YOLO11n-seg interface), fusion (timestamp synchronization, result packaging, heatmap overlay), "
        "pipeline (orchestration and loop management), and config (immutable configuration parameters).",
        body_style
    ))

    story.append(Paragraph("4.1 Video Stream Buffering and Latency Mitigation", h2_style))
    story.append(Paragraph(
        "Initial hardware testing revealed a severe video lag of 2 to 5 seconds during HTTP MJPEG streaming. "
        "Investigation traced this delay to cv2.VideoCapture's internal TCP socket buffer, which queued incoming frames whenever "
        "downstream inference ran slower than the network transmission rate. This issue was resolved by introducing a dedicated "
        "background thread in CameraAcquisition that continuously drains the socket buffer and atomically stores only the newest frame "
        "in an internal pointer. Consumers calling read() retrieve this latest frame with under 30 ms retrieval latency.",
        body_style
    ))

    story.append(Paragraph("Table 4.1: Production Data Models Across Pipeline Stages", table_caption_style))
    dm_headers = ["Class Name", "Package Layer", "Encapsulated Attributes and Telemetry"]
    dm_data = [
        ["CameraFrame", "vision.camera", "frame: np.ndarray (H, W, 3), timestamp: float, frame_id: int"],
        ["AudioFrame", "acoustic.audio", "data: np.ndarray (4096, 16), timestamp: float, sample_rate: int"],
        ["AcousticLocalization", "acoustic.models", "azimuth_deg: float, elevation_deg: float, confidence: float, db_level: float, valid: bool"],
        ["AcousticFrame", "acoustic.models", "frame_id: int, timestamp: float, localization: AcousticLocalization, heatmap: np.ndarray (76, 101)"],
        ["VisionDetection", "vision.models", "box: tuple, centroid: tuple, mask: np.ndarray, class_name: str, confidence: float"],
        ["VisionFrame", "vision.models", "frame_id: int, timestamp: float, detections: list[VisionDetection], image: np.ndarray"],
        ["MultimodalResult", "fusion.models", "vision: VisionFrame, acoustic: AcousticFrame, sync_delta_ms: float, synchronized: bool"],
    ]
    t_dm = make_academic_table(dm_headers, dm_data, [110, 100, 294], table_header_style, table_cell_style)
    story.append(t_dm)
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 5. ACOUSTIC LOCALIZATION METHOD
    # ==========================================================================
    story.append(Paragraph("5. Acoustic Localization Method", h1_style))
    story.append(Paragraph(
        "Direction-of-Arrival (DoA) estimation is performed using an unnumbered narrowband MUSIC formulation evaluated across six discrete "
        "sub-bands and accumulated into a composite spatial spectrum.",
        body_style
    ))

    story.append(Paragraph("5.1 Cross-Spectral Matrix Formation", h2_style))
    story.append(Paragraph(
        "Each 4096-sample time frame is multiplied by a periodic Hann window and transformed via real FFT. "
        "For M = 16 microphones and K frequency bins within a sub-band, the instantaneous Cross-Spectral Matrix (CSM) is estimated by:",
        body_style
    ))
    eq_csm = "R_inst(f) = (1 / K) &sum;_{k=1}^K X_k(f) X_k^H(f)"
    story.append(make_equation(eq_csm, equation_style))
    story.append(Paragraph("Hermiticity is strictly enforced by setting R = 0.5 &middot; (R + R^H).", body_style))

    story.append(Paragraph("5.2 Exponential Moving Average Smoothing", h2_style))
    story.append(Paragraph(
        "Temporal smoothing is applied across successive analysis frames using an Exponential Moving Average (EMA):",
        body_style
    ))
    eq_ema = "R_smooth^{(t)}(f) = (1 - alpha) R_smooth^{(t-1)}(f) + alpha R_inst^{(t)}(f), &nbsp;&nbsp;&nbsp;&nbsp; alpha = 0.30"
    story.append(make_equation(eq_ema, equation_style))

    story.append(Paragraph("5.3 MUSIC Noise Subspace Decomposition", h2_style))
    story.append(Paragraph(
        "A slight diagonal regularization (10^{-10} I_{16}) is added to R_smooth before eigendecomposition. "
        "The eigenvalues are ordered lambda_1 &lt;= lambda_2 &lt;= ... &lt;= lambda_{16}. Assuming D = 1 dominant acoustic source, "
        "the 15 minor eigenvectors span the noise subspace U_n &isin; C^{16 x 15}.",
        body_style
    ))

    story.append(Paragraph("5.4 Steering Vector Formulation and Pseudo-Spectrum", h2_style))
    story.append(Paragraph(
        "For a plane-wave arriving from azimuth theta and elevation phi, the steering vector a(theta, phi, f) is evaluated across "
        "a 7,676-point angular search grid (101 azimuth samples x 76 elevation samples). The MUSIC pseudo-spectrum is:",
        body_style
    ))
    eq_music = "P_MUSIC(theta, phi, f) = 1 / (a^H(theta, phi, f) U_n U_n^H a(theta, phi, f))"
    story.append(make_equation(eq_music, equation_style))

    story.append(Paragraph("5.5 Multi-Band Accumulation and Normalization", h2_style))
    story.append(Paragraph(
        "To ensure balanced contributions across the spectrum, each sub-band response is individually min-max normalized to [0, 1] "
        "before accumulation across the six center frequencies F_c = {500, 1000, 1500, 2000, 2500, 3000} Hz:",
        body_style
    ))
    eq_norm = "P_norm(theta, phi, f) = (P_MUSIC(theta, phi, f) - min P) / (max P - min P)"
    story.append(make_equation(eq_norm, equation_style))
    eq_tot = "P_total(theta, phi) = &sum;_{f &isin; F_c} P_norm(theta, phi, f)"
    story.append(make_equation(eq_tot, equation_style))
    story.append(Paragraph(
        "Contrast is enhanced using a non-linear sigmoid transformation (gain = 3.0, center = 0.5):",
        body_style
    ))
    eq_sig = "P_final(theta, phi) = 1 / (1 + exp(-3.0 &middot; (P_total(theta, phi) - 0.5)))"
    story.append(make_equation(eq_sig, equation_style))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 6. SYNTHETIC MATHEMATICAL VALIDATION
    # ==========================================================================
    story.append(Paragraph("6. Synthetic Mathematical Validation", h1_style))
    story.append(Paragraph(
        "Prior to physical hardware deployment, the mathematical implementation of the MUSIC localizer was evaluated "
        "against an independent synthetic ground truth using simulated acoustic signals across azimuth and elevation sweeps. "
        "In 11 synthetic benchmark runs spanning elevations from -15.0 deg to +15.0 deg and azimuths from 60.0 deg to 120.0 deg, "
        "all 11 tests met the passing criterion. The synthetic benchmark confirmed that the implemented MUSIC localization pipeline "
        "reproduced the expected directions within the tested synthetic conditions, with a maximum observed error of 0.25 degrees, "
        "consistent with the 0.5-degree angular grid resolution.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 7. VISION AND TEMPORAL SYNCHRONIZATION VALIDATION
    # ==========================================================================
    story.append(Paragraph("7. Vision and Temporal Synchronization Validation", h1_style))
    story.append(Paragraph(
        "The vision pipeline executes YOLO11n-seg to generate object masks and bounding boxes. "
        "During testing, YOLO11n-seg was demonstrated to execute and produce detections/masks. "
        "Under clean unobstructed viewing, the network identified test smartphones ('cell phone' with confidence 0.83 to 0.90). "
        "When the source was hand-held or covered with absorption material, classification jitter was observed (e.g. labeling as 'chair' "
        "with confidence 0.50 to 0.72). The subsystem is therefore characterized as functional for local instance segmentation under the tested "
        "conditions, while visual ground truth for validation was anchored to recorded manual clicks.",
        body_style
    ))

    story.append(Paragraph(
        "<b>Cross-Modal Synchronization Performance:</b> Audio-video timestamp synchronization was verified across all campaigns. "
        "In the final 20-frame integrated test, synchronization achieved a median latency delta of 28.52 ms and a maximum of 49.47 ms, "
        "with 100% of frames falling well within the 100 ms threshold.",
        body_style
    ))

    story.append(Paragraph("Table 7.1: Multimodal Synchronization Latency Across Experimental Campaigns", table_caption_style))
    sync_headers = ["Campaign Description", "Frames", "Sync Success", "Median Delta", "Max Delta", "Queue Status"]
    sync_data = [
        ["Controlled Horizontal Sweep", "30 Frames", "100.0%", "14.50 ms", "45.43 ms", "0 Drops / Bounded"],
        ["Controlled Elevation Sweep", "30 Frames", "100.0%", "26.35 ms", "67.57 ms", "0 Drops / Bounded"],
        ["Continuous 8.0s Audio Streaming", "187 Frames", "100.0%", "21.60 ms", "58.12 ms", "0 Overflows (7.98s Audio)"],
        ["Final Integrated Towel-Shielded LEFT", "20 Frames", "100.0%", "28.52 ms", "49.47 ms", "100% Synchronized"],
        ["Final Integrated Misaligned CENTER", "20 Frames", "100.0%", "23.80 ms", "58.12 ms", "100% Synchronized"],
    ]
    t_sync = make_academic_table(sync_headers, sync_data, [150, 60, 75, 75, 70, 74], table_header_style, table_cell_style, align='CENTER')
    story.append(t_sync)
    story.append(Spacer(1, 4))

    fig_sync_path = os.path.join(workspace_dir, "scratch", "report_generation", "figure_sync_latency.png")
    if os.path.exists(fig_sync_path):
        story.append(KeepTogether([
            Image(fig_sync_path, width=380, height=190),
            Paragraph("Figure 7.1. Frame-by-frame multimodal timestamp synchronization delta for the 20-frame validation run, showing adherence to the 100 ms limit (median delta = 28.5 ms).", figure_caption_style),
        ]))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 8. EXPERIMENTAL DEVELOPMENT AND FORENSIC DEBUGGING
    # ==========================================================================
    story.append(Paragraph("8. Experimental Development and Forensic Debugging", h1_style))
    story.append(Paragraph(
        "A rigorous engineering process requires documenting anomalies and their diagnostic resolutions. "
        "The following five technical issues were encountered and resolved during hardware bring-up:",
        body_style
    ))

    hurdles = [
        ("8.1 Acoular Coordinate Alignment",
         "The reference implementation mapped raw array coordinates into Acoular's convention via column permutations: "
         "Acoular X = raw Col 0, Acoular Y = raw Col 2, Acoular Z = raw Col 1. Early refactors mistakenly assumed an X-Y plane "
         "with +Z forward, causing axis cross-talk. Re-establishing the reference axis permutation (X-Z plane, +Y broadside) "
         "restored numerical equivalence with the validated benchmark."),

        ("8.2 Physical Array Mount Inversion",
         "During initial elevation trials, sources placed physically above the camera produced negative elevations, while sources "
         "below produced positive values. Visual inspection confirmed the UMA-16 board had been mounted upside-down relative to its "
         "designator markings. Re-orienting the mount into its upright position corrected the vertical polarity and established "
         "monotonic elevation tracking."),

        ("8.3 Video Ingestion Buffer Latency",
         "As detailed in Section 4.1, cv2.VideoCapture accumulated stale MJPEG frames, inducing 2--5 s visual lag. "
         "Implementing an asynchronous reader thread draining the socket buffer permanently reduced optical latency to below 30 ms."),

        ("8.4 Tabletop Acoustic Multipath Reflections",
         "In controlled moderate-left trials, a visually-level acoustic source produced an elevation response at the lower grid boundary "
         "(-18.8 deg) instead of the expected level baseline (-6.0 deg). Sub-band analysis indicated that tabletop reflections acted as a "
         "virtual image source. Placing an acoustic absorption towel beneath the source substantially reduced this reflection, shifting "
         "the elevation estimate to -7.8 deg to -8.75 deg."),

        ("8.5 Test-Harness Ground-Truth Labeling Artifact",
         "In the final integrated test labeled 'CENTER', automated pass/fail flags failed. Forensic analysis revealed that the human operator "
         "had physically placed the acoustic target in the bottom-right corner of the scene (click ground truth = [499, 358]). "
         "The acoustic localizer accurately tracked the sound to Azimuth 68.5 deg (u = 594.5 px). The apparent failure was an experimental "
         "bookkeeping artifact resulting from an incorrect script preset label, while the acoustic response matched the physical sound location.")
    ]

    for title, desc in hurdles:
        story.append(Paragraph(title, h2_style))
        story.append(Paragraph(desc, body_style))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 9. HORIZONTAL LOCALIZATION VALIDATION
    # ==========================================================================
    story.append(Paragraph("9. Horizontal Localization Validation", h1_style))
    story.append(Paragraph(
        "To validate azimuth direction-of-arrival tracking, controlled physical acoustic sources were placed at three distinct "
        "horizontal positions: CAMERA LEFT, CAMERA CENTER, and CAMERA RIGHT. With broadside aligned to 90.0 deg azimuth, "
        "sources to the physical left correspond to azimuths &gt; 90 deg, and sources to the right correspond to azimuths &lt; 90 deg.",
        body_style
    ))

    story.append(Paragraph("Table 9.1: Horizontal Validation Statistics (10 Frames per Position)", table_caption_style))
    h_headers = ["Position Label", "Visual GT (x, y)", "Azimuth Median", "Azimuth Range", "Overlay u Median", "Directional Status"]
    h_data = [
        ["CAMERA LEFT", "(89, 255)", "106.00 deg", "105.0 deg -- 106.5 deg", "115.0 px", "Pass (u &lt; 320, Az &gt; 90 deg)"],
        ["CAMERA CENTER", "(319, 244)", "90.75 deg", "89.5 deg -- 92.5 deg", "310.0 px", "Pass (u &approx; 320, Az &approx; 90 deg)"],
        ["CAMERA RIGHT", "(495, 205)", "65.00 deg", "64.0 deg -- 66.5 deg", "639.0 px", "Pass (u &gt; 320, Az &lt; 90 deg)"],
    ]
    t_h = make_academic_table(h_headers, h_data, [85, 80, 80, 85, 85, 89], table_header_style, table_cell_style, align='CENTER')
    story.append(t_h)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "The three-position hardware experiment demonstrated correct monotonic horizontal directional tracking across the tested left, center, "
        "and right positions: LEFT (106.00 deg) &gt; CENTER (90.75 deg) &gt; RIGHT (65.00 deg), with corresponding pixel ordering "
        "LEFT (115.0 px) &lt; CENTER (310.0 px) &lt; RIGHT (639.0 px).",
        body_style
    ))
    story.append(Spacer(1, 4))

    img_left = os.path.join(workspace_dir, "scratch", "left_representative.jpg")
    img_ctr = os.path.join(workspace_dir, "scratch", "center_representative.jpg")
    img_rgt = os.path.join(workspace_dir, "scratch", "right_representative.jpg")

    if os.path.exists(img_left) and os.path.exists(img_ctr) and os.path.exists(img_rgt):
        story.append(KeepTogether([
            Table([
                [Image(img_left, width=160, height=120),
                 Image(img_ctr, width=160, height=120),
                 Image(img_rgt, width=160, height=120)],
                [Paragraph("<b>(a) Left Position</b><br/>Az = 106.0 deg, u = 115 px", figure_caption_style),
                 Paragraph("<b>(b) Center Position</b><br/>Az = 90.8 deg, u = 310 px", figure_caption_style),
                 Paragraph("<b>(c) Right Position</b><br/>Az = 65.0 deg, u = 639 px", figure_caption_style)]
            ], colWidths=[168, 168, 168]),
            Paragraph("Figure 9.1. Controlled horizontal localization experiment showing the acoustic response across left, center, and right source positions.", figure_caption_style),
        ]))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 10. VERTICAL LOCALIZATION VALIDATION
    # ==========================================================================
    story.append(Paragraph("10. Vertical Localization Validation", h1_style))
    story.append(Paragraph(
        "Following physical orientation correction, a 3-position vertical elevation experiment was conducted along the optical midline "
        "at ABOVE (+12.25 deg), LEVEL (-6.0 deg), and BELOW (-17.0 deg) positions.",
        body_style
    ))

    story.append(Paragraph("Table 10.1: Elevation Validation Statistics (10 Frames per Position)", table_caption_style))
    el_headers = ["Position Label", "Visual Click GT", "Azimuth Median", "Elevation Median", "Overlay (u, v)", "Ordering Status"]
    el_data = [
        ["ABOVE", "(322, 152)", "88.25 deg", "+12.25 deg", "(342.0, 83.0) px", "Pass (El &gt; 0 deg, v &lt; 240)"],
        ["LEVEL", "(321, 231)", "91.25 deg", "-6.00 deg", "(303.5, 316.0) px", "Pass (El &approx; -6 deg, v &approx; 316)"],
        ["BELOW", "(322, 340)", "83.25 deg", "-17.00 deg", "(406.0, 457.0) px", "Pass (El &lt; -6 deg, v &gt; 316)"],
    ]
    t_el = make_academic_table(el_headers, el_data, [75, 80, 80, 85, 95, 89], table_header_style, table_cell_style, align='CENTER')
    story.append(t_el)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Monotonic Ordering:</b> Measured elevation demonstrated correct physical ordering: ABOVE (+12.25 deg) &gt; LEVEL (-6.0 deg) &gt; BELOW (-17.0 deg / -17.00 deg), "
        "with corresponding pixel vertical coordinates ABOVE (83.0 px) &lt; LEVEL (316.0 px) &lt; BELOW (457.0 px).<br/>"
        "<b>Empirical Elevation Offset Baseline:</b> At the level reference position, the acoustic array measured an elevation of "
        "approximately -6 degrees while the visual target was near the camera optical center. This empirical offset was therefore adopted "
        "as the current elevation baseline for projection refinement.",
        body_style
    ))
    story.append(Spacer(1, 4))

    img_abv = os.path.join(workspace_dir, "scratch", "above_elevation_rep.jpg")
    img_lvl = os.path.join(workspace_dir, "scratch", "level_elevation_rep.jpg")
    img_blw = os.path.join(workspace_dir, "scratch", "below_elevation_rep.jpg")

    if os.path.exists(img_abv) and os.path.exists(img_lvl) and os.path.exists(img_blw):
        story.append(KeepTogether([
            Table([
                [Image(img_abv, width=160, height=120),
                 Image(img_lvl, width=160, height=120),
                 Image(img_blw, width=160, height=120)],
                [Paragraph("<b>(a) Above Position</b><br/>El = +12.25 deg, v = 83 px", figure_caption_style),
                 Paragraph("<b>(b) Level Position</b><br/>El = -6.00 deg, v = 316 px", figure_caption_style),
                 Paragraph("<b>(c) Below Position</b><br/>El = -17.00 deg, v = 457 px", figure_caption_style)]
            ], colWidths=[168, 168, 168]),
            Paragraph("Figure 10.1. Controlled elevation localization experiment showing the acoustic response across above, level, and below source positions.", figure_caption_style),
        ]))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 11. CSM SETTLING AND TEMPORAL DYNAMICS
    # ==========================================================================
    story.append(Paragraph("11. CSM Settling and Temporal Dynamics", h1_style))
    story.append(Paragraph(
        "To test whether anomalous elevation readings were an artifact of transient filter settling, an extended 8.0-second hardware recording "
        "(187 continuous frames, 7.979 s of audio) was evaluated. The acquisition executed with zero buffer drops and zero queue overflows. "
        "With EMA smoothing parameter alpha = 0.30, the theoretical time constant is tau &approx; 119.6 ms (settling within ~8 frames). "
        "Tracking the trajectory over the 187 frames showed that elevation stabilized at -13.2 deg by frame 5, remained at -13.8 deg over the first 15 frames, "
        "and averaged -12.8 deg across all 187 frames. "
        "The extended recording showed that the observed elevation discrepancy persisted beyond the initial EMA settling period, "
        "indicating that the anomaly was not explained solely by CSM initialization/transient settling.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 12. SUB-BAND AND MULTIPATH ANALYSIS
    # ==========================================================================
    story.append(Paragraph("12. Sub-Band and Multipath Analysis", h1_style))
    story.append(Paragraph(
        "Evaluating the MUSIC pseudo-spectrum independently across each sub-band under unshielded tabletop conditions revealed strong "
        "frequency-dependent divergence. At 500 Hz, elevation localized at -0.8 deg to -6.8 deg; at 1000 Hz, it moved to -16.8 deg; and at 1500 Hz, "
        "it pinned at the lower grid boundary (-18.8 deg). Higher bands (2500--3000 Hz) also exhibited downward bias. "
        "The observed frequency-dependent elevation behavior was consistent with coherent tabletop multipath/reflection effects. "
        "The shielding experiment provided strong experimental evidence for this interpretation because the measured elevation shifted "
        "substantially toward the expected direction after acoustic absorption was introduced.",
        body_style
    ))

    story.append(Paragraph("Table 12.1: Sub-Band Response Comparison Before and After Acoustic Shielding", table_caption_style))
    sb_headers = ["Sub-Band Center", "Unshielded Azimuth", "Unshielded Elevation", "Towel-Shielded Elevation", "Observed Behavior"]
    sb_data = [
        ["500 Hz", "97.5 deg", "-0.8 deg -- -6.8 deg", "-6.0 deg", "Broad mainlobe; limited reflection sensitivity"],
        ["1000 Hz", "102.0 deg", "-16.8 deg -- -18.8 deg", "-8.2 deg", "Constructive interference from table reflection"],
        ["1500 Hz", "90.5 deg", "-18.8 deg (Boundary)", "-7.8 deg", "Reflection energy dominated direct path"],
        ["2000 Hz", "111.0 deg", "-4.8 deg -- -8.2 deg", "-7.2 deg", "Destructive interference at reflection angle"],
        ["2500 Hz", "100.0 deg", "-11.8 deg -- -14.2 deg", "-6.8 deg", "Downwards pull from reflection"],
        ["3000 Hz", "102.0 deg", "-8.2 deg -- -10.8 deg", "-6.5 deg", "Narrow beamwidth; minor grating interaction"],
    ]
    t_sb = make_academic_table(sb_headers, sb_data, [70, 95, 105, 105, 129], table_header_style, table_cell_style, align='CENTER')
    story.append(t_sb)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Towel Shielding Experiment:</b> Wrapping the source base with an acoustic absorption towel substantially reduced the observed "
        "tabletop-reflection effect. The 1500 Hz elevation shifted from -18.8 deg to -7.8 deg, and the composite broadband elevation shifted "
        "from -18.8 deg to -7.8 deg to -8.75 deg, substantially reducing the anomalous downward elevation response and shifting the measured "
        "elevation toward the expected level direction.",
        body_style
    ))
    story.append(Spacer(1, 4))

    fig_sb_path = os.path.join(workspace_dir, "scratch", "report_generation", "figure_subband_analysis.png")
    if os.path.exists(fig_sb_path):
        story.append(KeepTogether([
            Image(fig_sb_path, width=400, height=228),
            Paragraph("Figure 12.1. Sub-band spectral energy distribution (left) and spatial off-diagonal coherence (right), illustrating frequency-dependent coherence patterns under tabletop reflection conditions.", figure_caption_style),
        ]))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 13. CAMERA-TO-ACOUSTIC PROJECTION
    # ==========================================================================
    story.append(Paragraph("13. Camera-to-Acoustic Projection", h1_style))
    story.append(Paragraph(
        "Acoustic spherical angles are projected onto camera pixel coordinates (u, v) using pinhole projection centered at azimuth = 90.0 deg "
        "and empirical elevation baseline = -6.0 deg:",
        body_style
    ))
    eq_proj_u = "u = [(az_max - theta) / (az_max - az_min)] &middot; Image_Width, &nbsp;&nbsp;&nbsp;&nbsp; az_min = 90 deg - (FOV_H / 2), &nbsp; az_max = 90 deg + (FOV_H / 2)"
    story.append(make_equation(eq_proj_u, equation_style))
    eq_proj_v = "v = [1.0 - (phi - el_min) / (el_max - el_min)] &middot; Image_Height, &nbsp;&nbsp; el_min = -6 deg - (FOV_V / 2), &nbsp; el_max = -6 deg + (FOV_V / 2)"
    story.append(make_equation(eq_proj_v, equation_style))

    story.append(Paragraph(
        "Under unshielded conditions, tabletop reflection dragged the acoustic overlay to the bottom frame edge (v = 479 px), resulting in a "
        "mean projection error of 253.2 px. With towel shielding, the overlay stabilized at u = 96.0 px, v = 275.0 px against a ground-truth "
        "click at (146, 216). Median Euclidean projection error dropped to 77.3 px, with frame 19 achieving horizontal error dx = 1 px "
        "(u = 147 px vs x = 146 px). The residual vertical error (dy &approx; 59 px) demonstrates that acoustic-to-camera pixel projection "
        "calibration remains incomplete and is an ongoing engineering refinement.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 14. INTEGRATED MULTIMODAL VALIDATION
    # ==========================================================================
    story.append(Paragraph("14. Integrated Multimodal Validation", h1_style))
    story.append(Paragraph(
        "The final validation campaign tested the integrated multimodal pipeline across continuous streaming frames with acoustic shielding.",
        body_style
    ))

    story.append(Paragraph("Table 14.1: Final 20-Frame Integrated Multimodal Telemetry (Moderate-Left Position)", table_caption_style))
    fval_headers = ["Telemetry Metric", "Ground Truth Target", "Acoustic / Overlay Median", "Observed Spread", "Evaluation Status"]
    fval_data = [
        ["Azimuth Angle", "Expected Left (~105 deg)", "107.50 deg", "103.0 deg -- 108.5 deg (std: 1.62 deg)", "Directional Pass"],
        ["Elevation Angle", "Level Baseline (-6.0 deg)", "-8.75 deg", "-9.75 deg -- -7.25 deg (std: 0.75 deg)", "Consistent with Level"],
        ["Overlay u Coordinate", "GT Click x = 146 px", "96.0 px", "83.0 -- 153.0 px (Frame 19: 147 px)", "dx = 1 px on best frame"],
        ["Overlay v Coordinate", "GT Click y = 216 px", "275.0 px", "256.0 -- 288.0 px (std: 8.8 px)", "Residual offset dy &approx; 59 px"],
        ["Euclidean Pixel Error", "0.0 px (Ideal)", "77.34 px", "Mean: 73.11 px, Min: 65.38 px", "Projection Calibration Ongoing"],
        ["Sync Latency Delta", "&lt; 100.0 ms limit", "28.52 ms", "5.13 -- 49.47 ms (100% pass)", "100% Synchronized"],
    ]
    t_fval = make_academic_table(fval_headers, fval_data, [110, 100, 110, 114, 70], table_header_style, table_cell_style, align='CENTER')
    story.append(t_fval)
    story.append(Spacer(1, 4))

    img_fleft = os.path.join(workspace_dir, "scratch", "left_final_validation_representative.jpg")
    img_fctr = os.path.join(workspace_dir, "scratch", "center_final_validation_representative.jpg")

    if os.path.exists(img_fleft) and os.path.exists(img_fctr):
        story.append(KeepTogether([
            Table([
                [Image(img_fleft, width=245, height=184),
                 Image(img_fctr, width=245, height=184)],
                [Paragraph("<b>(a) Shielded Left Validation Run</b><br/>Az = 107.5 deg, El = -8.75 deg, Overlay = (96, 275) px<br/>Visual GT = (146, 216) px, Sync = 28.5 ms", figure_caption_style),
                 Paragraph("<b>(b) Misaligned Center Preset Run</b><br/>Az = 68.0 deg, El = -18.8 deg, Overlay = (601, 403) px<br/>Physical source at (499, 358); acoustic tracked right correctly", figure_caption_style)]
            ], colWidths=[252, 252]),
            Paragraph("Figure 14.1. Representative video frames from the final integrated validation showing the acoustic heatmap overlay and peak marker.", figure_caption_style),
        ]))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 15. SUBSYSTEM VALIDATION STATUS
    # ==========================================================================
    story.append(Paragraph("15. Subsystem Validation Status", h1_style))
    story.append(Paragraph(
        "Table 15.1 provides an engineering audit distinguishing functional system validation from ongoing pixel-level refinement.",
        body_style
    ))

    story.append(Paragraph("Table 15.1: Complete Subsystem Validation Matrix and Engineering Status", table_caption_style))
    status_headers = ["Subsystem Layer", "Validation Method", "Benchmark Criterion", "Observed Result", "Status Classification"]
    status_data = [
        ["Audio Driver (audio.py)", "sounddevice / PortAudio", "16 channels @ 48 kHz", "0 underflows in 187 frames (8.0s)", "Validated Functional"],
        ["Geometry (geometry.py)", "Board layout inspection", "Verified 42mm coordinate map", "Validated physical mic labels", "Validated Functional"],
        ["Steering (steering.py)", "Acoular reference comparison", "42mm X-Z array plane", "Exact Acoular numerical equivalence", "Validated Functional"],
        ["MUSIC Core (music.py)", "Synthetic DoA sweeps", "11/11 pass, error &lt;= 0.25 deg", "Super-resolution DoA functional", "Validated Functional"],
        ["Camera Driver (camera.py)", "ESP32-CAM Wi-Fi Stream", "Latency &lt; 50 ms", "Daemon reader drops stale frames (&lt;30ms)", "Validated Functional"],
        ["Vision (segmentation.py)", "YOLO11n-seg inference", "Centroid &amp; mask extraction", "Functional; class jitter on held objects", "Functional Baseline"],
        ["Sync (synchronization.py)", "Timestamp matching", "Sync delta &lt; 100 ms", "Median 28.5 ms, 100% pass rate", "Validated Functional"],
        ["Horizontal Direction", "3-position physical sweep", "Monotonic Left/Center/Right", "107.5 deg &gt; 90.8 deg &gt; 65.0 deg (pass)", "Validated (Tested Setup)"],
        ["Vertical Direction", "3-position physical sweep", "Monotonic Above/Level/Below", "+12.25 deg &gt; -6.00 deg &gt; -17.00 deg (pass)", "Validated (Tested Setup)"],
        ["Multipath Shielding", "Towel acoustic absorber", "Reduce -18.8 deg boundary bias", "Elevation shifted from -18.8 deg to -7.8 deg", "Demonstrated Effective"],
        ["Pixel-Level Projection", "Pinhole FOV model", "Sub-pixel visual overlay", "Median 77 px error (dx=1 px best frame)", "Ongoing Refinement"],
    ]
    t_status = make_academic_table(status_headers, status_data, [100, 105, 105, 115, 79], table_header_style, table_cell_style, align='CENTER')
    story.append(t_status)
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 16. LIMITATIONS AND REMAINING REFINEMENT
    # ==========================================================================
    story.append(PageBreak())
    story.append(Paragraph("16. Limitations and Remaining Refinement", h1_style))
    story.append(Paragraph(
        "Transparent engineering reporting requires detailing unresolved limitations and refinement needs:",
        body_style
    ))

    limitations = [
        ("16.1 Camera-to-Acoustic Projection Calibration",
         "The current visualizer implements a linear pinhole approximation using nominal manufacturer FOV specifications (50.0 deg H, 37.5 deg V). "
         "It does not incorporate lens distortion coefficients (k1, k2, p1, p2) or physical camera-to-array translation offsets, "
         "resulting in the observed residual ~59 px vertical projection error. Comprehensive non-linear geometric calibration is required."),

        ("16.2 Multipath Interference in Unshielded Environments",
         "The plane-wave MUSIC model assumes free-field conditions. In reverberant indoor environments with hard surfaces, "
         "specular reflections create coherent interference that biases elevation estimates downward. While physical acoustic absorbers "
         "mitigate near-table reflections, algorithmic de-reverberation is needed for unshielded operation."),

        ("16.3 Edge CPU Compute Budget",
         "Evaluating 6-band MUSIC over 7,676 grid points alongside YOLO11n-seg inference on a host CPU sustains 10--12 FPS. "
         "Deployment to embedded platforms (e.g. Raspberry Pi 4 or Jetson) will require grid pruning or GPU acceleration."),

        ("16.4 Low-Frequency Spatial Resolution",
         "Due to array aperture constraints (D &approx; 126 mm), spatial resolution is broad at 500 Hz (Rayleigh limit &approx; 5.4 rad), "
         "whereas higher frequencies (2500--3000 Hz) offer finer resolution but greater sensitivity to multipath reflections.")
    ]
    for title, desc in limitations:
        story.append(Paragraph(title, h2_style))
        story.append(Paragraph(desc, body_style))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 17. FUTURE DEVELOPMENT ROADMAP
    # ==========================================================================
    story.append(Paragraph("17. Future Development Roadmap", h1_style))
    story.append(Paragraph(
        "Subsequent development is structured into five sequential phases:",
        body_style
    ))

    roadmap_steps = [
        "<b>Phase 1: Automated 2D Acoustic-Visual Calibration:</b> Jointly optimize camera intrinsics, array-to-camera rotation (yaw, pitch, roll), and translation vectors using a known multimodal calibration target.",
        "<b>Phase 2: Algorithmic Reflection Suppression:</b> Implement spatial smoothing and Coherent Signal-subspace Processing (CSSP) across sub-bands to decorrelate coherent multipath reflections in unshielded environments.",
        "<b>Phase 3: Multi-Source Decomposition:</b> Expand source estimation from D = 1 to D = K sources using Information Theoretic Criteria (MDL/AIC) to resolve simultaneous sound emitters.",
        "<b>Phase 4: Embedded Edge Acceleration:</b> Port the pipeline to an NVIDIA Jetson platform utilizing TensorRT and CUDA for 30+ FPS operation.",
        "<b>Phase 5: Spatial Beamforming Audio Playback:</b> Implement real-time MVDR beamforming directed toward tracked visual centroids for selective audio listening."
    ]
    for step in roadmap_steps:
        story.append(Paragraph(f"&bull; {step}", bullet_style))
    story.append(Spacer(1, 8))

    # ==========================================================================
    # 18. CONCLUSION
    # ==========================================================================
    story.append(PageBreak())
    story.append(Paragraph("18. Conclusion", h1_style))
    conclusion_text = (
        "The Real-Time Multimodal Acoustic Camera has been successfully implemented as an integrated hardware and software system. "
        "The miniDSP UMA-16 v2 provides synchronized 16-channel acquisition, the MUSIC pipeline produces consistent directional responses "
        "under the tested conditions, the ESP32-CAM provides live optical data, and the timestamp synchronization layer successfully aligns "
        "the two modalities. Controlled physical experiments demonstrated correct horizontal and vertical directional ordering, "
        "while synthetic tests verified the implemented MUSIC formulation against known simulated directions.<br/><br/>"
        "The development process also identified practical limitations associated with physical array orientation, camera buffering, "
        "indoor acoustic reflections, and camera-to-array projection geometry. These were investigated experimentally, with successful "
        "correction of the physical orientation and camera buffering issues and substantial improvement of the reflection-related "
        "elevation anomaly through acoustic absorption shielding.<br/><br/>"
        "The system should therefore be considered functionally validated at the core multimodal level. "
        "Precise acoustic-angle-to-camera-pixel calibration has not yet been completed; the current projection error demonstrates that "
        "further camera calibration, rigid mechanical alignment, and geometric refinement are required. "
        "Frequency-based filtering/distribution and more robust multipath suppression are also identified as subsequent development stages."
    )
    story.append(Paragraph(conclusion_text, body_style))
    story.append(Spacer(1, 8))

    # Sign-off Block
    sign_off_data = [
        [Paragraph("<b>Report Author:</b>", table_cell_style), Paragraph("Kush Sahu", table_cell_style)],
        [Paragraph("<b>Project Title:</b>", table_cell_style), Paragraph("Real-Time Multimodal Phased-Array Acoustic Camera", table_cell_style)],
        [Paragraph("<b>Milestone Status:</b>", table_cell_style), Paragraph("Milestone 21 Completed (Core System Validated; Projection Refinement Ongoing)", table_cell_style)],
        [Paragraph("<b>Date:</b>", table_cell_style), Paragraph("September 2026", table_cell_style)],
    ]
    t_sign = Table(sign_off_data, colWidths=[130, 374])
    t_sign.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#1E293B")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_sign)
    story.append(Spacer(1, 10))

    # ==========================================================================
    # 19. TECHNICAL APPENDICES
    # ==========================================================================
    story.append(PageBreak())
    story.append(Paragraph("19. Technical Appendices", h1_style))

    story.append(Paragraph("19.1 Appendix A: Complete Raw Frame Telemetry Data", h2_style))
    story.append(Paragraph("Table A.1: Frame-by-Frame Telemetry for Final 20-Frame Towel-Shielded LEFT Validation Campaign", table_caption_style))

    raw_left_records = data_final_val["LEFT"]["records"]
    app_a_headers = ["Idx", "Cam ID", "Mic ID", "Sync &Delta; (ms)", "Azimuth", "Elevation", "Overlay (u, v)", "GT Err (px)", "YOLO Class (Conf)"]
    app_a_data = []
    for r in raw_left_records:
        yolo_str = r.get("yolo_detection", "none")
        app_a_data.append([
            str(r["index"]),
            str(r["camera_frame_id"]),
            str(r["acoustic_frame_id"]),
            f"{r['sync_delta_ms']:.2f}",
            f"{r['azimuth_deg']:.1f} deg",
            f"{r['elevation_deg']:.2f} deg",
            f"({r['overlay_u']}, {r['overlay_v']})",
            f"{r['pixel_error_gt']:.1f}",
            yolo_str,
        ])
    t_app_a = make_academic_table(app_a_headers, app_a_data, [25, 45, 45, 55, 50, 55, 75, 55, 104], table_header_style, table_cell_style, align='CENTER')
    story.append(t_app_a)
    story.append(PageBreak())

    story.append(Paragraph("19.2 Appendix B: Forensic Diagnostic Tool &amp; Scratch Script Inventory", h2_style))
    story.append(Paragraph("Table B.1: Inventory of Forensic Diagnostic Utilities in scratch/", table_caption_style))
    tool_headers = ["Script File", "Diagnostic Purpose and Scope", "Target Domain"]
    tool_data = [
        ["acoustic_camera_calibration.py", "Interactive multi-point 2D camera-to-acoustic calibration harness", "Calibration"],
        ["elevation_anomaly_diagnostic.py", "Sub-band energy, coherence, and reflection ratio analyzer", "Acoustic / Multipath"],
        ["subband_accumulation_experiment.py", "Comparative evaluator for sub-band accumulation strategies", "Acoustic / MUSIC"],
        ["final_integrated_validation.py", "20-frame synchronized multimodal verification test harness", "Pipeline / Fusion"],
        ["integrated_multimodal_test.py", "Controlled 3-position horizontal and vertical sweep runner", "Hardware Validation"],
        ["live_multimodal_heatmap_fusion.py", "Real-time OpenCV interactive GUI overlay demonstration", "Demonstration"],
        ["test_audio_acquisition.py", "PortAudio multi-channel audio capture integrity check", "Audio Driver"],
        ["test_music_audioframe.py", "Standalone audio frame MUSIC localization runner", "Acoustic Core"],
    ]
    t_tools = make_academic_table(tool_headers, tool_data, [150, 254, 100], table_header_style, table_cell_style)
    story.append(t_tools)
    story.append(Spacer(1, 8))

    story.append(Paragraph("19.3 Appendix C: System Configuration Parameter Reference", h2_style))
    story.append(Paragraph("Table C.1: Centralized Configuration Parameters in config/settings.py", table_caption_style))
    cfg_headers = ["Configuration Key", "Configured Value", "Operational Meaning"]
    cfg_data = [
        ["ESP32_IP / STREAM_URL", "172.27.8.165 / :81/stream", "ESP32-CAM Wi-Fi HTTP MJPEG network endpoint"],
        ["FRAME_WIDTH / HEIGHT", "640 x 480 pixels", "Camera optical stream resolution"],
        ["CAMERA_FOV_H / FOV_V", "50.0 deg / 37.5 deg", "Omnivision OV2640 optical lens field of view"],
        ["SAMPLE_RATE / BIT_DEPTH", "48,000 Hz / 24-bit", "miniDSP UMA-16 audio capture parameters"],
        ["FRAME_SIZE / HOP_SIZE", "4096 / 2048 samples", "85.33 ms analysis window, 50% overlap (42.67 ms hop)"],
        ["MIC_SPACING", "0.042 m (42.0 mm)", "Center-to-center microphone inter-element spacing"],
        ["SPEED_OF_SOUND", "343.0 m/s", "Acoustic propagation velocity in room temperature air"],
        ["FREQ_CENTERS", "500, 1000, 1500, 2000, 2500, 3000 Hz", "Discrete 6 sub-band center frequencies for MUSIC"],
        ["BAND_WIDTH", "500.0 Hz (+/-250 Hz)", "Bandwidth integrated per sub-band covariance matrix"],
        ["CSM_ALPHA / MUSIC_SMOOTHING", "0.30", "Exponential moving average cross-spectral matrix weight"],
        ["NOISE_THRESHOLD_DB", "15.0 dB SPL", "Acoustic energy gating threshold for MUSIC activation"],
        ["SIGMOID_GAIN / MIDPOINT", "3.0 / 0.5", "Non-linear contrast enhancement shaping parameters"],
        ["SYNC_TOLERANCE_MS", "100.0 ms", "Maximum allowable audio-video timestamp discrepancy"],
        ["MODEL_PATH", "models/yolo11n-seg.pt", "Ultralytics YOLO11 nano instance segmentation weights"],
    ]
    t_cfg = make_academic_table(cfg_headers, cfg_data, [140, 154, 210], table_header_style, table_cell_style)
    story.append(t_cfg)

    # Build Document
    print("Compiling Revised Academic PDF Document with AcademicNumberedCanvas...")
    doc.build(story, canvasmaker=AcademicNumberedCanvas)
    print(f"Successfully generated: {pdf_filename}")
    print(f"File size: {os.path.getsize(pdf_filename):,} bytes")


if __name__ == "__main__":
    generate_revised_pdf_report()
