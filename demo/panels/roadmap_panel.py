"""
demo.panels.roadmap_panel
=========================
Mode 10: Current Empirical Limitations & Future Research Roadmap.
Honest assessment of current experimental boundaries alongside
7 strategic research extensions for an OUR proposal.
Upgraded with high-contrast, dual-stroke, non-aliased typography.
"""

from __future__ import annotations
from typing import Any, Dict
import cv2
import numpy as np

from demo.ui.theme import (
    FONT_BODY, FONT_HEADER, AA, BG_CARD, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ACCENT,
    ACCENT_AMBER, ACCENT_CYAN, ACCENT_GREEN
)
from demo.ui.widgets import draw_panel, draw_card, draw_section


class RoadmapPanel:
    """Renders Mode 10: Current Empirical Limitations & Future Research Roadmap."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def render(self, canvas: np.ndarray, telemetry: Dict[str, Any]) -> None:
        """Render the limitations and research extensions cards."""
        x, y, w, h = self.x, self.y, self.w, self.h
        draw_panel(canvas, x, y, w, h, title="MODE 10: CURRENT LIMITATIONS & FUTURE RESEARCH EXTENSIONS")

        mid_x = x + (w // 2) - 10
        left_w = mid_x - x - 20
        right_w = (x + w - 20) - mid_x - 10

        # ====================================================
        # LEFT: CURRENT EMPIRICAL LIMITATIONS & REFINEMENTS
        # ====================================================
        draw_card(canvas, x + 20, y + 42, left_w, h - 64)
        draw_section(canvas, "Current Empirical Limitations & Refinements", x + 36, y + 68, ACCENT_AMBER, thickness=2)
        cv2.putText(canvas, "Honest engineering characterization of current prototype boundaries:", (x + 36, y + 88), FONT_BODY, 0.42, TEXT_SECONDARY, 1, AA)

        limitations = [
            ("1. Projection Calibration", "Linear pinhole model ignores lens distortion (k1, k2) and rigid translation offsets."),
            ("2. Multi-Band Weighting", "All 6 sub-bands currently contribute equally; frequency-adaptive weighting is pending."),
            ("3. Tabletop Multipath", "Indoor reflections cause downward elevation bias without acoustic absorber shielding."),
            ("4. Real-World Angular Accuracy", "Monotonic physical ordering confirmed; formal benchmark across full FOV is ongoing."),
            ("5. YOLO Object Benchmark", "YOLO11n-seg functional for class/mask extraction; formal mAP evaluation unbenchmarked."),
            ("6. End-to-End Latency", "Queue timestamp delta verified (<100 ms); physical glass-to-glass latency requires optical pulse test."),
            ("7. Distance / Source Depth", "Current plane-wave MUSIC formulation is 2D angular; spherical depth is uncharacterized."),
        ]

        ly = y + 112
        for title, desc in limitations:
            cv2.putText(canvas, title, (x + 36, ly), FONT_HEADER, 0.46, TEXT_PRIMARY, 1, AA)
            cv2.putText(canvas, desc, (x + 36, ly + 20), FONT_BODY, 0.40, TEXT_SECONDARY, 1, AA)
            ly += 46

        # ====================================================
        # RIGHT: FUTURE RESEARCH DIRECTIONS (OUR PROPOSAL)
        # ====================================================
        draw_card(canvas, mid_x + 10, y + 42, right_w, h - 64)
        draw_section(canvas, "Future Research Roadmap (OUR Continuation)", mid_x + 26, y + 68, ACCENT_CYAN, thickness=2)
        cv2.putText(canvas, "Proposed research extensions for upcoming faculty grant / publication:", (mid_x + 26, y + 88), FONT_BODY, 0.42, TEXT_SECONDARY, 1, AA)

        extensions = [
            ("Phase 1: Automated 2D Calibration", "Joint non-linear optimization of camera intrinsics and 6-DOF acoustic array extrinsics."),
            ("Phase 2: Frequency-Adaptive MUSIC", "Dynamic sub-band SNR estimation and coherence weighting to suppress reverberant reflections."),
            ("Phase 3: Algorithmic Dereverberation", "Spatial smoothing and Coherent Signal-subspace Processing (CSSP) for unshielded operation."),
            ("Phase 4: Multi-Source Decomposition", "Information Theoretic Criteria (MDL/AIC) to resolve simultaneous independent sound emitters."),
            ("Phase 5: Object-Sound Association", "Probabilistic Bayesian fusion associating detected YOLO object masks directly with acoustic peaks."),
            ("Phase 6: Spatial Selective Listening", "Steered MVDR beamforming directed toward tracked visual centroids for selective audio playback."),
            ("Phase 7: Embedded Edge Acceleration", "Optimization and deployment to NVIDIA Jetson Orin via TensorRT for 30+ FPS edge execution."),
        ]

        ry = y + 112
        for title, desc in extensions:
            cv2.putText(canvas, title, (mid_x + 26, ry), FONT_HEADER, 0.46, ACCENT_GREEN, 1, AA)
            cv2.putText(canvas, desc, (mid_x + 26, ry + 20), FONT_BODY, 0.40, TEXT_SECONDARY, 1, AA)
            ry += 46

        # Bottom transition banner
        ban_y = y + h - 34
        cv2.putText(canvas, "Ready for OUR Proposal: Proven hardware & software pipeline provides a solid baseline for advanced acoustic sensing.", (x + 36, ban_y), FONT_HEADER, 0.46, TEXT_ACCENT, 2, AA)
