"""
demo.ui.widgets
===============
Reusable visual components and typography rendering helpers for the Professor Demo GUI.
Engineered for crisp, anti-aliased, highly legible presentation at 1600x900.
"""

from __future__ import annotations
from typing import Tuple, Optional
import cv2
import numpy as np

from demo.ui.theme import (
    FONT, FONT_TITLE, FONT_HEADER, FONT_BODY, FONT_MONO, AA,
    SCALE_APP_TITLE, SCALE_MODE_TITLE, SCALE_PANEL_TITLE, SCALE_SECTION,
    SCALE_LABEL, SCALE_VALUE, SCALE_VALUE_BOLD, SCALE_TABLE_HEADER, SCALE_TABLE_VALUE, SCALE_SMALL,
    BG_CARD, BG_PANEL, BG_HOVER, BORDER, BG_VISUALIZATION,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ACCENT, TEXT_ON_DARK,
    ACCENT_PRIMARY, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER, ACCENT_UI,
    RADIUS_CARD, RADIUS_BUTTON, RADIUS_BADGE, SPACE_SM, SPACE_MD
)


def draw_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = SCALE_LABEL,
    color: Tuple[int, int, int] = TEXT_PRIMARY,
    thickness: int = 1,
    font: int = FONT_BODY,
) -> Tuple[int, int]:
    """Draw anti-aliased text with cv2.LINE_AA and return (width, height)."""
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness, AA)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    return tw, th


def draw_app_title(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = TEXT_PRIMARY,
) -> Tuple[int, int]:
    """Draw the primary top-level application title (large, thickness 2)."""
    return draw_text(canvas, text, x, y, scale=SCALE_APP_TITLE, color=color, thickness=2, font=FONT_TITLE)


def draw_mode_title(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = TEXT_ACCENT,
) -> Tuple[int, int]:
    """Draw a main mode title (prominent, thickness 2)."""
    return draw_text(canvas, text, x, y, scale=SCALE_MODE_TITLE, color=color, thickness=2, font=FONT_HEADER)


def draw_title(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = TEXT_ACCENT,
) -> Tuple[int, int]:
    """Draw a primary panel or mode title."""
    return draw_text(canvas, text, x, y, scale=SCALE_PANEL_TITLE, color=color, thickness=2, font=FONT_HEADER)


def draw_section(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = TEXT_ACCENT,
    thickness: int = 1,
) -> Tuple[int, int]:
    """Draw a card/section header."""
    return draw_text(canvas, text, x, y, scale=SCALE_SECTION, color=color, thickness=thickness, font=FONT_HEADER)


def draw_label(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = TEXT_SECONDARY,
) -> Tuple[int, int]:
    """Draw a field or telemetry label."""
    return draw_text(canvas, text, x, y, scale=SCALE_LABEL, color=color, thickness=1, font=FONT_BODY)


def draw_value(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = TEXT_PRIMARY,
    bold: bool = False,
) -> Tuple[int, int]:
    """Draw a telemetry readout or numeric value."""
    scale = SCALE_VALUE_BOLD if bold else SCALE_VALUE
    thickness = 2 if bold else 1
    return draw_text(canvas, text, x, y, scale=scale, color=color, thickness=thickness, font=FONT_BODY)


def draw_table_header(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = TEXT_SECONDARY,
) -> Tuple[int, int]:
    """Draw a table column header."""
    return draw_text(canvas, text, x, y, scale=SCALE_TABLE_HEADER, color=color, thickness=1, font=FONT_HEADER)


def draw_table_cell(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = TEXT_PRIMARY,
    bold: bool = False,
) -> Tuple[int, int]:
    """Draw a table cell value."""
    thickness = 2 if bold else 1
    return draw_text(canvas, text, x, y, scale=SCALE_TABLE_VALUE, color=color, thickness=thickness, font=FONT_BODY)


def draw_note(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = TEXT_MUTED,
) -> Tuple[int, int]:
    """Draw an explanatory note or footnote."""
    return draw_text(canvas, text, x, y, scale=SCALE_SMALL, color=color, thickness=1, font=FONT_BODY)


def draw_rounded_rect(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: Tuple[int, int, int],
    radius: int = RADIUS_CARD,
    thickness: int = -1,
) -> None:
    """Draw a clipped, anti-aliased rounded rectangle using OpenCV primitives."""
    if w <= 0 or h <= 0:
        return

    x2, y2 = x + w - 1, y + h - 1
    radius = max(0, min(int(radius), (min(w, h) - 1) // 2))
    if radius == 0:
        cv2.rectangle(canvas, (x, y), (x2, y2), color, thickness, AA)
        return

    if thickness < 0:
        cv2.rectangle(canvas, (x + radius, y), (x2 - radius, y2), color, -1)
        cv2.rectangle(canvas, (x, y + radius), (x2, y2 - radius), color, -1)
        for cx, cy in (
            (x + radius, y + radius),
            (x2 - radius, y + radius),
            (x + radius, y2 - radius),
            (x2 - radius, y2 - radius),
        ):
            cv2.circle(canvas, (cx, cy), radius, color, -1, AA)
        return

    cv2.line(canvas, (x + radius, y), (x2 - radius, y), color, thickness, AA)
    cv2.line(canvas, (x + radius, y2), (x2 - radius, y2), color, thickness, AA)
    cv2.line(canvas, (x, y + radius), (x, y2 - radius), color, thickness, AA)
    cv2.line(canvas, (x2, y + radius), (x2, y2 - radius), color, thickness, AA)
    for center, start, end in (
        ((x + radius, y + radius), 180, 270),
        ((x2 - radius, y + radius), 270, 360),
        ((x2 - radius, y2 - radius), 0, 90),
        ((x + radius, y2 - radius), 90, 180),
    ):
        cv2.ellipse(canvas, center, (radius, radius), 0, start, end, color, thickness, AA)


def rounded_rect_contains(px: int, py: int, x: int, y: int, w: int, h: int, radius: int) -> bool:
    """Return whether a point lies inside the visible rounded rectangle."""
    if w <= 0 or h <= 0 or px < x or py < y or px >= x + w or py >= y + h:
        return False
    radius = max(0, min(int(radius), (min(w, h) - 1) // 2))
    if radius == 0:
        return True
    nearest_x = min(max(px, x + radius), x + w - radius - 1)
    nearest_y = min(max(py, y + radius), y + h - radius - 1)
    return (px - nearest_x) ** 2 + (py - nearest_y) ** 2 <= radius ** 2


def draw_button(
    canvas: np.ndarray,
    label: str,
    x: int,
    y: int,
    w: int,
    h: int,
    active: bool = False,
    hover: bool = False,
    disabled: bool = False,
    radius: int = RADIUS_BUTTON,
    scale: float = 0.46,
    border_color: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw a rounded button with consistent active, inactive, and disabled states."""
    if w <= 0 or h <= 0:
        return

    if disabled:
        bg, text_color, edge = BG_PANEL, TEXT_MUTED, BORDER
    elif active:
        bg, text_color, edge = ACCENT_PRIMARY, TEXT_ON_DARK, border_color or ACCENT_PRIMARY
    elif hover:
        bg, text_color, edge = BG_HOVER, TEXT_PRIMARY, border_color or ACCENT_UI
    else:
        bg, text_color, edge = BG_PANEL, TEXT_PRIMARY, border_color or BORDER

    draw_rounded_rect(canvas, x, y, w, h, bg, radius)
    draw_rounded_rect(canvas, x, y, w, h, edge, radius, 1)
    if w < 20 or h < 14:
        return
    (text_w, text_h), _ = cv2.getTextSize(label, FONT_BODY, scale, 1)
    max_text_w = max(1, w - 2 * SPACE_SM)
    fit_scale = min(scale, scale * max_text_w / max(1, text_w))
    (text_w, text_h), _ = cv2.getTextSize(label, FONT_BODY, fit_scale, 1)
    tx = x + max(0, (w - text_w) // 2)
    ty = y + (h + text_h) // 2 - 1
    cv2.putText(canvas, label, (tx, ty), FONT_BODY, fit_scale, text_color, 1, AA)


def draw_status_chip(
    canvas: np.ndarray,
    x: int,
    y: int,
    text: str,
    tone: str = "neutral",
    height: int = 24,
) -> int:
    """Draw a compact rounded status chip and return its width."""
    if height <= 0:
        return 0
    tones = {
        "neutral": (BG_PANEL, BORDER, TEXT_SECONDARY),
        "success": (BG_CARD, ACCENT_GREEN, ACCENT_GREEN),
        "warning": (BG_CARD, ACCENT_AMBER, ACCENT_AMBER),
        "error": (BG_CARD, ACCENT_RED, ACCENT_RED),
        "accent": (BG_CARD, ACCENT_UI, ACCENT_UI),
    }
    bg, edge, text_color = tones.get(tone, tones["neutral"])
    (text_w, _), _ = cv2.getTextSize(text, FONT_BODY, 0.40, 1)
    width = text_w + 2 * SPACE_SM
    draw_rounded_rect(canvas, x, y, width, height, bg, RADIUS_BADGE)
    draw_rounded_rect(canvas, x, y, width, height, edge, RADIUS_BADGE, 1)
    if height < 14:
        return width
    (_, text_h), _ = cv2.getTextSize(text, FONT_BODY, 0.40, 1)
    cv2.putText(canvas, text, (x + SPACE_SM, y + (height + text_h) // 2 - 1), FONT_BODY, 0.40, text_color, 1, AA)
    return width


def draw_divider(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int] = BORDER,
) -> None:
    """Draw a subtle anti-aliased divider between related content."""
    cv2.line(canvas, (x1, y1), (x2, y2), color, 1, AA)


def _draw_fitted_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    max_width: int,
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw text at a reduced scale when needed to fit its available width."""
    if max_width <= 0:
        return
    (text_w, _), _ = cv2.getTextSize(text, FONT_BODY, scale, thickness)
    fit_scale = scale * min(1.0, max_width / max(1, text_w))
    if fit_scale <= 0:
        return
    cv2.putText(canvas, text, (x, y), FONT_BODY, fit_scale, color, thickness, AA)


def draw_measurement_card(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    value: str,
    value_color: Tuple[int, int, int] = TEXT_PRIMARY,
) -> None:
    """Draw a compact rounded card for one scannable measurement."""
    if w <= 0 or h <= 0:
        return
    draw_card(canvas, x, y, w, h)
    if w < 64 or h < 36:
        return
    text_x = x + SPACE_SM
    text_width = w - 2 * SPACE_SM
    _draw_fitted_text(canvas, label, text_x, y + 20, text_width, SCALE_LABEL, TEXT_SECONDARY)
    value_y = y + h - SPACE_MD
    if value == "—":
        marker_width = min(22, text_width)
        marker_y = value_y - max(3, int(SCALE_VALUE_BOLD * 10))
        cv2.line(canvas, (text_x, marker_y), (text_x + marker_width, marker_y), value_color, 2, AA)
    else:
        _draw_fitted_text(
            canvas,
            value,
            text_x,
            value_y,
            text_width,
            SCALE_VALUE_BOLD,
            value_color,
            thickness=2,
        )


def draw_panel(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    title: Optional[str] = None,
    bg: Tuple[int, int, int] = BG_PANEL,
    border_color: Tuple[int, int, int] = BORDER,
) -> None:
    """Draw a styled panel container with an optional top title bar."""
    if w <= 0 or h <= 0:
        return
    draw_rounded_rect(canvas, x, y, w, h, bg, RADIUS_CARD)
    draw_rounded_rect(canvas, x, y, w, h, border_color, RADIUS_CARD, 1)
    if title and w >= 80 and h >= 28:
        title_h = min(34, h)
        draw_rounded_rect(canvas, x + 1, y + 1, max(1, w - 2), title_h, BG_CARD, RADIUS_CARD)
        cv2.rectangle(canvas, (x + 1, y + min(RADIUS_CARD, title_h)), (x + w - 2, y + title_h), BG_CARD, -1)
        draw_divider(canvas, x + 1, y + title_h, x + w - 2, y + title_h, border_color)
        cv2.putText(canvas, title, (x + SPACE_MD, y + 23), FONT_HEADER, SCALE_MODE_TITLE, TEXT_ACCENT, 2, AA)


def draw_card(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    bg: Tuple[int, int, int] = BG_CARD,
    border: Tuple[int, int, int] = BORDER,
) -> None:
    """Draw an elevated content card."""
    if w <= 0 or h <= 0:
        return
    draw_rounded_rect(canvas, x, y, w, h, bg, RADIUS_CARD)
    draw_rounded_rect(canvas, x, y, w, h, border, RADIUS_CARD, 1)


def draw_badge(
    canvas: np.ndarray,
    x: int,
    y: int,
    text: str,
    status_ok: bool = True,
    warning: bool = False,
) -> int:
    """Draw a bold pill status badge and return the total badge width."""
    if status_ok:
        color = ACCENT_GREEN
        prefix = "OK: "
    elif warning:
        color = ACCENT_AMBER
        prefix = "WARN: "
    else:
        color = ACCENT_RED
        prefix = "OFF: "

    label = prefix + text
    (tw, th), _ = cv2.getTextSize(label, FONT_BODY, 0.46, 1)
    pw, ph = tw + 20, 24
    draw_rounded_rect(canvas, x, y, pw, ph, BG_CARD, RADIUS_BADGE)
    draw_rounded_rect(canvas, x, y, pw, ph, color, RADIUS_BADGE, 1)
    cv2.putText(canvas, label, (x + 10, y + 17), FONT_BODY, 0.46, TEXT_PRIMARY, 1, AA)
    return pw


def draw_metric_row(
    canvas: np.ndarray,
    x: int,
    y: int,
    label: str,
    value: str,
    value_color: Tuple[int, int, int] = TEXT_PRIMARY,
    width: int = 500,
    bold_value: bool = False,
) -> None:
    """Draw a structured key-value telemetry line with aligned dotted leader."""
    val_scale = SCALE_VALUE_BOLD if bold_value else SCALE_VALUE
    val_thick = 2 if bold_value else 1

    cv2.putText(canvas, label, (x, y), FONT_BODY, SCALE_LABEL, TEXT_SECONDARY, 1, AA)
    (lw, _), _ = cv2.getTextSize(label, FONT_BODY, SCALE_LABEL, 1)
    (vw, _), _ = cv2.getTextSize(value, FONT_BODY, val_scale, val_thick)

    # Draw dotted leader if space permits
    dot_start = x + lw + 12
    dot_end = x + width - vw - 10
    if dot_end > dot_start + 16:
        for dx in range(dot_start, dot_end, 8):
            cv2.circle(canvas, (dx, y - 4), 1, BORDER, -1)

    cv2.putText(canvas, value, (x + width - vw, y), FONT_BODY, val_scale, value_color, val_thick, AA)


def draw_bar(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    fraction: float,
    color: Tuple[int, int, int] = ACCENT_UI,
    bg: Tuple[int, int, int] = BG_VISUALIZATION,
) -> None:
    """Draw a clean horizontal meter bar."""
    if w <= 0 or h <= 0:
        return
    frac = float(np.clip(fraction, 0.0, 1.0))
    draw_rounded_rect(canvas, x, y, w, h, bg, min(h // 2, RADIUS_BADGE))
    fill_w = int(w * frac)
    if fill_w > 0:
        draw_rounded_rect(canvas, x, y, fill_w, h, color, min(h // 2, RADIUS_BADGE))
    draw_rounded_rect(canvas, x, y, w, h, BORDER, min(h // 2, RADIUS_BADGE), 1)


class Button:
    """Interactive clickable rounded button with crisp typography."""

    def __init__(
        self,
        key: str,
        label: str,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Tuple[int, int, int] = ACCENT_UI,
    ) -> None:
        self.key = key
        self.label = label
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.active = False
        self.hover = False

    def contains(self, px: int, py: int) -> bool:
        return rounded_rect_contains(px, py, self.x, self.y, self.w, self.h, RADIUS_BUTTON)

    def draw(self, canvas: np.ndarray) -> None:
        draw_button(
            canvas,
            self.label,
            self.x,
            self.y,
            self.w,
            self.h,
            active=self.active,
            hover=self.hover,
            border_color=self.color if (self.active or self.hover) else BORDER,
        )
