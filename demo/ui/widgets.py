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
    BG_CARD, BG_PANEL, BORDER, BORDER_ACTIVE,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_ACCENT,
    ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER, ACCENT_CYAN
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
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1)
    if title:
        cv2.rectangle(canvas, (x, y), (x + w, y + 32), BG_CARD, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + 32), border_color, 1)
        cv2.putText(canvas, title, (x + 14, y + 22), FONT_HEADER, SCALE_MODE_TITLE, TEXT_ACCENT, 2, AA)


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
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1)


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
    cv2.rectangle(canvas, (x, y), (x + pw, y + ph), (28, 22, 18), -1)
    cv2.rectangle(canvas, (x, y), (x + pw, y + ph), color, 1)
    cv2.putText(canvas, label, (x + 10, y + 17), FONT_BODY, 0.46, color, 1, AA)
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
            cv2.circle(canvas, (dx, y - 4), 1, (80, 68, 56), -1)

    cv2.putText(canvas, value, (x + width - vw, y), FONT_BODY, val_scale, value_color, val_thick, AA)


def draw_bar(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    fraction: float,
    color: Tuple[int, int, int] = ACCENT_CYAN,
    bg: Tuple[int, int, int] = (30, 24, 20),
) -> None:
    """Draw a clean horizontal meter bar."""
    frac = float(np.clip(fraction, 0.0, 1.0))
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    fill_w = int(w * frac)
    if fill_w > 0:
        cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER, 1)


class Button:
    """Interactive clickable rectangular button with crisp typography."""

    def __init__(
        self,
        key: str,
        label: str,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Tuple[int, int, int] = ACCENT_CYAN,
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
        return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)

    def draw(self, canvas: np.ndarray) -> None:
        bg = BORDER_ACTIVE if self.active else ((65, 52, 42) if self.hover else BG_CARD)
        border = self.color if (self.active or self.hover) else BORDER
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1)
        cv2.rectangle(
            canvas,
            (self.x, self.y),
            (self.x + self.w, self.y + self.h),
            border,
            2 if self.active else 1,
        )
        (tw, th), _ = cv2.getTextSize(self.label, FONT_BODY, 0.46, 1)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2 - 1
        txt_col = TEXT_PRIMARY if not self.active else (255, 255, 255)
        cv2.putText(canvas, self.label, (tx, ty), FONT_BODY, 0.46, txt_col, 1, AA)
