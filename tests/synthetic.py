"""Generadores de imágenes sintéticas para tests."""
import cv2
import numpy as np


def blank_page(height: int = 800, width: int = 600) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def page_with_text_block(
    height: int = 800,
    width: int = 600,
    text_x_left: int = 120,
    text_x_right: int = 480,
    n_lines: int = 12,
) -> np.ndarray:
    """Página blanca con líneas de "texto" simuladas como rectángulos oscuros."""
    img = blank_page(height, width)
    line_height = (height - 80) // (n_lines + 1)
    for i in range(n_lines):
        y = 40 + (i + 1) * line_height
        cv2.rectangle(img, (text_x_left, y), (text_x_right, y + 10), 30, -1)
    return img


def add_bracket(
    img: np.ndarray,
    y_top: int,
    y_bottom: int,
    x: int,
    side: str = "right",
    thickness: int = 3,
    arm_len: int = 18,
) -> np.ndarray:
    """Dibuja un corchete `[` o `]` en (x, y_top..y_bottom)."""
    cv2.line(img, (x, y_top), (x, y_bottom), 0, thickness)
    if side == "right":
        cv2.line(img, (x, y_top), (x - arm_len, y_top), 0, thickness)
        cv2.line(img, (x, y_bottom), (x - arm_len, y_bottom), 0, thickness)
    else:
        cv2.line(img, (x, y_top), (x + arm_len, y_top), 0, thickness)
        cv2.line(img, (x, y_bottom), (x + arm_len, y_bottom), 0, thickness)
    return img
