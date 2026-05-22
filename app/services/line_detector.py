import cv2
import numpy as np

from app.core.config import config


def _margin_strip(binary: np.ndarray, side: str, text_block: dict | None = None) -> tuple[np.ndarray, int, int]:
    """Recorta la franja vertical correspondiente al margen externo.

    Devuelve (strip, x_offset_inicio, x_offset_fin). Si no hay text_block,
    usa un 15% del ancho de la imagen.
    """
    h, w = binary.shape

    if text_block is not None:
        if side == "left":
            x0, x1 = 0, max(text_block["x_left"], 1)
        else:
            x0, x1 = min(text_block["x_right"] + 1, w - 1), w
    else:
        band = max(int(w * 0.15), 1)
        if side == "left":
            x0, x1 = 0, band
        else:
            x0, x1 = w - band, w

    if x1 <= x0:
        x1 = x0 + 1

    return binary[:, x0:x1], x0, x1


def _detect_hough(strip: np.ndarray) -> list[dict]:
    """Detecta líneas verticales (lados largos de los corchetes) con Hough probabilística."""
    if strip.size == 0:
        return []

    h = strip.shape[0]
    min_len = max(int(h * 0.015), 8)
    max_gap = max(int(h * 0.01), 3)

    lines = cv2.HoughLinesP(
        strip,
        rho=1,
        theta=np.pi / 180,
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )

    if lines is None:
        return []

    candidates: list[dict] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > abs(y2 - y1):
            continue
        y_top = int(min(y1, y2))
        y_bottom = int(max(y1, y2))
        if y_bottom - y_top < min_len:
            continue
        candidates.append({"y_top": y_top, "y_bottom": y_bottom, "source": "hough"})

    return candidates


def _detect_density(strip: np.ndarray) -> list[dict]:
    """Detecta rangos Y con alta densidad de píxeles oscuros sostenida."""
    if strip.size == 0:
        return []

    row_density = (strip > 0).sum(axis=1)

    min_pixels = config.DENSITY_MIN_PIXELS
    active = row_density >= min_pixels

    candidates: list[dict] = []
    in_run = False
    y_start = 0

    for y, is_active in enumerate(active):
        if is_active and not in_run:
            in_run = True
            y_start = y
        elif not is_active and in_run:
            in_run = False
            if y - y_start >= 5:
                candidates.append({"y_top": y_start, "y_bottom": y - 1, "source": "density"})

    if in_run:
        candidates.append({"y_top": y_start, "y_bottom": len(active) - 1, "source": "density"})

    return candidates


def detect_lines(binary: np.ndarray, side: str, text_block: dict | None = None) -> list[dict]:
    """Devuelve candidatos de líneas (rangos Y) detectados en el margen externo.

    Combina Hough (geometría) + densidad (acumulación de píxeles).
    """
    if side not in ("left", "right"):
        raise ValueError("side debe ser 'left' o 'right'")

    strip, _, _ = _margin_strip(binary, side, text_block)
    return _detect_hough(strip) + _detect_density(strip)
