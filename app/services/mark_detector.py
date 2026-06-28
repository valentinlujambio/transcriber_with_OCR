"""Deteccion de marcas manuales sobre la pagina (paso 2 del pipeline).

Sobre la imagen binaria (texto/marcas = 255) busca dos tipos de marca
hechas a mano con lapiz, distinguiendolas del texto impreso por su
geometria:

- Subrayado: trazo horizontal largo y fino, con texto justo encima. El
  texto impreso no forma corridas horizontales continuas tan largas.
- Corchete `[` / `]`: trazo vertical alto (escala de parrafo) y fino,
  dibujado en el margen, con brazos cortos hacia un solo lado y el lado
  externo (el margen) practicamente vacio.

Cada deteccion es un dict con su bounding box `{x, y, w, h, type, ...}`.
"""
import cv2
import numpy as np

from app.core.config import config


def _density(region: np.ndarray) -> float:
    return float((region > 0).mean()) if region.size else 0.0


def detect_underlines(binary: np.ndarray) -> list[dict]:
    """Detecta subrayados como trazos horizontales largos con texto encima."""
    h, w = binary.shape
    hlen = max(int(w * config.UNDERLINE_MIN_WIDTH_RATIO), 25)
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1))

    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hkernel)
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, hkernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(horiz, 8)
    marks: list[dict] = []
    for i in range(1, n):
        x, y, ww, hh, _ = stats[i]
        if ww < hlen or hh > ww * 0.2 or ww / max(hh, 1) < config.UNDERLINE_MIN_ASPECT:
            continue

        gap = max(int(hh * 2), 6)
        above = binary[max(0, y - gap):y, x:x + ww]
        below = binary[y + hh:min(h, y + hh + gap), x:x + ww]
        # un subrayado tiene texto encima y bastante menos debajo
        if _density(above) < config.UNDERLINE_TEXT_ABOVE_MIN:
            continue
        if _density(above) <= _density(below):
            continue

        marks.append({"x": int(x), "y": int(y), "w": int(ww), "h": int(hh), "type": "underline"})
    return marks


def detect_brackets(binary: np.ndarray) -> list[dict]:
    """Detecta corchetes `[` / `]` como trazos verticales altos con brazos."""
    h, w = binary.shape
    vlen = max(int(h * 0.02), 40)
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen))

    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vkernel)
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, vkernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(vert, 8)
    min_h = int(h * config.BRACKET_MIN_HEIGHT_RATIO)
    arm_min = config.BRACKET_ARM_DENSITY_MIN

    marks: list[dict] = []
    for i in range(1, n):
        x, y, ww, hh, _ = stats[i]
        if hh < min_h or ww > hh * 0.2 or hh / max(ww, 1) < config.BRACKET_MIN_ASPECT:
            continue

        arm = max(int(hh * 0.12), 18)
        band = max(int(hh * 0.06), 12)

        def win(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
            return binary[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]

        # brazos a derecha (=> '[') o a izquierda (=> ']') en los extremos
        right_top = _density(win(y - band, y + band, x + ww, x + ww + arm)) > arm_min
        right_bot = _density(win(y + hh - band, y + hh + band, x + ww, x + ww + arm)) > arm_min
        left_top = _density(win(y - band, y + band, x - arm, x)) > arm_min
        left_bot = _density(win(y + hh - band, y + hh + band, x - arm, x)) > arm_min

        right_arms = int(right_top) + int(right_bot)
        left_arms = int(left_top) + int(left_bot)
        if right_arms == left_arms:  # sin brazos, o ambiguo (linea con texto a los dos lados)
            continue

        toward_right = right_arms > left_arms
        # el lado externo (margen, opuesto a los brazos) debe estar ~vacio
        if toward_right:
            outer = win(y, y + hh, x - arm, x)
            side = "left_bracket"   # '['
        else:
            outer = win(y, y + hh, x + ww, x + ww + arm)
            side = "right_bracket"  # ']'
        if _density(outer) > config.BRACKET_OUTER_DENSITY_MAX:
            continue

        marks.append({"x": int(x), "y": int(y), "w": int(ww), "h": int(hh), "type": "bracket", "side": side})
    return marks


def detect_marks(binary: np.ndarray, detect_underlines_too: bool | None = None) -> dict[str, list[dict]]:
    """Detecta corchetes y, opcionalmente, subrayados sobre la imagen binaria.

    La deteccion de subrayados es mas ruidosa; por ahora viene desactivada
    (`config.DETECT_UNDERLINES`). Pasar `detect_underlines_too=True` la fuerza.
    """
    if binary.ndim != 2:
        raise ValueError("Se espera imagen binaria 2D")
    if detect_underlines_too is None:
        detect_underlines_too = config.DETECT_UNDERLINES
    underlines = detect_underlines(binary) if detect_underlines_too else []
    return {"underlines": underlines, "brackets": detect_brackets(binary)}


def build_mask(shape: tuple[int, int], marks: list[dict]) -> np.ndarray:
    """Arma una mascara binaria (255) con los bounding boxes de las marcas."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for m in marks:
        x, y, w, h = m["x"], m["y"], m["w"], m["h"]
        mask[y:y + h, x:x + w] = 255
    return mask
