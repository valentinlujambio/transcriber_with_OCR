"""Deteccion del ROI de la pagina de interes (paso previo del pipeline).

Las fotos son tomadas con celular: incluyen escritorio, dedos, la otra
pagina del libro abierto, cuadernos, fondo, etc. Antes de binarizar y
buscar marcas conviene recortar la pagina de interes para eliminar ese
ruido externo.

Estrategia (pagina "mas grande / centrada"): se fusiona el texto impreso
en bloques solidos y se toma el bloque de mayor area (la columna de texto
mas prominente). Su bounding box se expande con un padding consciente de
vecinos: hacia cada lado se agranda para incluir el margen (donde van los
corchetes) pero sin invadir la otra pagina (se limita a la mitad del hueco
hasta el siguiente bloque de texto relevante).
"""
import cv2
import numpy as np

from app.core.config import config

Bbox = tuple[int, int, int, int]  # x0, y0, x1, y1


def _text_blocks(gray: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Devuelve bloques de texto impreso (x, y, w, h, area) ordenados por area."""
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        config.PAGE_TEXT_BLOCK, config.PAGE_TEXT_C,
    )
    # descartar "texto" sobre fondo que no es papel (fondo oscuro)
    _, paper = cv2.threshold(
        cv2.GaussianBlur(gray, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    th = cv2.bitwise_and(th, paper)

    # fusionar lineas/parrafos en bloques solidos: cierre vertical grande
    # (puentea huecos entre parrafos) + horizontal moderado (no salta de pagina)
    close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.PAGE_MERGE_W, config.PAGE_MERGE_H)
    )
    merged = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close)
    opening = cv2.getStructuringElement(cv2.MORPH_RECT, (config.PAGE_OPEN, config.PAGE_OPEN))
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, opening)

    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    boxes = [
        (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
         int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
         int(stats[i, cv2.CC_STAT_AREA]))
        for i in range(1, n)
    ]
    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes


def detect_page_bbox(bgr: np.ndarray) -> Bbox | None:
    """Devuelve el bbox (x0, y0, x1, y1) de la pagina de interes en coords
    de la imagen original, o None si no se detecta texto."""
    H, W = bgr.shape[:2]
    scale = config.PAGE_DETECT_MAX_DIM / max(H, W)
    scale = min(scale, 1.0)
    small = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1.0 else bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape

    boxes = _text_blocks(gray)
    if not boxes:
        return None

    x, y, w, h, area = boxes[0]
    neighbors = [b for b in boxes[1:] if b[4] >= area * config.PAGE_MIN_NEIGHBOR]

    def y_overlap(b: tuple[int, int, int, int, int]) -> bool:
        return not (b[1] + b[3] < y or b[1] > y + h)

    left_gap, right_gap = x, sw - (x + w)
    for b in neighbors:
        if not y_overlap(b):
            continue
        bx, bw = b[0], b[2]
        if bx + bw <= x:
            left_gap = min(left_gap, x - (bx + bw))
        elif bx >= x + w:
            right_gap = min(right_gap, bx - (x + w))

    want_x = int(w * config.PAGE_PAD_X)
    pad_l = min(want_x, max(left_gap // 2, 0))
    pad_r = min(want_x, max(right_gap // 2, 0))
    pad_y = int(h * config.PAGE_PAD_Y)

    x0 = max(0, x - pad_l)
    y0 = max(0, y - pad_y)
    x1 = min(sw, x + w + pad_r)
    y1 = min(sh, y + h + pad_y)

    inv = 1.0 / scale
    return (int(x0 * inv), int(y0 * inv), int(x1 * inv), int(y1 * inv))


def crop_to_page(bgr: np.ndarray) -> np.ndarray:
    """Recorta la imagen al ROI de la pagina de interes. Si no se detecta,
    devuelve la imagen original."""
    bbox = detect_page_bbox(bgr)
    if bbox is None:
        return bgr
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return bgr
    return bgr[y0:y1, x0:x1]
