import io

import cv2
import numpy as np
from PIL import Image

from app.core.config import config


def load_image(image_bytes: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance(image: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Escala de grises a mayor resolucion con contraste local realzado.

    Paso 1 del pipeline: re-escala la foto (las fotos de celular vienen
    comprimidas) y aplica CLAHE para compensar la luz despareja y la
    curvatura de la pagina antes de binarizar.
    """
    if scale is None:
        scale = config.PREPROC_SCALE

    gray = to_gray(image)
    if scale and scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Tiles de tamano constante en pixeles (no grilla fija): asi el realce es
    # estable en imagenes y ROIs de distinto tamano.
    tile = config.CLAHE_TILE_PX
    gh = max(1, round(gray.shape[0] / tile))
    gw = max(1, round(gray.shape[1] / tile))
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP, tileGridSize=(gw, gh))
    return clahe.apply(gray)


def _paper_mask(gray: np.ndarray) -> np.ndarray:
    """Mascara de las zonas de papel (claras). Las regiones oscuras que entran
    al ROI (cuaderno, dedos, sombra externa) generan ruido que CLAHE amplifica
    y produce falsos subrayados; se descartan quedandonos con el papel."""
    smooth = cv2.GaussianBlur(gray, (0, 0), config.PAPER_BLUR_SIGMA)
    _, paper = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Rellenar los huecos internos (texto/marcas oscuras encerradas por papel),
    # de cualquier tamano, sin reconectar las intrusiones oscuras externas: una
    # region oscura es "hueco" si NO toca el borde de la imagen. Asi el cuaderno
    # / dedos / sombra (que entran por un borde) quedan fuera, pero el texto no
    # se borra.
    h, w = paper.shape
    inv = cv2.bitwise_not(paper)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    out = paper.copy()
    for i in range(1, n):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches_border:
            out[labels == i] = 255
    return out


def binarize(gray: np.ndarray) -> np.ndarray:
    """Binarizacion adaptativa: texto/marcas = 255 sobre fondo 0."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    block = config.ADAPTIVE_BLOCK
    if block % 2 == 0:
        block += 1
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        config.ADAPTIVE_C,
    )
    return cv2.bitwise_and(binary, _paper_mask(gray))


def preprocess(image: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Devuelve la imagen binaria (texto/marcas = 255) realzada con CLAHE.

    El resultado esta en el espacio re-escalado por `scale` (default
    `config.PREPROC_SCALE`); las coordenadas de las etapas posteriores se
    refieren a esa resolucion.
    """
    return binarize(enhance(image, scale))
