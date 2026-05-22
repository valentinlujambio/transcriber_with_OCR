import numpy as np


def detect_external_margin(binary: np.ndarray, text_block: dict) -> str:
    """Determina el margen externo comparando densidad fuera del bloque de texto.

    El "margen externo" es el lado con mayor cantidad de marcas/píxeles oscuros
    fuera del bloque principal (donde el usuario marca con corchetes).
    """
    if binary.ndim != 2:
        raise ValueError("Se espera imagen binaria 2D")

    x_left = text_block["x_left"]
    x_right = text_block["x_right"]

    left_strip = binary[:, :x_left] if x_left > 0 else np.zeros((binary.shape[0], 0), dtype=binary.dtype)
    right_strip = binary[:, x_right + 1:] if x_right + 1 < binary.shape[1] else np.zeros((binary.shape[0], 0), dtype=binary.dtype)

    left_density = float(left_strip.sum()) / max(left_strip.size, 1)
    right_density = float(right_strip.sum()) / max(right_strip.size, 1)

    return "left" if left_density > right_density else "right"
