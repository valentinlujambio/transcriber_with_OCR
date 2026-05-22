import numpy as np


def detect_text_block(binary: np.ndarray, smoothing: int = 25, threshold_ratio: float = 0.15) -> dict:
    """Detecta los límites X del bloque principal de texto.

    Usa proyección vertical (suma por columna) de píxeles oscuros (255 en binaria
    invertida). Suaviza la señal y selecciona la zona contigua de máxima densidad.
    """
    if binary.ndim != 2:
        raise ValueError("Se espera imagen binaria 2D")

    column_density = binary.sum(axis=0).astype(np.float64) / 255.0

    if smoothing > 1:
        kernel = np.ones(smoothing, dtype=np.float64) / smoothing
        column_density = np.convolve(column_density, kernel, mode="same")

    if column_density.max() == 0:
        return {"x_left": 0, "x_right": binary.shape[1] - 1}

    threshold = column_density.max() * threshold_ratio
    above = np.where(column_density >= threshold)[0]

    if above.size == 0:
        return {"x_left": 0, "x_right": binary.shape[1] - 1}

    x_left = int(above[0])
    x_right = int(above[-1])
    return {"x_left": x_left, "x_right": x_right}
