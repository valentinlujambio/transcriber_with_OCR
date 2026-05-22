import numpy as np

from app.services.preprocessor import preprocess
from tests.synthetic import page_with_text_block


def test_preprocess_returns_binary_image():
    img = page_with_text_block()
    binary = preprocess(img)

    assert binary.ndim == 2
    assert binary.dtype == np.uint8
    unique = set(np.unique(binary).tolist())
    assert unique.issubset({0, 255})


def test_preprocess_text_pixels_become_foreground():
    img = page_with_text_block()
    binary = preprocess(img)
    # Hay píxeles activos en algún lugar (las líneas de texto).
    assert binary.sum() > 0
