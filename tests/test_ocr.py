import numpy as np
import pytest

from app.services.ocr.base import OCRStrategy


class _DummyOCR(OCRStrategy):
    def extract_text(self, image_region: np.ndarray) -> str:
        return f"len={image_region.size}"


def test_ocrstrategy_is_abstract():
    with pytest.raises(TypeError):
        OCRStrategy()  # type: ignore[abstract]


def test_dummy_subclass_works():
    ocr = _DummyOCR()
    region = np.zeros((10, 10), dtype=np.uint8)
    assert ocr.extract_text(region) == "len=100"


def test_factory_unknown_provider_raises(monkeypatch):
    from app.core import config as config_module
    from app.services import ocr as ocr_module

    monkeypatch.setattr(config_module.config, "OCR_PROVIDER", "no_existe")
    with pytest.raises(ValueError):
        ocr_module.get_ocr_provider()
