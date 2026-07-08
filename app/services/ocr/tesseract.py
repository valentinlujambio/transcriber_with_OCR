import numpy as np
import pytesseract

from app.core.config import config as app_config
from app.services.ocr.base import OCRStrategy


class TesseractOCR(OCRStrategy):
    def __init__(self, lang: str | None = None, config: str | None = None) -> None:
        self._lang = lang or app_config.OCR_LANG
        self._config = config or app_config.OCR_TESS_CONFIG
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def extract_text(self, image_region: np.ndarray) -> str:
        text = pytesseract.image_to_string(image_region, lang=self._lang, config=self._config)
        return text.strip()