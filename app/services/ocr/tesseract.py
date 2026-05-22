import numpy as np
import pytesseract

from app.services.ocr.base import OCRStrategy


class TesseractOCR(OCRStrategy):
    def __init__(self, lang: str = "spa+eng", config: str = "--psm 6") -> None:
        self._lang = lang
        self._config = config
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def extract_text(self, image_region: np.ndarray) -> str:
        text = pytesseract.image_to_string(image_region, lang=self._lang, config=self._config)
        return text.strip()