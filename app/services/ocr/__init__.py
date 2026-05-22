from app.core.config import config
from app.services.ocr.base import OCRStrategy


def get_ocr_provider() -> OCRStrategy:
    provider = config.OCR_PROVIDER

    if provider == "tesseract":
        from app.services.ocr.tesseract import TesseractOCR
        return TesseractOCR()

    if provider == "google_vision":
        from app.services.ocr.google_vision import GoogleVisionOCR
        return GoogleVisionOCR()

    raise ValueError(f"OCR provider desconocido: {provider}")
