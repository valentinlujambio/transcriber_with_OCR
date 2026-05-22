import io

import cv2
import numpy as np

from app.services.ocr.base import OCRStrategy


class GoogleVisionOCR(OCRStrategy):
    def __init__(self) -> None:
        from google.cloud import vision

        self._client = vision.ImageAnnotatorClient()
        self._vision = vision

    def extract_text(self, image_region: np.ndarray) -> str:
        success, encoded = cv2.imencode(".png", image_region)
        if not success:
            raise RuntimeError("No se pudo codificar la región para Google Vision")

        image = self._vision.Image(content=encoded.tobytes())
        response = self._client.document_text_detection(image=image)

        if response.error.message:
            raise RuntimeError(f"Google Vision error: {response.error.message}")

        return (response.full_text_annotation.text or "").strip()
