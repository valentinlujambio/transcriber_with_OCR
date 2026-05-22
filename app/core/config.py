import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OCR_PROVIDER: str = os.getenv("OCR_PROVIDER", "tesseract").lower()
    RECONCILIATION_THRESHOLD: float = float(os.getenv("RECONCILIATION_THRESHOLD", "0.5"))
    HOUGH_THRESHOLD: int = int(os.getenv("HOUGH_THRESHOLD", "80"))
    DENSITY_MIN_PIXELS: int = int(os.getenv("DENSITY_MIN_PIXELS", "30"))
    GOOGLE_APPLICATION_CREDENTIALS: str | None = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


config = Config()
