import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OCR_PROVIDER: str = os.getenv("OCR_PROVIDER", "tesseract").lower()
    # Tesseract: idioma y config. `spa` es clave para los acentos; `--psm 6`
    # asume un bloque de texto uniforme (el parrafo marcado). Ajustable por env
    # (p.ej. OCR_TESS_CONFIG="--oem 1 --psm 6" para probar el motor LSTM).
    OCR_LANG: str = os.getenv("OCR_LANG", "spa+eng")
    OCR_TESS_CONFIG: str = os.getenv("OCR_TESS_CONFIG", "--psm 6")
    RECONCILIATION_THRESHOLD: float = float(os.getenv("RECONCILIATION_THRESHOLD", "0.5"))
    HOUGH_THRESHOLD: int = int(os.getenv("HOUGH_THRESHOLD", "80"))
    DENSITY_MIN_PIXELS: int = int(os.getenv("DENSITY_MIN_PIXELS", "30"))
    GOOGLE_APPLICATION_CREDENTIALS: str | None = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # --- Deteccion del ROI de la pagina de interes (paso previo) ---
    PAGE_DETECT_MAX_DIM: int = int(os.getenv("PAGE_DETECT_MAX_DIM", "1000"))
    PAGE_TEXT_BLOCK: int = int(os.getenv("PAGE_TEXT_BLOCK", "25"))
    PAGE_TEXT_C: int = int(os.getenv("PAGE_TEXT_C", "12"))
    PAGE_MERGE_H: int = int(os.getenv("PAGE_MERGE_H", "55"))   # cierre vertical (alto)
    PAGE_MERGE_W: int = int(os.getenv("PAGE_MERGE_W", "35"))   # cierre horizontal (ancho)
    PAGE_OPEN: int = int(os.getenv("PAGE_OPEN", "12"))
    PAGE_PAD_X: float = float(os.getenv("PAGE_PAD_X", "0.10"))
    PAGE_PAD_Y: float = float(os.getenv("PAGE_PAD_Y", "0.06"))
    PAGE_MIN_NEIGHBOR: float = float(os.getenv("PAGE_MIN_NEIGHBOR", "0.04"))
    # Lomo del libro
    PAGE_SPINE_BAND: float = float(os.getenv("PAGE_SPINE_BAND", "0.15"))     # banda vertical ignorada (arriba/abajo)
    PAGE_SPINE_FLANK: float = float(os.getenv("PAGE_SPINE_FLANK", "0.05"))   # ancho de los flancos a comparar
    PAGE_SPINE_FLANK_MIN: float = float(os.getenv("PAGE_SPINE_FLANK_MIN", "110"))  # brillo minimo de pagina en los flancos
    PAGE_SPINE_MIN_VALLEY: float = float(os.getenv("PAGE_SPINE_MIN_VALLEY", "12"))  # profundidad minima del valle oscuro
    PAGE_SPINE_MARGIN: float = float(os.getenv("PAGE_SPINE_MARGIN", "0.012"))  # offset hacia adentro del lomo

    PREPROC_SCALE: float = float(os.getenv("PREPROC_SCALE", "2.0"))
    CLAHE_CLIP: float = float(os.getenv("CLAHE_CLIP", "1.5"))
    # Tamano de tile de CLAHE en pixeles (constante)
    CLAHE_TILE_PX: int = int(os.getenv("CLAHE_TILE_PX", "300"))
    ADAPTIVE_BLOCK: int = int(os.getenv("ADAPTIVE_BLOCK", "41"))
    ADAPTIVE_C: int = int(os.getenv("ADAPTIVE_C", "15"))
    # Mascara de papel: descarta zonas oscuras (no-papel) que entran al ROI.
    PAPER_BLUR_SIGMA: float = float(os.getenv("PAPER_BLUR_SIGMA", "15.0"))

    # Por ahora solo corchetes
    DETECT_UNDERLINES: bool = os.getenv("DETECT_UNDERLINES", "0") == "1"
    # Subrayado: trazo horizontal largo, fino, con texto justo encima.
    UNDERLINE_MIN_WIDTH_RATIO: float = float(os.getenv("UNDERLINE_MIN_WIDTH_RATIO", "0.05"))
    UNDERLINE_MIN_ASPECT: float = float(os.getenv("UNDERLINE_MIN_ASPECT", "10.0"))
    UNDERLINE_TEXT_ABOVE_MIN: float = float(os.getenv("UNDERLINE_TEXT_ABOVE_MIN", "0.06"))
    # Corchete: trazo vertical alto (escala de parrafo), fino, con brazos
    # hacia un lado y el margen externo (opuesto) practicamente vacio.
    BRACKET_MIN_HEIGHT_RATIO: float = float(os.getenv("BRACKET_MIN_HEIGHT_RATIO", "0.035"))
    BRACKET_MIN_ASPECT: float = float(os.getenv("BRACKET_MIN_ASPECT", "12.0"))
    BRACKET_ARM_DENSITY_MIN: float = float(os.getenv("BRACKET_ARM_DENSITY_MIN", "0.15"))
    BRACKET_OUTER_DENSITY_MAX: float = float(os.getenv("BRACKET_OUTER_DENSITY_MAX", "0.05"))


config = Config()
