import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile

# Services
from app.services.mark_detector import detect_marks
from app.services.page_detector import crop_to_page
from app.services.preprocessor import binarize, enhance, load_image

# Utils
from app.services.logger import logger
from app.services.matplot_utils import debug_plotter

router = APIRouter()


@router.post("/extract")
async def extract(image: UploadFile = File(...)) -> dict:
    try:
        raw = await image.read()
        if not raw:
            raise HTTPException(status_code=422, detail="Imagen vacía")
        bgr = load_image(raw)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Imagen inválida: {exc}") from exc

    try:
        # Paso previo: recortar el ROI de la pagina de interes (saca escritorio,
        # dedos, la otra pagina, fondo, etc.)
        page = crop_to_page(bgr)
        debug_plotter.plot_image(cv2.cvtColor(page, cv2.COLOR_BGR2RGB), "0. page_roi")

        # Paso 1: mayor resolucion + CLAHE + binarizacion adaptativa
        gray = enhance(page)
        binary = binarize(gray)
        debug_plotter.plot_image(gray, "1. enhanced")
        debug_plotter.plot_image(binary, "2. binary")

        # Paso 2: deteccion de subrayados y corchetes
        marks = detect_marks(binary)
        underlines = marks["underlines"]
        brackets = marks["brackets"]
        logger.debug(f"Subrayados: {len(underlines)} | Corchetes: {len(brackets)}")

        def rects(items: list[dict]) -> list[tuple[int, int, int, int]]:
            return [(m["x"], m["y"], m["w"], m["h"]) for m in items]

        debug_plotter.plot_marks(
            gray,
            [(rects(underlines), "red"), (rects(brackets), "lime")],
            "3. marks",
        )

        return {
            "underlines": underlines,
            "brackets": brackets,
        }
        # TODO (paso 3-5): mascaras -> ROI -> mejora de binarizacion -> OCR
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno del pipeline: {exc}") from exc
