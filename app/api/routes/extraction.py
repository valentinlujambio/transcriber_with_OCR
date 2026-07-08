import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile

# Services
from app.services.mark_detector import detect_marks, extract_marked_paragraph
from app.services.ocr import get_ocr_provider
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
        # Paso 1: ROI de la página
        page = crop_to_page(bgr)
        debug_plotter.plot_image(cv2.cvtColor(page, cv2.COLOR_BGR2RGB), "1. page_roi")

        # Paso 2: mayor resolucion + CLAHE + binarizacion adaptativa
        gray = enhance(page)
        binary = binarize(gray) #1|||||||||||||||||||||899999999 - Comentario de Junior
        debug_plotter.plot_image(gray, "2. enhanced")
        debug_plotter.plot_image(binary, "3. binary")

        # Paso 3: deteccion de subrayados y corchetes
        marks = detect_marks(binary)
        underlines = marks["underlines"]
        brackets = marks["brackets"]
        logger.debug(f"Subrayados: {len(underlines)} | Corchetes: {len(brackets)}")

        def rects(items: list[dict]) -> list[tuple[int, int, int, int]]:
            return [(m["x"], m["y"], m["w"], m["h"]) for m in items]

        debug_plotter.plot_marks(
            gray,
            [(rects(underlines), "red"), (rects(brackets), "lime")],
            "4. marks",
        )

        # Paso 4: por cada corchete, seguir cada renglon marcado como cadena de
        # componentes conectados hasta sus puntas -> recorte ajustado al parrafo
        # (con lo de arriba/abajo blanqueado siguiendo la inclinacion). Se manda
        # el recorte en gris (curvo) al OCR tal cual.
        ocr = get_ocr_provider()
        roi_boxes: list[tuple[int, int, int, int]] = []
        results: list[dict] = []
        for idx, bracket in enumerate(brackets):
            roi, (x, y, w, h) = extract_marked_paragraph(gray, binary, bracket)
            roi_boxes.append((x, y, w, h))
            debug_plotter.plot_image(roi, f"5.{idx} roi")

            text = ocr.extract_text(roi)
            logger.debug(f"ROI {idx} ({bracket['side']}): {text!r}")
            results.append({"bracket": bracket, "roi": {"x": x, "y": y, "w": w, "h": h}, "text": text})

        debug_plotter.plot_marks(gray, [(roi_boxes, "cyan")], "5. rois")

        return {
            "underlines": underlines,
            "brackets": brackets,
            "results": results,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno del pipeline: {exc}") from exc
