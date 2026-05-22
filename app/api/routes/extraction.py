import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.line_detector import detect_lines
from app.services.margin_detector import detect_external_margin
from app.services.ocr import get_ocr_provider
from app.services.preprocessor import load_image, preprocess
from app.services.reconciler import reconcile
from app.services.text_block import detect_text_block
from app.services.matplot_utils import plot_image

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
        binary = preprocess(bgr)
        # plot_image(binary, "1. binary")
        text_block = detect_text_block(binary)
        # plot_image(binary, "2. text_block")
        side = detect_external_margin(binary, text_block)
        # plot_image(binary, "3. side")
        candidates = detect_lines(binary, side, text_block)
        # plot_image(binary, "4. candidates")
        fragments_y = reconcile(candidates)
        # plot_image(binary, "5. fragments")
        ocr = get_ocr_provider()
        # plot_image(binary, "6. ocr")
        gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # plot_image(gray, "7. gray")
        
        x_left = text_block["x_left"]
        x_right = text_block["x_right"]

        fragments: list[dict] = []
        for frag in fragments_y:
            y_top = max(0, frag["y_top"])
            y_bottom = min(gray.shape[0] - 1, frag["y_bottom"])
            if y_bottom <= y_top:
                continue
            region = gray[y_top:y_bottom + 1, x_left:x_right + 1]
            if region.size == 0:
                continue
            text = ocr.extract_text(region)
            fragments.append({"y_top": y_top, "y_bottom": y_bottom, "text": text})

        return {"fragments": fragments}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno del pipeline: {exc}") from exc
