"""Deteccion de marcas manuales sobre la pagina (paso 2 del pipeline).

Sobre la imagen binaria (texto/marcas = 255) busca dos tipos de marca
hechas a mano con lapiz, distinguiendolas del texto impreso por su
geometria:

- Subrayado: trazo horizontal largo y fino, con texto justo encima. El
  texto impreso no forma corridas horizontales continuas tan largas.
- Corchete `[` / `]`: trazo vertical alto (escala de parrafo) y fino,
  dibujado en el margen, con brazos cortos hacia un solo lado y el lado
  externo (el margen) practicamente vacio.

Cada deteccion es un dict con su bounding box `{x, y, w, h, type, ...}`.
"""
import cv2
import numpy as np

from app.core.config import config


def _density(region: np.ndarray) -> float:
    return float((region > 0).mean()) if region.size else 0.0


def detect_underlines(binary: np.ndarray) -> list[dict]:
    """Detecta subrayados como trazos horizontales largos con texto encima."""
    h, w = binary.shape
    hlen = max(int(w * config.UNDERLINE_MIN_WIDTH_RATIO), 25)
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1))

    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hkernel)
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, hkernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(horiz, 8)
    marks: list[dict] = []
    for i in range(1, n):
        x, y, ww, hh, _ = stats[i]
        if ww < hlen or hh > ww * 0.2 or ww / max(hh, 1) < config.UNDERLINE_MIN_ASPECT:
            continue

        gap = max(int(hh * 2), 6)
        above = binary[max(0, y - gap):y, x:x + ww]
        below = binary[y + hh:min(h, y + hh + gap), x:x + ww]
        # un subrayado tiene texto encima y bastante menos debajo
        if _density(above) < config.UNDERLINE_TEXT_ABOVE_MIN:
            continue
        if _density(above) <= _density(below):
            continue

        marks.append({"x": int(x), "y": int(y), "w": int(ww), "h": int(hh), "type": "underline"})
    return marks


def detect_brackets(binary: np.ndarray) -> list[dict]:
    """Detecta corchetes `[` / `]` como trazos verticales altos con brazos."""
    h, w = binary.shape
    vlen = max(int(h * 0.02), 40)
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen))

    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vkernel)
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, vkernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(vert, 8)
    min_h = int(h * config.BRACKET_MIN_HEIGHT_RATIO)
    arm_min = config.BRACKET_ARM_DENSITY_MIN

    marks: list[dict] = []
    for i in range(1, n):
        x, y, ww, hh, _ = stats[i]
        if hh < min_h or ww > hh * 0.2 or hh / max(ww, 1) < config.BRACKET_MIN_ASPECT:
            continue

        arm = max(int(hh * 0.12), 18)
        band = max(int(hh * 0.06), 12)

        def win(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
            return binary[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]

        # brazos a derecha (=> '[') o a izquierda (=> ']') en los extremos
        right_top = _density(win(y - band, y + band, x + ww, x + ww + arm)) > arm_min
        right_bot = _density(win(y + hh - band, y + hh + band, x + ww, x + ww + arm)) > arm_min
        left_top = _density(win(y - band, y + band, x - arm, x)) > arm_min
        left_bot = _density(win(y + hh - band, y + hh + band, x - arm, x)) > arm_min

        right_arms = int(right_top) + int(right_bot)
        left_arms = int(left_top) + int(left_bot)
        if right_arms == left_arms:  # sin brazos, o ambiguo (linea con texto a los dos lados)
            continue

        toward_right = right_arms > left_arms
        # el lado externo (margen, opuesto a los brazos) debe estar ~vacio
        if toward_right:
            outer = win(y, y + hh, x - arm, x)
            side = "left_bracket"   # '['
        else:
            outer = win(y, y + hh, x + ww, x + ww + arm)
            side = "right_bracket"  # ']'
        if _density(outer) > config.BRACKET_OUTER_DENSITY_MAX:
            continue

        marks.append({"x": int(x), "y": int(y), "w": int(ww), "h": int(hh), "type": "bracket", "side": side})
    return marks


def detect_marks(binary: np.ndarray, detect_underlines_too: bool | None = None) -> dict[str, list[dict]]:
    """Detecta corchetes y, opcionalmente, subrayados sobre la imagen binaria.

    La deteccion de subrayados es mas ruidosa; por ahora viene desactivada
    (`config.DETECT_UNDERLINES`). Pasar `detect_underlines_too=True` la fuerza.
    """
    if binary.ndim != 2:
        raise ValueError("Se espera imagen binaria 2D")
    if detect_underlines_too is None:
        detect_underlines_too = config.DETECT_UNDERLINES
    underlines = detect_underlines(binary) if detect_underlines_too else []
    return {"underlines": underlines, "brackets": detect_brackets(binary)}


def roi_from_bracket(
    shape: tuple[int, int],
    bracket: dict,
    pad_y: float = 0.08,
    pad_x: float = 1.5,
) -> tuple[int, int, int, int]:
    """Extiende un corchete hacia el lado de sus brazos para abarcar la columna
    de texto que marca, devolviendo el bounding box `(x, y, w, h)` del ROI.

    Un `[` (brazos a la derecha) marca texto a su derecha: el ROI va desde el
    corchete hasta el borde derecho de la pagina. Un `]` es el caso espejado.
    El alto del corchete define el alto del ROI.

    El corchete esta hecho a mano y, por la perspectiva de la foto, suele
    invadir el comienzo de los renglones y cortar las primeras letras. Para no
    perderlas el ROI arranca varios anchos de trazo *por fuera* del corchete
    (`pad_x`) y se le da un margen vertical generoso (`pad_y`), de modo que la
    region marcada quede holgada aunque la marca este corrida.
    """
    h, w = shape[:2]
    bx, by, bw, bh = bracket["x"], bracket["y"], bracket["w"], bracket["h"]

    pad_v = int(bh * pad_y)
    y0 = max(0, by - pad_v)
    y1 = min(h, by + bh + pad_v)

    pad_h = max(int(bw * pad_x), 12)
    if bracket.get("side") == "left_bracket":   # '[' -> texto a la derecha
        x0 = max(0, bx - pad_h)
        x1 = w
    else:                                        # ']' -> texto a la izquierda
        x0 = 0
        x1 = min(w, bx + bw + pad_h)

    return (x0, y0, x1 - x0, y1 - y0)


def _bracket_chain_rows(binary: np.ndarray, bracket: dict) -> list[list[dict]] | None:
    """Encadena los renglones que abarca el corchete y los devuelve como filas
    (de arriba a abajo), cada una con sus componentes ordenados por x.

    Un renglon es una sucesion de componentes conectados (letras/palabras). Por
    cada fila que el corchete abarca se toma como semilla el caracter pegado a
    la marca y se encadena hacia ambos lados saltando al vecino que se solapa
    verticalmente (hueco entre bordes acotado y desvio vertical penalizado, para
    no treparse a la linea de arriba cuando el texto va inclinado). Devuelve
    None si no hay componentes o semillas.
    """
    H, W = binary.shape
    bx, by, bw, bh = bracket["x"], bracket["y"], bracket["w"], bracket["h"]

    # componentes = letras/palabras; excluyo el trazo del corchete y ruido.
    # OJO: en la binaria las palabras pueden fusionarse en componentes largos;
    # solo se descartan los realmente anomalos (no cortar la cadena).
    n, _, stats, cent = cv2.connectedComponentsWithStats(binary, 8)
    comps: list[dict] = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a < 4 or h > 0.06 * H or w > 0.75 * W:
            continue
        if x < bx + bw and x + w > bx and y < by + bh and y + h > by:
            continue  # pisa el bbox del corchete (la marca misma)
        comps.append({"id": i, "x": x, "y": y, "w": w, "h": h,
                      "cx": float(cent[i][0]), "cy": float(cent[i][1])})
    if not comps:
        return None

    hmed = float(np.median([c["h"] for c in comps]))
    max_gap = 1.6 * hmed
    inset = 0.7 * hmed

    def voverlap(a: dict, c: dict) -> bool:
        top = max(a["y"], c["y"])
        bot = min(a["y"] + a["h"], c["y"] + c["h"])
        return (bot - top) > 0.30 * min(a["h"], c["h"])

    def neighbor(cur: dict, direction: int, used: set) -> dict | None:
        best, best_cost = None, 1e9
        for c in comps:
            if c["id"] in used:
                continue
            if direction < 0:
                if c["cx"] >= cur["cx"]:
                    continue
                gap = cur["x"] - (c["x"] + c["w"])
            else:
                if c["cx"] <= cur["cx"]:
                    continue
                gap = c["x"] - (cur["x"] + cur["w"])
            dvert = abs(c["cy"] - cur["cy"])
            if gap > max_gap or gap < -0.5 * hmed or dvert > 0.8 * hmed or not voverlap(cur, c):
                continue
            cost = max(gap, 0.0) + 2.0 * dvert
            if cost < best_cost:
                best, best_cost = c, cost
        return best

    # semillas: caracteres pegados a la marca, dentro del alto del corchete.
    # El inset es solo ARRIBA (el tope del trazo suele sobresalir por encima
    # del primer renglon); abajo se toma hasta el final del corchete, si no se
    # pierde el ultimo renglon.
    side_right = bracket.get("side") == "right_bracket"  # ']' -> texto a la izquierda
    if side_right:
        near = [c for c in comps if bx - 0.15 * W < c["cx"] < bx and by + inset <= c["cy"] <= by + bh]
    else:
        near = [c for c in comps if bx + bw < c["cx"] < bx + bw + 0.15 * W and by + inset <= c["cy"] <= by + bh]
    if not near:
        return None

    near.sort(key=lambda c: c["cy"])
    rows: list[list[dict]] = [[near[0]]]
    for c in near[1:]:
        if c["cy"] - rows[-1][-1]["cy"] > 0.6 * hmed:
            rows.append([c])
        else:
            rows[-1].append(c)

    # encadenar cada fila desde su caracter mas cercano a la marca
    used: set = set()
    chained_rows: list[list[dict]] = []
    for row in rows:
        seed = max(row, key=lambda c: c["cx"]) if side_right else min(row, key=lambda c: c["cx"])
        if seed["id"] in used:
            continue
        used.add(seed["id"])
        line = [seed]
        for direction in (-1, +1):
            cur = seed
            while True:
                nx = neighbor(cur, direction, used)
                if nx is None:
                    break
                used.add(nx["id"])
                line.append(nx)
                cur = nx
        line.sort(key=lambda c: c["cx"])
        chained_rows.append(line)

    chained_rows.sort(key=lambda ln: float(np.mean([c["cy"] for c in ln])))
    return chained_rows


def extract_marked_paragraph(
    gray: np.ndarray,
    binary: np.ndarray,
    bracket: dict,
    pad: int = 6,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Recorta el parrafo marcado por el corchete siguiendo sus renglones.

    El bbox cubre todas las cadenas hasta la punta de cada renglon. Como el
    parrafo va inclinado, un rectangulo derecho mete fragmentos de los renglones
    vecinos en las esquinas: se blanquea lo que queda por encima del primer
    renglon y por debajo del ultimo, con bordes que siguen la inclinacion de las
    cadenas (no se endereza nada, el texto queda curvo tal cual). Devuelve
    `(recorte, (x, y, w, h))`; si no hay cadenas cae al rectangulo simple.
    """
    rows = _bracket_chain_rows(binary, bracket)
    if not rows:
        x, y, w, h = roi_from_bracket(gray.shape, bracket)
        return gray[y:y + h, x:x + w], (x, y, w, h)

    comps = [c for row in rows for c in row]
    H, W = gray.shape[:2]
    x0 = max(0, min(c["x"] for c in comps) - pad)
    y0 = max(0, min(c["y"] for c in comps) - pad)
    x1 = min(W, max(c["x"] + c["w"] for c in comps) + pad)
    y1 = min(H, max(c["y"] + c["h"] for c in comps) + pad)

    crop = gray[y0:y1, x0:x1].copy()

    # bordes inclinados: techo = tope del primer renglon, piso = base del ultimo.
    # Se rellena con el tono del papel (NO blanco puro): un bloque grande de 255
    # desplaza el umbral global de Otsu de tesseract entre "blanco" y "papel" y
    # todo el texto queda como una mancha -> OCR vacio.
    paper = int(np.median(crop))
    xs = np.arange(x0, x1, dtype=np.float32)
    top_row, bot_row = rows[0], rows[-1]
    ytop = np.interp(xs, [c["cx"] for c in top_row], [c["y"] for c in top_row]) - pad
    ybot = np.interp(xs, [c["cx"] for c in bot_row], [c["y"] + c["h"] for c in bot_row]) + pad
    yy = np.arange(y0, y1, dtype=np.float32)[:, None]
    crop[(yy < ytop[None, :]) | (yy > ybot[None, :])] = paper

    return crop, (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def build_mask(shape: tuple[int, int], marks: list[dict]) -> np.ndarray:
    """Arma una mascara binaria (255) con los bounding boxes de las marcas."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for m in marks:
        x, y, w, h = m["x"], m["y"], m["w"], m["h"]
        mask[y:y + h, x:x + w] = 255
    return mask
