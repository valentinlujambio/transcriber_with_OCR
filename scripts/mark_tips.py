"""Experimento aislado: seguir un renglon como cadena de componentes conectados.

Un renglon es una sucesion de componentes (letras) pegados horizontalmente. En
vez de morfologia (que pega renglones vecinos) o traceo por pixeles (que se va
de linea), aca se ENCADENAN los componentes: desde la letra pegada a la marca
se salta al vecino (izquierda/derecha) que se solapa verticalmente con el
componente actual. Como cada salto se compara con el vecino inmediato, la cadena
sigue sola la inclinacion/curva del renglon.

Toma `debug/4. marks.png`, ubica la marca verde, encadena el renglon superior e
inferior que abarca, y marca la primera y la ultima letra de cada uno.

Uso:  python scripts/mark_tips.py [ruta_png]
Salida: debug/tips.png  (+ prints)
"""
import sys

import cv2
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "debug/4. marks.png"
OUT = "debug/tips.png"


def main() -> None:
    img = cv2.imread(SRC)
    if img is None:
        raise SystemExit(f"No pude abrir {SRC}")
    H, W = img.shape[:2]

    # 1) Marca = pixeles verdes (lime).
    b, g, r = (c.astype(int) for c in cv2.split(img))
    green = ((g > 120) & (g - r > 60) & (g - b > 60)).astype(np.uint8) * 255
    ys, xs = np.nonzero(green)
    if xs.size == 0:
        raise SystemExit("No encontre la marca verde en la imagen")
    mx0, mx1, my0, my1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    side = "right" if (mx0 + mx1) / 2 > W / 2 else "left"  # marca derecha => texto a la izquierda
    print(f"marca: x=[{mx0},{mx1}] y=[{my0},{my1}] lado={side}")

    # 2) Texto binarizado, sin la franja verde.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 12)
    th[green > 0] = 0

    # 3) Componentes conectados = letras (descarto ruido y manchas grandes).
    n, _, stats, cent = cv2.connectedComponentsWithStats(th, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a < 4 or h > 0.06 * H or w > 0.4 * W:
            continue
        comps.append({"id": i, "x": x, "y": y, "w": w, "h": h,
                      "cx": float(cent[i][0]), "cy": float(cent[i][1])})
    if not comps:
        raise SystemExit("No encontre componentes de texto")
    hmed = float(np.median([c["h"] for c in comps]))
    max_gap = 1.6 * hmed                  # espacio (entre BORDES) que puede saltar
    inset = 0.7 * hmed                    # margen interno de la marca (descarta filas espurias en las puntas)
    print(f"componentes: {len(comps)}  alto_mediana={hmed:.0f}  max_gap={max_gap:.0f}")

    def voverlap(a, c) -> bool:
        top = max(a["y"], c["y"])
        bot = min(a["y"] + a["h"], c["y"] + c["h"])
        return (bot - top) > 0.30 * min(a["h"], c["h"])

    def neighbor(cur, direction, used):
        """Componente que continua el renglon hacia `direction` (-1 izq, +1 der).

        Cercania horizontal por ESPACIO entre bordes (salta palabras anchas sin
        problema). Para que NO trepe a la linea de arriba/abajo cuando la
        inclinacion hace que se solapen: se prohibe todo salto vertical mayor a
        ~0.8 de alto de letra y, entre los candidatos, se elige el de menor costo
        = hueco + penalizacion por desvio vertical."""
        best, best_cost = None, 1e9
        for c in comps:
            if c["id"] in used:
                continue
            if direction < 0:
                if c["cx"] >= cur["cx"]:
                    continue
                gap = cur["x"] - (c["x"] + c["w"])   # hueco entre borde der de c y borde izq de cur
            else:
                if c["cx"] <= cur["cx"]:
                    continue
                gap = c["x"] - (cur["x"] + cur["w"])
            dvert = abs(c["cy"] - cur["cy"])
            if gap > max_gap or gap < -0.5 * hmed or dvert > 0.8 * hmed or not voverlap(cur, c):
                continue
            cost = max(gap, 0.0) + 2.0 * dvert       # preferir el que sigue la misma baseline
            if cost < best_cost:
                best, best_cost = c, cost
        return best

    def chain(seed):
        used = {seed["id"]}
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
        return line

    # 4) Semillas: caracteres pegados a la marca, ESTRICTAMENTE dentro del alto
    #    de la marca (sin expandir, para no colar la linea de arriba/abajo). Se
    #    agrupan por fila y se toma la fila de mas arriba y la de mas abajo; la
    #    semilla de cada una es el caracter mas cercano a la marca.
    if side == "right":
        near = [c for c in comps if mx0 - 0.15 * W < c["cx"] < mx0 and my0 + inset <= c["cy"] <= my1 - inset]
    else:
        near = [c for c in comps if mx1 < c["cx"] < mx1 + 0.15 * W and my0 + inset <= c["cy"] <= my1 - inset]
    if not near:
        raise SystemExit("No encontre letras pegadas a la marca")

    near.sort(key=lambda c: c["cy"])
    rows = [[near[0]]]
    for c in near[1:]:
        if c["cy"] - rows[-1][-1]["cy"] > 0.6 * hmed:   # salto de fila
            rows.append([c])
        else:
            rows[-1].append(c)
    print(f"filas pegadas a la marca: {len(rows)}")

    def nearest_to_mark(row):
        return max(row, key=lambda c: c["cx"]) if side == "right" else min(row, key=lambda c: c["cx"])

    seeds = {"superior": nearest_to_mark(rows[0]), "inferior": nearest_to_mark(rows[-1])}

    # 5) Encadenar y marcar puntas.
    vis = img.copy()
    for nombre, seed in seeds.items():
        line = chain(seed)
        for c in line:                     # toda la cadena en verde fino
            cv2.rectangle(vis, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), (0, 180, 0), 1)
        for c in (line[0], line[-1]):      # puntas en rojo grueso
            cv2.rectangle(vis, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), (0, 0, 255), 2)
        print(f"  {nombre}: {len(line)} componentes  primera_x={line[0]['x']}  ultima_x={line[-1]['x'] + line[-1]['w']}")

    cv2.imwrite(OUT, vis)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
