from app.core.config import config


def _overlap_ratio(a: dict, b: dict) -> float:
    """Calcula intersección / unión en el eje Y entre dos candidatos."""
    inter_top = max(a["y_top"], b["y_top"])
    inter_bottom = min(a["y_bottom"], b["y_bottom"])
    inter = max(0, inter_bottom - inter_top)

    union_top = min(a["y_top"], b["y_top"])
    union_bottom = max(a["y_bottom"], b["y_bottom"])
    union = max(1, union_bottom - union_top)

    return inter / union


def reconcile(candidates: list[dict], threshold: float | None = None) -> list[dict]:
    """Fusiona candidatos que se solapan más de `threshold` en su unión (IoU).

    Devuelve una lista ordenada por y_top con `{y_top, y_bottom}`.
    """
    if threshold is None:
        threshold = config.RECONCILIATION_THRESHOLD

    if not candidates:
        return []

    sorted_cands = sorted(candidates, key=lambda c: c["y_top"])
    merged: list[dict] = []

    for cand in sorted_cands:
        if not merged:
            merged.append({"y_top": cand["y_top"], "y_bottom": cand["y_bottom"]})
            continue

        last = merged[-1]
        if _overlap_ratio(last, cand) >= threshold:
            last["y_top"] = min(last["y_top"], cand["y_top"])
            last["y_bottom"] = max(last["y_bottom"], cand["y_bottom"])
        else:
            merged.append({"y_top": cand["y_top"], "y_bottom": cand["y_bottom"]})

    return merged
