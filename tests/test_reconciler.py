from app.services.reconciler import reconcile


def test_reconcile_merges_overlapping_candidates():
    candidates = [
        {"y_top": 100, "y_bottom": 200, "source": "hough"},
        {"y_top": 110, "y_bottom": 210, "source": "density"},
    ]
    result = reconcile(candidates, threshold=0.5)
    assert len(result) == 1
    assert result[0]["y_top"] == 100
    assert result[0]["y_bottom"] == 210


def test_reconcile_keeps_disjoint_candidates_separate():
    candidates = [
        {"y_top": 100, "y_bottom": 200, "source": "hough"},
        {"y_top": 400, "y_bottom": 500, "source": "density"},
    ]
    result = reconcile(candidates, threshold=0.5)
    assert len(result) == 2


def test_reconcile_empty_input():
    assert reconcile([]) == []


def test_reconcile_output_sorted_by_y_top():
    candidates = [
        {"y_top": 500, "y_bottom": 600, "source": "hough"},
        {"y_top": 100, "y_bottom": 200, "source": "density"},
        {"y_top": 300, "y_bottom": 350, "source": "hough"},
    ]
    result = reconcile(candidates, threshold=0.5)
    tops = [r["y_top"] for r in result]
    assert tops == sorted(tops)
