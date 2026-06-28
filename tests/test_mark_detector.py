import numpy as np

from app.services.mark_detector import build_mask, detect_brackets, detect_marks, detect_underlines
from app.services.preprocessor import preprocess
from tests.synthetic import add_bracket, add_underline, page_with_text_block


def test_detect_underlines_finds_underline_under_text():
    img = page_with_text_block(width=600, height=800, text_x_left=120, text_x_right=480)
    # subrayado justo debajo de una linea de texto
    img = add_underline(img, y=200, x_left=140, x_right=460, thickness=4)
    binary = preprocess(img, scale=1.0)

    unders = detect_underlines(binary)

    assert len(unders) >= 1
    assert any(abs(u["y"] - 200) < 20 and u["w"] > u["h"] * 5 for u in unders)


def test_detect_underlines_empty_on_plain_text():
    img = page_with_text_block(width=600, height=800)
    binary = preprocess(img, scale=1.0)

    # el texto impreso no debe generar subrayados
    assert detect_underlines(binary) == []


def test_detect_brackets_finds_left_bracket():
    img = page_with_text_block(width=600, height=800, text_x_left=120, text_x_right=480)
    # corchete '[' alto (escala de parrafo) en el margen izquierdo
    img = add_bracket(img, y_top=120, y_bottom=520, x=90, side="left", thickness=5, arm_len=60)
    binary = preprocess(img, scale=1.0)

    brackets = detect_brackets(binary)

    assert len(brackets) >= 1
    assert any(b["side"] == "left_bracket" for b in brackets)


def test_detect_marks_returns_both_groups():
    img = page_with_text_block(width=600, height=800)
    binary = preprocess(img, scale=1.0)

    marks = detect_marks(binary)

    assert set(marks.keys()) == {"underlines", "brackets"}
    assert isinstance(marks["underlines"], list)
    assert isinstance(marks["brackets"], list)


def test_detect_marks_rejects_non_2d():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    try:
        detect_marks(rgb)
        assert False, "deberia rechazar imagen no binaria"
    except ValueError:
        pass


def test_build_mask_marks_bounding_boxes():
    marks = [{"x": 5, "y": 10, "w": 20, "h": 4}]
    mask = build_mask((50, 50), marks)

    assert mask.shape == (50, 50)
    assert mask[12, 15] == 255
    assert mask[0, 0] == 0
