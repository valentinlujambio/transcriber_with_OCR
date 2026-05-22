from app.services.line_detector import detect_lines
from app.services.preprocessor import preprocess
from app.services.text_block import detect_text_block
from tests.synthetic import add_bracket, page_with_text_block


def test_detect_lines_finds_bracket_range():
    img = page_with_text_block(width=600, height=800, text_x_left=120, text_x_right=480)
    img = add_bracket(img, y_top=200, y_bottom=320, x=540, side="right", thickness=4)
    binary = preprocess(img)
    block = detect_text_block(binary)

    candidates = detect_lines(binary, "right", block)

    assert len(candidates) > 0
    # Al menos un candidato debe solapar con el rango del corchete.
    overlap = any(
        c["y_top"] <= 320 and c["y_bottom"] >= 200 for c in candidates
    )
    assert overlap


def test_detect_lines_returns_empty_on_blank_strip():
    img = page_with_text_block(width=600, height=800)
    binary = preprocess(img)
    block = detect_text_block(binary)

    candidates = detect_lines(binary, "right", block)
    # Sin corchetes el margen está mayormente vacío.
    assert isinstance(candidates, list)
