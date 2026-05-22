from app.services.margin_detector import detect_external_margin
from app.services.preprocessor import preprocess
from app.services.text_block import detect_text_block
from tests.synthetic import add_bracket, page_with_text_block


def test_detects_right_margin_when_bracket_on_right():
    img = page_with_text_block(width=600, text_x_left=120, text_x_right=480)
    img = add_bracket(img, y_top=200, y_bottom=300, x=540, side="right")
    binary = preprocess(img)
    block = detect_text_block(binary)

    assert detect_external_margin(binary, block) == "right"


def test_detects_left_margin_when_bracket_on_left():
    img = page_with_text_block(width=600, text_x_left=120, text_x_right=480)
    img = add_bracket(img, y_top=200, y_bottom=300, x=60, side="left")
    binary = preprocess(img)
    block = detect_text_block(binary)

    assert detect_external_margin(binary, block) == "left"
