from app.services.preprocessor import preprocess
from app.services.text_block import detect_text_block
from tests.synthetic import page_with_text_block


def test_detect_text_block_finds_expected_x_range():
    img = page_with_text_block(text_x_left=120, text_x_right=480)
    binary = preprocess(img)

    block = detect_text_block(binary)

    assert "x_left" in block and "x_right" in block
    assert block["x_left"] < block["x_right"]
    # Tolerancia generosa por el suavizado.
    assert abs(block["x_left"] - 120) < 60
    assert abs(block["x_right"] - 480) < 60
