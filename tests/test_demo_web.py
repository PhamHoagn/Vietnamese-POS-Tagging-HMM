"""Regression checks for the demo's two entry modes and contextual predictions."""

import pytest

from scripts.demo_web import Demo


@pytest.fixture(scope="module")
def demo():
    return Demo()


@pytest.mark.parametrize(("text", "tag"), [
    ("Tôi thấy bàn đẹp.", "NOUN"),
    ("Tôi thấy cái bàn rất đẹp.", "NOUN"),
    ("Chúng tôi đang bàn về dự án.", "VERB"),
])
def test_ambiguous_word_in_context(demo, text, tag):
    words = demo.segmenter.split(text)
    assert demo.contextual.predict(words)[words.index("bàn")] == tag


def test_word_boundaries_and_manual_override(demo):
    words = demo.segmenter.split("Tôi đến Hà Nội, lúc 8.30!")
    assert words == ["Tôi", "đến", "Hà Nội", ",", "lúc", "8.30", "!"]
    status, page = demo.render(None, "Hà | Nội | .", "manual", "best")
    assert status == 200
    assert page.count(b'<th scope="row">') == 3
    assert "Hà | Nội | ." in page.decode()


def test_raw_and_manual_predictions_agree_when_boundaries_match(demo):
    raw = "Tôi thấy cái bàn rất đẹp."
    manual = "Tôi | thấy | cái | bàn | rất | đẹp | ."
    left = demo.render(raw, None, "auto", "best")[1].split(b"<tbody>")[1]
    right = demo.render(None, manual, "manual", "best")[1].split(b"<tbody>")[1]
    assert left == right


def test_gold_requires_matching_all_tokens(demo):
    for manual, expected in [("Tôi | học | .", True), ("Tôi | thích | .", False)]:
        status, page = demo.render(None, manual, "manual", "best")
        assert status == 200
        assert ("Đúng theo nhãn chuẩn" in page.decode()) == expected


@pytest.mark.parametrize(("raw", "manual", "input_mode"), [
    ("", None, "auto"), (None, "một || hai", "manual"),
    ("a " * 101, None, "auto"), (None, "x" * 101, "manual"),
])
def test_invalid_input_offers_correction(demo, raw, manual, input_mode):
    status, page = demo.render(raw, manual, input_mode, "best")
    assert status == 400
    assert "Nhập từ 1 đến 100 token" in page.decode()
