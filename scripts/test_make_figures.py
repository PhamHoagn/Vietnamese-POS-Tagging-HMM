"""Kiểm thử biểu đồ từ bảng toy, không đọc corpus thật."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from scripts.make_figures import make_figures
from src.data_loader import Sentence
from src.eda import export_eda


def test_figures_from_csv(tmp_path: Path) -> None:
    """CSV toy tạo ba PNG 200 dpi, giữ nguyên đầu vào và đóng figure."""
    splits = {s: [Sentence(None, ["x", "."], ["NOUN", "PUNCT"])]
              for s in ("train", "dev", "test")}
    tables = tmp_path / "tables"
    export_eda(splits, tables)
    before = {p.name: p.read_bytes() for p in tables.iterdir()}
    with (tables / "sentence_lengths.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"split": s, "length": "2", "count": "1"} for s in splits]
    paths = make_figures(tables, tmp_path / "figures")
    assert len(paths) == 3
    for path in paths:
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.info["dpi"] == pytest.approx((200, 200), abs=0.1)
    assert not plt.get_fignums()
    assert before == {p.name: p.read_bytes() for p in tables.iterdir()}


def test_missing_length_csv_fails(tmp_path: Path) -> None:
    """Thiếu phân bố độ dài phải báo lỗi thay vì ước lượng từ thống kê tổng hợp."""
    with pytest.raises(FileNotFoundError):
        make_figures(tmp_path, tmp_path / "figures")
    assert not (tmp_path / "figures").exists()
