"""Kiểm tra EDA bằng dữ liệu toy tính tay, không đọc corpus thật."""

import csv
from copy import deepcopy
from pathlib import Path

import pytest

from src.contracts import UPOS_TAGS
from src.data_loader import Sentence
from src.eda import export_eda


def test_eda_counts_and_distribution(tmp_path: Path) -> None:
    """Hai câu dài 3/1 cho mean=median=2; type phân biệt hoa thường và giữ dấu cách."""
    splits = {
        "train": [Sentence("a", ["sinh viên", "học", "."], ["NOUN", "VERB", "PUNCT"]),
                  Sentence("b", ["sinh viên"], ["NOUN"])],
        "dev": [Sentence(None, ["Học", "học"], ["VERB", "VERB"])],
        "test": [],
    }
    before = deepcopy(splits)
    paths = export_eda(splits, tmp_path / "tables")
    with paths["eda"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0] == {
        "split": "train", "sentence_count": "2", "token_count": "4",
        "sentence_length_min": "1", "sentence_length_mean": "2",
        "sentence_length_median": "2.0", "sentence_length_max": "3",
        "token_type_count": "3", "punct_token_count": "1",
    }
    assert rows[1]["token_type_count"] == "2"
    assert rows[2]["sentence_count"] == rows[2]["token_count"] == "0"
    assert rows[2]["sentence_length_mean"] == ""
    with paths["upos_distribution"].open(encoding="utf-8", newline="") as stream:
        distribution = list(csv.DictReader(stream))
    assert len(distribution) == 51
    for split, total in (("train", 4), ("dev", 2), ("test", 0)):
        group = [r for r in distribution if r["split"] == split]
        assert [r["upos"] for r in group] == list(UPOS_TAGS)
        assert sum(int(r["count"]) for r in group) == total
        if total:
            assert sum(float(r["percentage"]) for r in group) == pytest.approx(100)
        else:
            assert all(r["percentage"] == "" for r in group)
    noun = next(r for r in distribution if r["split"] == "train" and r["upos"] == "NOUN")
    assert noun["count"] == "2" and float(noun["percentage"]) == 50
    assert splits == before


@pytest.mark.parametrize("problem", ["missing_split", "invalid_tag", "unaligned"])
def test_invalid_input_writes_nothing(tmp_path: Path, problem: str) -> None:
    """Đầu vào lỗi phải bị từ chối trước khi xuất bất kỳ bảng nào."""
    sentence = Sentence(None, ["x"], ["NOUN"])
    splits = {"train": [], "dev": [], "test": [sentence]}
    if problem == "missing_split":
        del splits["test"]
    elif problem == "invalid_tag":
        sentence.tags[0] = "_"
    else:
        sentence.tags.clear()
    output = tmp_path / "tables"
    with pytest.raises(ValueError):
        export_eda(splits, output)
    assert not output.exists()
