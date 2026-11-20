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


def test_oov_uses_original_train_frequencies(tmp_path: Path) -> None:
    """OOV có f=0; rare đã thấy có f=1/2; lặp từ ở dev không thay tần suất train."""
    splits = {
        "train": [Sentence(None, ["a", "b", "b", "c", "c", "c"], ["NOUN"] * 6)],
        "dev": [Sentence(None, ["a", "b", "c", "x", "x"],
                         ["NOUN", "VERB", "VERB", "NOUN", "NOUN"])],
        "test": [Sentence(None, ["x", "a", "A"], ["NOUN", "NOUN", "PROPN"])],
    }
    original = deepcopy(splits)
    paths = export_eda(splits, tmp_path)
    with paths["oov_stats"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 36
    assert [r["upos"] for r in rows[:18]] == ["ALL", *UPOS_TAGS]
    indexed = {(r["split"], r["upos"]): r for r in rows}
    overall = indexed["dev", "ALL"]
    for metric, count in (("oov", 2), ("f_train_le_1", 3), ("f_train_le_2", 4),
                          ("seen_rare_le_1", 1), ("seen_rare_le_2", 2)):
        assert int(overall[f"{metric}_count"]) == count
        assert float(overall[f"{metric}_rate"]) == pytest.approx(count / 5)
    assert float(indexed["dev", "NOUN"]["oov_rate"]) == pytest.approx(2 / 3)
    assert float(indexed["dev", "VERB"]["oov_rate"]) == 0
    assert indexed["dev", "ADJ"]["oov_rate"] == ""
    assert int(indexed["test", "ALL"]["oov_count"]) == 2
    assert splits == original
    for row in rows:
        for cutoff in (1, 2):
            assert int(row[f"f_train_le_{cutoff}_count"]) == (
                int(row["oov_count"]) + int(row[f"seen_rare_le_{cutoff}_count"])
            )


def test_oov_empty_train_and_empty_split(tmp_path: Path) -> None:
    """Train rỗng khiến mọi token OOV; nhóm không có token có rate không xác định."""
    paths = export_eda({"train": [], "dev": [],
                        "test": [Sentence(None, ["x"], ["NOUN"])]}, tmp_path)
    with paths["oov_stats"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["oov_rate"] == "" for row in rows if row["split"] == "dev")
    test = next(row for row in rows if row["split"] == "test" and row["upos"] == "ALL")
    assert float(test["oov_rate"]) == 1
    assert float(test["f_train_le_2_rate"]) == 1
    assert float(test["seen_rare_le_2_rate"]) == 0


def test_audit_duplicates_and_issues(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Trùng dựa trên tuple token, bỏ qua tag/ID; mọi cặp và lỗi đều được báo."""
    from src.eda import audit_splits

    splits = {
        "train": [Sentence("same", ["sinh viên"], ["NOUN"]),
                  Sentence("same", ["sinh viên"], ["VERB"]),
                  Sentence(None, ["sinh", "viên"], ["NOUN", "NOUN"]),
                  Sentence(None, [], [])],
        "dev": [Sentence(None, ["sinh viên"], ["_"])],
        "test": [Sentence(None, ["sinh viên"], ["NN"]), Sentence(None, [], [])],
    }
    before = deepcopy(splits)
    path = audit_splits(splits, tmp_path)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    within = [r for r in rows if r["issue"] == "duplicate_within"]
    assert len(within) == 1
    assert (within[0]["sentence_index_a"], within[0]["sentence_index_b"]) == ("0", "1")
    cross = [r for r in rows if r["issue"] == "duplicate_cross"]
    assert len([r for r in cross if (r["split_a"], r["split_b"]) == ("train", "dev")]) == 2
    assert len([r for r in cross if (r["split_a"], r["split_b"]) == ("train", "test")]) == 3
    assert len([r for r in cross if (r["split_a"], r["split_b"]) == ("dev", "test")]) == 1
    assert len([r for r in rows if r["issue"] == "empty_sentence"]) == 2
    invalid = [r for r in rows if r["issue"] == "unknown_tag"]
    assert {r["tag"] for r in invalid} == {"_", "NN"}
    assert all(r["token_index"] == "0" for r in invalid)
    assert splits == before
    summary = capsys.readouterr().out
    assert "train ∩ test: 3 cặp" in summary
    assert "train: 1 cặp" in summary


def test_audit_no_issues(tmp_path: Path) -> None:
    """Không có lỗi vẫn xuất CSV có header để công cụ đọc nhất quán."""
    from src.eda import audit_splits

    path = audit_splits({"train": [Sentence(None, ["x"], ["NOUN"])],
                         "dev": [], "test": []}, tmp_path)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert "issue" in reader.fieldnames
        assert list(reader) == []
