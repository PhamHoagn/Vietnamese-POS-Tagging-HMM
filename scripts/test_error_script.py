"""Kiểm tra phân tích lỗi trên CSV gộp tự tạo, không dùng corpus thật."""

import csv
from pathlib import Path

import pytest

from src.error_script import export_error_analysis


def test_errors_pairs_and_deterministic_examples(tmp_path: Path) -> None:
    """B0 có 2 NOUN→VERB OOV nên nhóm này được lấy ví dụ đầu theo thứ tự CSV."""
    source = tmp_path / "predictions.csv"
    source.write_text(
        "sent_id,idx,token,gold,pred_b0,pred_h1,pred_h2,is_oov\n"
        "s1,0,Tôi,PRON,PRON,PRON,PRON,False\n"
        "s1,1,học,VERB,NOUN,VERB,NOUN,False\n"
        "s1,2,mới,NOUN,VERB,VERB,NOUN,True\n"
        "s1,3,.,PUNCT,NOUN,PUNCT,PUNCT,False\n"
        "s2,0,lạ,NOUN,VERB,NOUN,NOUN,True\n",
        encoding="utf-8",
    )
    before = source.read_bytes()
    paths = export_error_analysis(source, {"Tôi", "học", "."}, tmp_path / "out", example_limit=5)
    with paths["errors"].open(encoding="utf-8", newline="") as stream:
        errors = list(csv.DictReader(stream))
    assert len(errors) == 6
    assert {(row["model"], row["token"]) for row in errors} == {
        ("B0", "học"), ("H2", "học"), ("B0", "mới"), ("H1", "mới"),
        ("B0", "."), ("B0", "lạ"),
    }
    with paths["pairs"].open(encoding="utf-8", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    top = pairs[0]
    assert (top["model"], top["oov_group"], top["punctuation_group"], top["gold"], top["pred"], top["count"]) == (
        "B0", "OOV", "NON_PUNCT", "NOUN", "VERB", "2"
    )
    assert float(top["percentage_within_group"]) == 100
    report = paths["examples"].read_text(encoding="utf-8")
    assert report.splitlines()[0] == "| Câu/token | Gold | B0 | H1 | H2 | OOV? | Nhóm lỗi/giải thích |"
    assert "s1 / 2: Tôi học [mới] ." in report
    assert "[điền sau khi xem]" in report
    assert source.read_bytes() == before


def test_oov_must_come_from_train_or_csv(tmp_path: Path) -> None:
    """Thiếu OOV source thì không thể tách OOV/IV nên phải từ chối thay vì đoán."""
    source = tmp_path / "predictions.csv"
    source.write_text("sent_id,idx,token,gold,pred_b0,pred_h1\ns,0,x,NOUN,NOUN,NOUN\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is_oov hoặc vocab_train"):
        export_error_analysis(source, output_dir=tmp_path / "out", example_limit=5)
    paths = export_error_analysis(source, {"x"}, tmp_path / "out", example_limit=5)
    assert paths["errors"].exists()


def test_is_oov_and_train_vocab_must_agree(tmp_path: Path) -> None:
    """CSV OOV mâu thuẫn train vocab là dấu hiệu leakage hoặc nhầm corpus cần chặn."""
    source = tmp_path / "predictions.csv"
    source.write_text(
        "sent_id,idx,token,gold,pred_b0,pred_h1,is_oov\ns,0,x,NOUN,VERB,NOUN,True\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="không khớp"):
        export_error_analysis(source, {"x"}, tmp_path / "out", example_limit=5)


def test_h2_is_optional_and_no_errors_has_header(tmp_path: Path) -> None:
    """H2 chưa có vẫn xuất bảng đúng mẫu; không bịa ví dụ khi mọi token đều đúng."""
    source = tmp_path / "predictions.csv"
    source.write_text(
        "sent_id,idx,token,gold,pred_b0,pred_h1,is_oov\ns,0,x,NOUN,NOUN,NOUN,False\n",
        encoding="utf-8",
    )
    paths = export_error_analysis(source, output_dir=tmp_path / "out", example_limit=5)
    report = paths["examples"].read_text(encoding="utf-8").splitlines()
    assert len(report) == 2
    for key in ("errors", "pairs"):
        with paths[key].open(encoding="utf-8") as stream:
            assert list(csv.DictReader(stream)) == []
