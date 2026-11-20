from pathlib import Path

import pytest

from src.data_loader import load_conllu

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.conllu"


def test_fixture_sentences_tokens_and_skipped_rows():
    corpus = load_conllu(FIXTURE)
    assert [len(tokens) for tokens, _ in corpus] == [5, 3, 3]
    assert sum(len(tokens) for tokens, _ in corpus) == 11
    assert corpus[0] == (["Tôi", "yêu", "Hà", "Nội", "."],
                         ["PRON", "VERB", "PROPN", "PROPN", "PUNCT"])
    assert all(len(tokens) == len(tags) for tokens, tags in corpus)


def test_internal_spaces_preserved():
    tokens, tags = load_conllu(FIXTURE)[1]
    assert tokens == ["Hà Nội", "đẹp", "."]
    assert tags == ["PROPN", "ADJ", "PUNCT"]


def test_eof_without_blank_line_and_extra_separators(tmp_path):
    path = tmp_path / "toy.conllu"
    path.write_text("\n# comment\n\n1\tx y\t_\tNOUN\t_\t_\t0\troot\t_\t_", encoding="utf-8")
    assert load_conllu(path) == [(["x y"], ["NOUN"])]


def test_empty_file_has_no_sentences(tmp_path):
    path = tmp_path / "empty.conllu"
    path.write_text("# no sentences\n\n", encoding="utf-8")
    assert load_conllu(path) == []


@pytest.mark.parametrize("row", [
    "1\tx\tNOUN",
    "0\tx\t_\tNOUN\t_\t_\t0\troot\t_\t_",
    "bad\tx\t_\tNOUN\t_\t_\t0\troot\t_\t_",
    "1\tx\t_\t_\t_\t_\t0\troot\t_\t_",
    "1\t\t_\tNOUN\t_\t_\t0\troot\t_\t_",
])
def test_malformed_row_reports_line_number(tmp_path, row):
    path = tmp_path / "bad.conllu"
    path.write_text("# comment\n" + row, encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:"):
        load_conllu(path)


def test_sentence_ids_and_nfc(tmp_path: Path) -> None:
    """Giữ ID, khoảng trắng và chỉ chuẩn hóa khi được yêu cầu."""
    from src.data_loader import Sentence, read_conllu

    path = tmp_path / "unicode.conllu"
    form = "Ha\u0300 No\u0323\u0302i"
    path.write_text(
        f"# sent_id = first\n1\t{form}\t_\tPROPN\t_\t_\t0\troot\t_\t_\n\n"
        "1\tx\t_\tNOUN\t_\t_\t0\troot\t_\t_", encoding="utf-8"
    )
    sentences = read_conllu(path)
    assert sentences[0] == Sentence("first", [form], ["PROPN"])
    assert sentences[1].sent_id is None
    assert read_conllu(path, nfc=True)[0].tokens == ["Hà Nội"]
    with pytest.raises(AssertionError):
        Sentence(None, ["x"], [])


@pytest.mark.parametrize("tag", ["_", "NN", "noun"])
def test_invalid_upos_is_explicit(tmp_path: Path, tag: str) -> None:
    """Lỗi chứa dòng và nhãn bị từ chối."""
    from src.data_loader import read_conllu

    path = tmp_path / "bad.conllu"
    path.write_text(f"1\tx\t_\t{tag}\t_\t_\t0\troot\t_\t_", encoding="utf-8")
    with pytest.raises(ValueError, match=rf":1: UPOS '{tag}'"):
        read_conllu(path)


def test_load_splits_uses_manifest(tmp_path: Path) -> None:
    """Đọc đúng ánh xạ split, không phụ thuộc tên file cố định."""
    import json
    from src.data_loader import load_splits

    records = []
    for index, split in enumerate(("test", "train", "dev")):
        name = f"toy-{index}.conllu"
        (tmp_path / name).write_text(
            f"# sent_id = {split}\n1\tx y\t_\tNOUN\t_\t_\t0\troot\t_\t_",
            encoding="utf-8",
        )
        records.append({"split": split, "filename": name})
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"files": records}), encoding="utf-8")
    splits = load_splits(tmp_path)
    assert list(splits) == ["train", "dev", "test"]
    assert all(splits[s][0].sent_id == s for s in splits)
    assert splits["train"][0].tokens == ["x y"]
    records[0]["filename"] = "../outside.conllu"
    manifest.write_text(json.dumps({"files": records}), encoding="utf-8")
    with pytest.raises(ValueError, match="tên file"):
        load_splits(tmp_path)


@pytest.mark.parametrize("records", [[], [None, None, None], [
    {"split": "train", "filename": "a.conllu"},
    {"split": "train", "filename": "b.conllu"},
    {"split": "test", "filename": "c.conllu"},
]])
def test_load_splits_rejects_bad_manifest(tmp_path: Path, records: list) -> None:
    """Không chấp nhận manifest thiếu hoặc trùng split."""
    import json
    from src.data_loader import load_splits

    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": records}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_splits(tmp_path)
