from pathlib import Path

import pytest

from src.data_loader import load_conllu, read_conllu

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.conllu"


def test_fixture_sentences_tokens_and_skipped_rows() -> None:
    """Chỉ dòng ID nguyên tạo token nên bỏ range/empty node mà vẫn căn chỉnh."""
    corpus = load_conllu(FIXTURE)
    assert [len(tokens) for tokens, _ in corpus] == [5, 3, 3]
    assert sum(len(tokens) for tokens, _ in corpus) == 11
    assert corpus[0] == (["Tôi", "yêu", "Hà", "Nội", "."],
                         ["PRON", "VERB", "PROPN", "PROPN", "PUNCT"])
    assert all(len(tokens) == len(tags) for tokens, tags in corpus)


def test_internal_spaces_preserved() -> None:
    """Khoảng trắng bên trong FORM không phải ranh giới cột hay token."""
    tokens, tags = load_conllu(FIXTURE)[1]
    assert tokens == ["Hà Nội", "đẹp", "."]
    assert tags == ["PROPN", "ADJ", "PUNCT"]


def test_eof_without_blank_line_and_extra_separators(tmp_path: Path) -> None:
    """EOF kết thúc câu cuối dù không có dòng trống, còn dòng trống không tạo câu rỗng."""
    path = tmp_path / "toy.conllu"
    path.write_text("\n# comment\n\n1\tx y\t_\tNOUN\t_\t_\t0\troot\t_\t_", encoding="utf-8")
    assert load_conllu(path) == [(["x y"], ["NOUN"])]


def test_empty_file_has_no_sentences(tmp_path: Path) -> None:
    """Comment không chứa token nên file chỉ có comment không tạo câu."""
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
def test_malformed_row_reports_line_number(tmp_path: Path, row: str) -> None:
    """Dòng sai cấu trúc phải báo vị trí để truy được lỗi dữ liệu."""
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


MINI = Path(__file__).parent / "fixtures" / "loader_mini.conllu"


def test_comments_are_metadata_only() -> None:
    """Comment text không sinh token; chỉ sent_id được giữ làm metadata."""
    sentences = read_conllu(MINI)
    assert [s.sent_id for s in sentences] == ["mini-1", "mini-single"]
    assert sentences[0].tokens == ["sinh viên", "học", "."]
    assert sentences[0].tags == ["NOUN", "VERB", "PUNCT"]


def test_multiword_id_1_2_is_skipped() -> None:
    """ID 1-2 là dòng bao nhiều token, không được đếm thêm hay kiểm tra UPOS '_'."""
    sentence = read_conllu(MINI)[0]
    assert "MWT_ONLY" not in sentence.tokens
    assert len(sentence.tokens) == 3


def test_empty_node_3_1_is_skipped() -> None:
    """ID 3.1 là empty node nên không thuộc chuỗi token dùng gán nhãn."""
    sentence = read_conllu(MINI)[0]
    assert "EMPTY_ONLY" not in sentence.tokens
    assert sentence.tags == ["NOUN", "VERB", "PUNCT"]


def test_sinh_vien_is_one_token() -> None:
    """FORM 'sinh viên' là một cột tab nên phải giữ nguyên thành một token."""
    sentence = read_conllu(MINI)[0]
    assert sentence.tokens[0] == "sinh viên"
    assert sentence.tags[0] == "NOUN"
    assert "sinh" not in sentence.tokens and "viên" not in sentence.tokens


def test_single_token_sentence() -> None:
    """Một token vẫn tạo câu hợp lệ và có đúng một nhãn."""
    sentence = read_conllu(MINI)[1]
    assert sentence.tokens == ["Cha\u0300o"]
    assert sentence.tags == ["INTJ"]


@pytest.mark.parametrize("name, tag", [("loader_invalid_upos.conllu", "NN"),
                                      ("loader_missing_upos.conllu", "_")])
def test_invalid_fixture_upos(name: str, tag: str) -> None:
    """Nhãn ngoài UPOS hoặc '_' không có gold hợp lệ nên phải báo lỗi rõ ràng."""
    with pytest.raises(ValueError, match=rf":3: UPOS '{tag}'"):
        read_conllu(MINI.parent / name)


@pytest.mark.parametrize("nfc", [False, True])
def test_every_sentence_is_aligned(nfc: bool) -> None:
    """Mỗi dòng token hợp lệ đóng góp đồng thời một FORM và một UPOS."""
    sentences = read_conllu(MINI, nfc=nfc)
    assert [len(s.tokens) for s in sentences] == [3, 1]
    assert all(len(s.tokens) == len(s.tags) for s in sentences)


def test_nfc_preserves_token_count() -> None:
    """NFC chỉ ghép dấu Unicode, không tách/gộp token hoặc thay nhãn và ID câu."""
    original = read_conllu(MINI)
    normalized = read_conllu(MINI, nfc=True)
    assert original[1].tokens == ["Cha\u0300o"]
    assert normalized[1].tokens == ["Chào"]
    assert len(original) == len(normalized) == 2
    for before, after in zip(original, normalized):
        assert len(before.tokens) == len(after.tokens)
        assert before.tags == after.tags
        assert before.sent_id == after.sent_id
