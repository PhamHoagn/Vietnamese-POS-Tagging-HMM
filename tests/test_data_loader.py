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
