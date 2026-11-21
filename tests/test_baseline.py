from collections import Counter
from pathlib import Path

import pytest

from src.baseline import MostFrequentTagger
from src.contracts import UPOS_TAGS
from src.data_loader import Sentence


def test_word_majority_and_global_oov_fallback():
    train = [(["a", "b", "a", "a"], ["NOUN", "VERB", "VERB", "VERB"])]
    model = MostFrequentTagger()
    assert model.fit(train) is model
    assert model.predict(["a", "b", "unseen"]) == ["VERB", "VERB", "VERB"]
    assert model.train_vocab_ == {"a", "b"}
    assert "unseen" not in model.train_vocab_
    assert set(model.predict(["a", "unseen"])) <= set(UPOS_TAGS)


def test_ties_are_lexical_and_independent_of_order():
    train = [(["a", "a", "b", "b"], ["VERB", "NOUN", "VERB", "NOUN"])]
    forward = MostFrequentTagger().fit(train)
    reverse = MostFrequentTagger().fit([(train[0][0][::-1], train[0][1][::-1])])
    assert forward.predict(["a", "unknown"]) == ["NOUN", "NOUN"]
    assert reverse.predict(["a", "unknown"]) == ["NOUN", "NOUN"]


def test_predict_requires_fit_even_for_empty_sentence():
    with pytest.raises(RuntimeError, match="fit"):
        MostFrequentTagger().predict([])
    assert MostFrequentTagger().fit([(["a"], ["NOUN"])]).predict([]) == []


@pytest.mark.parametrize("train", [[], [([], [])], [(["a"], [])],
                                      [(["a"], ["BAD"])], [([""], ["NOUN"])]])
def test_invalid_training_data(train):
    with pytest.raises(ValueError):
        MostFrequentTagger().fit(train)


def test_refit_replaces_training_state():
    model = MostFrequentTagger().fit([(["old"], ["NOUN"])])
    model.fit([(["new"], ["VERB"])])
    assert model.train_vocab_ == {"new"}
    assert model.predict(["old"]) == ["VERB"]


def test_failed_refit_preserves_previous_model():
    model = MostFrequentTagger().fit([(["old"], ["NOUN"])])
    with pytest.raises(ValueError):
        model.fit([])
    assert model.predict(["old"]) == ["NOUN"]


def test_dataclass_ties_follow_all_upos() -> None:
    """Mọi cặp nhãn hòa phải chọn nhãn đứng trước trong UPOS, bất kể thứ tự train."""
    from src.baseline import MostFrequentTag
    from src.data_loader import Sentence

    for i, first in enumerate(UPOS_TAGS):
        for second in UPOS_TAGS[i + 1:]:
            model = MostFrequentTag().fit([Sentence(None, ["x", "x"], [second, first])])
            assert model.predict(["x", "new"]) == [first, first]


def test_export_predictions_preserves_model_and_original_vocab(tmp_path) -> None:
    """Export giữ token có dấu cách, OOV theo train gốc và không học gold dev."""
    import csv
    from copy import deepcopy
    from src.baseline import MostFrequentTag, export_predictions
    from src.data_loader import Sentence

    model = MostFrequentTag().fit([Sentence("t", ["sinh viên", "học", "học"], ["NOUN", "VERB", "VERB"])])
    dev = [Sentence("d", ["sinh viên", "mới"], ["PROPN", "ADJ"])]
    before = deepcopy(model.__dict__)
    path = export_predictions(model, dev, model.train_vocab_, tmp_path / "pred.csv", split="dev")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {"split": "dev", "sent_id": "d", "idx": "0", "token": "sinh viên", "gold": "PROPN", "pred": "NOUN", "is_oov": "False"},
        {"split": "dev", "sent_id": "d", "idx": "1", "token": "mới", "gold": "ADJ", "pred": "VERB", "is_oov": "True"},
    ]
    assert model.__dict__ == before
    with pytest.raises(ValueError, match="vocab_train"):
        export_predictions(model, dev, {"học"}, tmp_path / "bad.csv")
    assert not (tmp_path / "bad.csv").exists()


def test_export_missing_metadata_and_empty_input(tmp_path) -> None:
    """Không suy đoán split/sent_id; đầu vào rỗng vẫn ghi header."""
    import csv
    from src.baseline import MostFrequentTag, export_predictions
    from src.data_loader import Sentence

    model = MostFrequentTag().fit([Sentence(None, ["x"], ["NOUN"])])
    path = export_predictions(model, [Sentence(None, ["x"], ["NOUN"])], model.train_vocab_, tmp_path / "p.csv")
    with path.open(encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["split"] == row["sent_id"] == ""
    export_predictions(model, [], model.train_vocab_, path)
    with path.open(encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == []


@pytest.fixture
def five_sentence_train() -> list[Sentence]:
    """Corpus giả 5 câu, bảng đếm được tính tay để kiểm tra độc lập."""
    from src.data_loader import Sentence

    # a: VERB=2, NOUN=2 -> NOUN theo thứ tự UPOS; b: NOUN=3.
    # c: VERB=2; d: ADJ=1. Tổng: NOUN=5, VERB=4, ADJ=1 -> majority NOUN.
    return [
        Sentence("t1", ["a", "b"], ["VERB", "NOUN"]),
        Sentence("t2", ["a", "c"], ["NOUN", "VERB"]),
        Sentence("t3", ["a", "b"], ["VERB", "NOUN"]),
        Sentence("t4", ["a", "d"], ["NOUN", "ADJ"]),
        Sentence("t5", ["c", "b"], ["VERB", "NOUN"]),
    ]


def test_five_sentences_tie_and_unseen(five_sentence_train: list[Sentence]) -> None:
    """Hòa theo từ chọn NOUN; unseen dùng majority toàn train, không dùng từ cuối."""
    from src.baseline import MostFrequentTag

    model = MostFrequentTag().fit(five_sentence_train)
    # a hòa N=2/V=2; unseen dùng N=5 > V=4 > ADJ=1.
    assert model.predict(["a", "b", "c", "d", "unseen"]) == ["NOUN", "NOUN", "VERB", "ADJ", "NOUN"]
    assert model.word_tags_ == {"a": "NOUN", "b": "NOUN", "c": "VERB", "d": "ADJ"}
    assert model.train_vocab_ == {"a", "b", "c", "d"}


def test_five_sentences_refit_is_identical(five_sentence_train: list[Sentence], monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit lại không cộng dồn counts; trạng thái và dự đoán phải giống lần đầu."""
    from copy import deepcopy
    import src.baseline as baseline

    captured = []
    original = baseline._most_frequent

    def capture(counts: Counter[str]) -> str:
        captured.append(dict(counts))
        return original(counts)

    # Chụp bảng đếm tại điểm chọn nhãn vì model chỉ lưu nhãn thắng, không lưu counts.
    monkeypatch.setattr(baseline, "_most_frequent", capture)
    model = baseline.MostFrequentTag().fit(five_sentence_train)
    expected_counts = [{"VERB": 2, "NOUN": 2}, {"NOUN": 3}, {"VERB": 2},
                       {"ADJ": 1}, {"VERB": 4, "NOUN": 5, "ADJ": 1}]
    assert captured == expected_counts
    state = deepcopy(model.__dict__)
    first = model.predict(["a", "c", "unknown"])
    captured.clear()
    assert model.fit(five_sentence_train) is model
    assert captured == expected_counts  # Không được tăng thành N=10, V=8, ADJ=2.
    assert model.__dict__ == state
    assert model.predict(["a", "c", "unknown"]) == first == ["NOUN", "VERB", "NOUN"]


def test_changed_dev_cannot_change_training_state(five_sentence_train: list[Sentence], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Đổi token/gold dev và gọi export không được đếm lại hoặc sửa mô hình."""
    from copy import deepcopy
    import src.baseline as baseline
    from src.data_loader import Sentence

    train_before = deepcopy(five_sentence_train)
    model = baseline.MostFrequentTag().fit(five_sentence_train)
    state = deepcopy(model.__dict__)

    def forbid_recount(counts: Counter[str]) -> str:
        raise AssertionError("predict/export không được chọn lại nhãn từ counts")

    monkeypatch.setattr(baseline, "_most_frequent", forbid_recount)
    dev = [Sentence("dev", ["a", "new"], ["VERB", "VERB"])]
    baseline.export_predictions(model, dev, model.train_vocab_, tmp_path / "before.csv", split="dev")
    # Nếu học từ dev mới, a sẽ nghiêng về ADJ và new sẽ thành từ đã thấy; cả hai đều sai.
    dev[0].tokens[:] = ["a"] * 20 + ["new"] * 20
    dev[0].tags[:] = ["ADJ"] * 40
    baseline.export_predictions(model, dev, model.train_vocab_, tmp_path / "after.csv", split="dev")
    assert model.__dict__ == state
    assert model.predict(["a", "new"]) == ["NOUN", "NOUN"]
    assert "new" not in model.train_vocab_
    assert five_sentence_train == train_before


@pytest.mark.parametrize("tokens, expected", [
    ([], []),
    (["new"], ["NOUN"]),
    (["a", "c", "a", "d", "new"], ["NOUN", "VERB", "NOUN", "ADJ", "NOUN"]),
])
def test_prediction_length_and_empty_sentence(five_sentence_train: list[Sentence], tokens: list[str], expected: list[str]) -> None:
    """Mỗi token có đúng một nhãn; câu rỗng sau fit có 0 token nên trả 0 nhãn."""
    from src.baseline import MostFrequentTag

    model = MostFrequentTag().fit(five_sentence_train)
    # Theo bảng tính tay: a=NOUN, c=VERB, d=ADJ, new=majority NOUN.
    result = model.predict(tokens)
    assert result == expected
    assert len(result) == len(tokens)
