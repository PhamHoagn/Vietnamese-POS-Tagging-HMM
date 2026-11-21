import pytest

from src.baseline import MostFrequentTagger
from src.contracts import UPOS_TAGS


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
