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
