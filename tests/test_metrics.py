import pytest

from src.contracts import UPOS_TAGS
from src.metrics import evaluate_pos


def test_hand_calculated_accuracy_and_fixed_label_macro_f1():
    gold = [(["a", "b", "c", "."], ["NOUN", "VERB", "NOUN", "PUNCT"])]
    predictions = [["NOUN", "NOUN", "NOUN", "PUNCT"]]
    scores = evaluate_pos(gold, predictions, {"a", "b"}, ["NOUN", "VERB", "PUNCT", "ADJ"])
    assert scores["token_accuracy"] == pytest.approx(3 / 4)
    # NOUN: 2*2/(2*2+1+0)=4/5; VERB=0; PUNCT=1; absent ADJ=0.
    assert scores["macro_f1"] == pytest.approx((4 / 5 + 0 + 1 + 0) / 4)
    assert {key: scores[key] for key in ("oov_accuracy", "oov_rate", "accuracy_no_punct")} == {
        "oov_accuracy": None, "oov_rate": None, "accuracy_no_punct": None,
    }


@pytest.mark.parametrize("gold,predicted", [([], []), ([([], [])], [[]])])
def test_empty_evaluation(gold, predicted):
    result = evaluate_pos(gold, predicted, set(), UPOS_TAGS)
    assert result["token_accuracy"] == 0.0
    assert result["macro_f1"] == 0.0


def test_accuracy_is_weighted_by_tokens_not_sentences():
    gold = [(["a"], ["NOUN"]), (["b", "c", "d"], ["VERB"] * 3)]
    scores = evaluate_pos(gold, [["NOUN"], ["NOUN"] * 3], set(), UPOS_TAGS)
    assert scores["token_accuracy"] == 1 / 4


@pytest.mark.parametrize("predictions,labels", [
    ([], UPOS_TAGS),
    ([[]], UPOS_TAGS),
    ([["BAD"]], UPOS_TAGS),
    ([["NOUN"]], []),
    ([["NOUN"]], ["NOUN", "NOUN"]),
    ([["NOUN"]], ["BAD"]),
    ([["NOUN"]], ["VERB"]),
    ([["VERB"]], ["NOUN"]),
])
def test_reject_misalignment_invalid_tags_or_missing_fixed_labels(predictions, labels):
    with pytest.raises(ValueError):
        evaluate_pos([(["a"], ["NOUN"])], predictions, {"a"}, labels)
