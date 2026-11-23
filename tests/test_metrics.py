import pytest

from src.contracts import UPOS_TAGS
from src.metrics import evaluate_pos


def test_hand_calculated_accuracy_and_fixed_label_macro_f1():
    gold = [(["a", "b", "c", "."], ["NOUN", "VERB", "NOUN", "PUNCT"])]
    predictions = [["NOUN", "NOUN", "NOUN", "PUNCT"]]
    scores = evaluate_pos(
        gold, predictions, {"a", "b"}, ["NOUN", "VERB", "PUNCT", "ADJ"]
    )
    assert scores["token_accuracy"] == pytest.approx(3 / 4)
    # NOUN F1=4/5; VERB=0; PUNCT=1; absent ADJ=0.
    assert scores["macro_f1"] == pytest.approx((4 / 5 + 0 + 1 + 0) / 4)
    assert scores["oov_accuracy"] == pytest.approx(1.0)
    assert scores["oov_rate"] == pytest.approx(1 / 2)
    assert scores["accuracy_no_punct"] == pytest.approx(2 / 3)


def test_gold_support_macro_f1_keeps_errors_predicted_as_an_absent_tag():
    gold = [(["x", "y"], ["NOUN", "NOUN"])]
    predictions = [["NOUN", "VERB"]]
    labels = ["NOUN", "VERB", "ADJ"]
    scores = evaluate_pos(gold, predictions, set(), labels, macro_f1_gold_support=True)
    assert scores["token_accuracy"] == 0.5
    assert scores["macro_f1"] == pytest.approx(2 / 3)
    assert evaluate_pos(gold, predictions, set(), labels)["macro_f1"] == pytest.approx(2 / 9)


@pytest.mark.parametrize("gold,predicted", [([], []), ([([], [])], [[]])])
def test_empty_evaluation(gold, predicted):
    result = evaluate_pos(gold, predicted, set(), UPOS_TAGS)
    assert result == {
        "token_accuracy": 0.0,
        "macro_f1": 0.0,
        "oov_accuracy": 0.0,
        "oov_rate": 0.0,
        "accuracy_no_punct": 0.0,
    }


def test_accuracy_is_weighted_by_tokens_not_sentences():
    gold = [(["a"], ["NOUN"]), (["b", "c", "d"], ["VERB"] * 3)]
    scores = evaluate_pos(gold, [["NOUN"], ["NOUN"] * 3], set(), UPOS_TAGS)
    assert scores["token_accuracy"] == 1 / 4


def test_oov_uses_raw_training_vocabulary():
    gold = [(["known", "unseen", "rare", "."], ["NOUN", "VERB", "ADJ", "PUNCT"])]
    predicted = [["NOUN", "NOUN", "ADJ", "PUNCT"]]
    scores = evaluate_pos(gold, predicted, {"known", "rare", "."}, UPOS_TAGS)
    assert scores["oov_accuracy"] == 0.0
    assert scores["oov_rate"] == pytest.approx(1 / 4)


def test_accuracy_no_punct_filters_using_gold_tag():
    gold = [(["a", ".", "b"], ["NOUN", "PUNCT", "VERB"])]
    predicted = [["NOUN", "VERB", "NOUN"]]
    scores = evaluate_pos(gold, predicted, {"a", ".", "b"}, UPOS_TAGS)
    assert scores["accuracy_no_punct"] == pytest.approx(1 / 2)


def test_zero_oov_and_zero_non_punct_denominators_return_zero():
    gold = [(["."], ["PUNCT"])]
    scores = evaluate_pos(gold, [["PUNCT"]], {"."}, UPOS_TAGS)
    assert scores["oov_accuracy"] == 0.0
    assert scores["oov_rate"] == 0.0
    assert scores["accuracy_no_punct"] == 0.0


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


def test_reject_invalid_training_vocabulary():
    with pytest.raises(ValueError):
        evaluate_pos([(["a"], ["NOUN"])], [["NOUN"]], {"", "a"}, UPOS_TAGS)
