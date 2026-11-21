"""Evaluation metrics for Vietnamese UPOS tagging."""

from collections.abc import Sequence, Set
from typing import TypedDict

from sklearn.metrics import f1_score

from .contracts import Corpus, UPOS_TAGS, validate_corpus


class POSMetrics(TypedDict):
    token_accuracy: float
    macro_f1: float
    oov_accuracy: float
    oov_rate: float
    accuracy_no_punct: float


def evaluate_pos(
    gold_data: Corpus,
    predictions: list[list[str]],
    train_vocab: Set[str],
    labels: Sequence[str],
) -> POSMetrics:
    """Evaluate predictions against gold UPOS labels.

    OOV is determined from the raw training vocabulary (before any HMM cutoff).
    Tokens are flattened across sentences for token-level metrics. A metric whose
    denominator is zero is defined as 0.0.
    """
    validate_corpus(gold_data)
    if len(gold_data) != len(predictions):
        raise ValueError("Gold and predictions must contain the same number of sentences.")

    fixed_labels = list(labels)
    if (not fixed_labels or len(fixed_labels) != len(set(fixed_labels))
            or any(label not in UPOS_TAGS for label in fixed_labels)):
        raise ValueError("labels must be a non-empty sequence of unique UPOS labels.")
    if any(not isinstance(word, str) or not word for word in train_vocab):
        raise ValueError("train_vocab must contain non-empty strings from training only.")

    tokens_flat: list[str] = []
    gold_flat: list[str] = []
    pred_flat: list[str] = []

    for index, ((tokens, gold), predicted) in enumerate(zip(gold_data, predictions)):
        if len(tokens) != len(predicted):
            raise ValueError(f"Sentence {index}: token/prediction lengths differ.")
        if any(not isinstance(tag, str) or tag not in UPOS_TAGS for tag in predicted):
            raise ValueError(f"Sentence {index}: predictions must be valid UPOS labels.")
        tokens_flat.extend(tokens)
        gold_flat.extend(gold)
        pred_flat.extend(predicted)

    if not (set(gold_flat) | set(pred_flat)).issubset(fixed_labels):
        raise ValueError("labels must include every gold and predicted tag.")

    total = len(tokens_flat)
    correct = [gold == pred for gold, pred in zip(gold_flat, pred_flat)]
    token_accuracy = sum(correct) / total if total else 0.0
    macro_f1 = (
        float(f1_score(
            gold_flat,
            pred_flat,
            labels=fixed_labels,
            average="macro",
            zero_division=0,
        ))
        if total else 0.0
    )

    oov_mask = [token not in train_vocab for token in tokens_flat]
    oov_count = sum(oov_mask)
    oov_correct = sum(is_correct for is_correct, is_oov in zip(correct, oov_mask) if is_oov)
    oov_accuracy = oov_correct / oov_count if oov_count else 0.0
    oov_rate = oov_count / total if total else 0.0

    non_punct_mask = [tag != "PUNCT" for tag in gold_flat]
    non_punct_count = sum(non_punct_mask)
    non_punct_correct = sum(
        is_correct
        for is_correct, is_non_punct in zip(correct, non_punct_mask)
        if is_non_punct
    )
    accuracy_no_punct = (
        non_punct_correct / non_punct_count if non_punct_count else 0.0
    )

    return {
        "token_accuracy": token_accuracy,
        "macro_f1": macro_f1,
        "oov_accuracy": oov_accuracy,
        "oov_rate": oov_rate,
        "accuracy_no_punct": accuracy_no_punct,
    }
