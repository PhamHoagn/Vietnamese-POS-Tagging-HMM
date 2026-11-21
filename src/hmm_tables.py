"""Load explicit probability tables for testing the production HMM decoders.

This module converts and validates tables; it never trains, smooths, normalizes
them again, or implements a second decoder/oracle. Non-UPOS toy labels are only
permitted through this explicit API, not through fit(train_data).
"""

from collections.abc import Mapping, Sequence
from itertools import product
from types import MappingProxyType

import numpy as np

from .hmm_common import END, START, UNK, _readonly


def _distribution(row: Mapping[str, float], keys: tuple[str, ...]) -> np.ndarray:
    if not isinstance(row, Mapping) or set(row) != set(keys):
        raise ValueError("Probability row must contain exactly the declared outcomes.")
    values = np.array([row[key] for key in keys], dtype=np.float64)
    if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
            or not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("Probabilities must be finite, non-negative and sum to one.")
    return values


def _logs(probabilities: np.ndarray) -> np.ndarray:
    result = np.full(probabilities.shape, -np.inf, dtype=np.float64)
    np.log(probabilities, out=result, where=probabilities > 0)
    return _readonly(result)


def initialize_from_probabilities(
    model, labels: Sequence[str], vocabulary: Sequence[str], transition: Mapping,
    emission: Mapping[str, Mapping[str, float]], *, initial: Mapping[str, float] | None = None,
):
    """Copy probability tables into the same model/decoder used by trained HMMs.

    Giữ nguyên thứ tự nhãn/từ và xác suất, chỉ đổi biểu diễn sang log cho decoder.
    Bigram dùng initial riêng trên S, không tự chuẩn hóa nó theo hàng S+EOS.
    Trigram giữ lịch sử (START,START), (START,tag) và các cặp tag đầy đủ.
    """
    labels, vocabulary = tuple(labels), tuple(vocabulary)
    for name, values in (("labels", labels), ("vocabulary", vocabulary)):
        if (not values or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{name} must contain distinct non-empty strings.")
    if START in labels or END in labels or set(emission) != set(labels):
        raise ValueError("Labels exclude boundaries and must match emission rows.")
    size = len(labels)
    tag_index = {tag: index for index, tag in enumerate(labels)}
    targets = (*labels, END)
    if model.model_kind == "bigram":
        if set(transition) != set(labels) or initial is None:
            raise ValueError("Bigram tables require initial and one transition row per label.")
        probabilities = np.zeros((size + 1, size + 1), dtype=np.float64)
        for tag in labels:
            probabilities[tag_index[tag]] = _distribution(transition[tag], targets)
        probabilities[size, :size] = _distribution(initial, labels)
        histories = frozenset((tag,) for tag in labels)
    else:
        if initial is not None:
            raise ValueError("Trigram tables use two STARTs, not a separate initial distribution.")
        required = {(START, START), *((START, tag) for tag in labels), *product(labels, repeat=2)}
        allowed = set(product((*labels, START), repeat=2))
        if not required.issubset(transition) or not set(transition).issubset(allowed):
            raise ValueError("Trigram tables must define all reachable histories.")
        probabilities = np.zeros((size + 1,) * 3, dtype=np.float64)
        for (first, second), row in transition.items():
            u = size if first == START else tag_index[first]
            v = size if second == START else tag_index[second]
            probabilities[u, v] = _distribution(row, targets)
        histories = frozenset(transition)
    emission_probs = np.stack([_distribution(emission[tag], vocabulary) for tag in labels])
    model.tags = labels
    model.tag_to_index_ = MappingProxyType(tag_index)
    word_index = {word: index for index, word in enumerate(vocabulary)}
    model.unk_index_ = word_index.get("<UNK>")
    if model.unk_index_ is not None:
        word_index[UNK] = model.unk_index_
    model.word_to_index_ = MappingProxyType(word_index)
    model.retained_vocab_ = frozenset(word for word in vocabulary if word != "<UNK>")
    model.vocab_ = frozenset(vocabulary)
    model._provided_transition_probs = _readonly(probabilities)
    model._provided_emission_probs = _readonly(emission_probs)
    model._provided_histories = histories
    model.transition_log_probs_ = _logs(probabilities)
    model.emission_log_probs_ = _logs(emission_probs)
    model._table_mode = True
    model._fitted_parameters = (model.alpha, model.cutoff)
    return model
