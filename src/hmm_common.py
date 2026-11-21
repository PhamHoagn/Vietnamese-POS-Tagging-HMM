"""Train-only counts, UNK mapping and reusable probability caches for TV2."""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .contracts import Corpus, validate_corpus, validate_hmm_config

START = "<s>"
END = "</s>"
CODE_VERSION = "tv2-hmm-1"


class _SpecialToken(Enum):
    UNK = "<UNK>"


# Unlike the string '<UNK>', this sentinel cannot collide with a real FORM.
UNK = _SpecialToken.UNK
EmissionSymbol = str | _SpecialToken


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def add_alpha_logs(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Normalize each row with Add-alpha, including zero-count contexts.

    Mỗi hàng chứa đủ các phần tử của miền chuẩn hóa, kể cả EOS hoặc UNK.
    Dùng logaddexp để tính log(C + alpha), tránh tràn số khi alpha rất lớn.
    Hàng toàn số 0 trở thành uniform; không cần backoff sang mô hình khác.
    """
    count_logs = np.full(counts.shape, -np.inf, dtype=np.float64)
    np.log(counts, out=count_logs, where=counts > 0)
    totals = counts.sum(axis=1, keepdims=True)
    total_logs = np.full(totals.shape, -np.inf, dtype=np.float64)
    np.log(totals, out=total_logs, where=totals > 0)
    log_alpha = np.log(alpha)
    return _readonly(
        np.logaddexp(count_logs, log_alpha)
        - np.logaddexp(total_logs, log_alpha + np.log(counts.shape[1]))
    )


@dataclass(frozen=True)
class TrainingCounts:
    fingerprint: str
    tags: tuple[str, ...]
    word_counts: Mapping[str, int]
    tag_counts: np.ndarray
    word_tag_counts: Mapping[tuple[str, str], int]
    bigram_counts: np.ndarray  # rows: tags + START; columns: tags + END
    trigram_counts: np.ndarray  # context axes: tags + START; target axis: tags + END


@dataclass(frozen=True)
class EmissionCounts:
    words: tuple[str, ...]  # Retained FORM strings, without the UNK sentinel.
    word_to_index: Mapping[EmissionSymbol, int]
    unk_map: Mapping[str, EmissionSymbol]
    counts: np.ndarray  # UNK is always the final column.


@dataclass(frozen=True)
class HMMProbabilities:
    transition: np.ndarray
    emission: np.ndarray


class HMMTrainingCache:
    """Explicitly share this object across models to reuse train-only work.

    Cache keys include a SHA-256 of the full ordered training corpus; another
    corpus cannot silently reuse its vocabulary or probabilities. Cached arrays
    and mappings are read-only. No global cache or dev/test state is used.
    """

    def __init__(self) -> None:
        self._training: dict[str, TrainingCounts] = {}
        self._emissions: dict[tuple[str, int], EmissionCounts] = {}
        self._bigram: dict[tuple[str, float, int], HMMProbabilities] = {}
        self._trigram: dict[tuple[str, float, int], HMMProbabilities] = {}
        self._emission_logs: dict[tuple[str, float, int], np.ndarray] = {}

    def training(self, train_data: Corpus, *, use_cache: bool) -> TrainingCounts:
        """Đếm một lần trên các câu train, reset START/EOS ở từng câu.

        Chỉ lấy các UPOS thực sự xuất hiện và sắp xếp tên để cố định chỉ số.
        Mỗi câu không rỗng đóng góp START -> y1 và yn -> END đúng một lần.
        Bỏ câu rỗng; raw vocabulary và counts chưa áp dụng cutoff.
        """
        validate_corpus(train_data)
        sentences = tuple((tuple(words), tuple(tags)) for words, tags in train_data if words)
        if not sentences:
            raise ValueError("Training corpus must contain at least one token.")
        encoded = json.dumps(sentences, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        fingerprint = hashlib.sha256(encoded).hexdigest()
        if use_cache and fingerprint in self._training:
            return self._training[fingerprint]
        word_counts: Counter[str] = Counter()
        tag_counts: Counter[str] = Counter()
        word_tag_counts: Counter[tuple[str, str]] = Counter()
        transitions: Counter[tuple[str, str]] = Counter()
        triples: Counter[tuple[str, str, str]] = Counter()
        for words, sentence_tags in sentences:
            earlier = previous = START
            for word, tag in zip(words, sentence_tags):
                word_counts[word] += 1
                tag_counts[tag] += 1
                word_tag_counts[word, tag] += 1
                transitions[previous, tag] += 1
                triples[earlier, previous, tag] += 1
                earlier, previous = previous, tag
            transitions[previous, END] += 1
            triples[earlier, previous, END] += 1
        tags = tuple(sorted(tag_counts))
        tag_to_index = {tag: index for index, tag in enumerate(tags)}
        boundary = len(tags)
        bigram_counts = np.zeros((boundary + 1, boundary + 1), dtype=np.int64)
        trigram_counts = np.zeros((boundary + 1,) * 3, dtype=np.int64)
        for (previous, following), count in transitions.items():
            row = boundary if previous == START else tag_to_index[previous]
            column = boundary if following == END else tag_to_index[following]
            bigram_counts[row, column] = count
        # Hai START chỉ là lịch sử. Mỗi câu thêm đúng một EOS sau nhãn cuối;
        # câu một token kết thúc tại lịch sử (START, tag), không cần nhãn giả.
        # Chỉ đếm theo token/câu, không có ba vòng lặp qua mọi tổ hợp nhãn.
        for (earlier, previous, following), count in triples.items():
            first = boundary if earlier == START else tag_to_index[earlier]
            second = boundary if previous == START else tag_to_index[previous]
            target = boundary if following == END else tag_to_index[following]
            trigram_counts[first, second, target] = count
        result = TrainingCounts(
            fingerprint, tags, MappingProxyType(dict(word_counts)),
            _readonly(np.array([tag_counts[tag] for tag in tags], dtype=np.int64)),
            MappingProxyType(dict(word_tag_counts)), _readonly(bigram_counts), _readonly(trigram_counts),
        )
        if use_cache:
            self._training[fingerprint] = result
        return result

    def emissions(self, training: TrainingCounts, cutoff: int, *, use_cache: bool) -> EmissionCounts:
        key = (training.fingerprint, cutoff)
        if use_cache and key in self._emissions:
            return self._emissions[key]
        words = tuple(sorted(word for word, count in training.word_counts.items() if count > cutoff))
        word_to_index: dict[EmissionSymbol, int] = {word: index for index, word in enumerate(words)}
        unk_index = len(words)
        word_to_index[UNK] = unk_index
        unk_map = {word: word if count > cutoff else UNK
                   for word, count in training.word_counts.items()}
        tag_to_index = {tag: index for index, tag in enumerate(training.tags)}
        counts = np.zeros((len(training.tags), len(words) + 1), dtype=np.int64)
        for (word, tag), count in training.word_tag_counts.items():
            counts[tag_to_index[tag], word_to_index.get(word, unk_index)] += count
        result = EmissionCounts(words, MappingProxyType(word_to_index),
                                MappingProxyType(unk_map), _readonly(counts))
        if use_cache:
            self._emissions[key] = result
        return result

    def bigram_probabilities(
        self, training: TrainingCounts, emissions: EmissionCounts,
        alpha: float, cutoff: int, *, use_cache: bool,
    ) -> HMMProbabilities:
        alpha, cutoff = validate_hmm_config(alpha, cutoff)
        key = (training.fingerprint, alpha, cutoff)
        if use_cache and key in self._bigram:
            return self._bigram[key]
        result = HMMProbabilities(add_alpha_logs(training.bigram_counts, alpha),
                                 self.emission_logs(training, emissions, alpha, cutoff, use_cache=use_cache))
        if use_cache:
            self._bigram[key] = result
        return result

    def emission_logs(
        self, training: TrainingCounts, emissions: EmissionCounts,
        alpha: float, cutoff: int, *, use_cache: bool,
    ) -> np.ndarray:
        """Reuse the exact same emission log table across Bigram and Trigram."""
        key = (training.fingerprint, alpha, cutoff)
        if use_cache and key in self._emission_logs:
            return self._emission_logs[key]
        result = add_alpha_logs(emissions.counts, alpha)
        if use_cache:
            self._emission_logs[key] = result
        return result

    def trigram_probabilities(
        self, training: TrainingCounts, emissions: EmissionCounts,
        alpha: float, cutoff: int, *, use_cache: bool,
    ) -> HMMProbabilities:
        """Add-alpha for every context, including unseen pairs -> uniform.

        Làm phẳng hai trục lịch sử thành các hàng độc lập để chuẩn hóa.
        Mỗi hàng vẫn có đủ |S|+1 đích, gồm EOS; reshape lại tensor ba chiều.
        Không backoff về Bigram khi count của lịch sử bằng 0.
        """
        alpha, cutoff = validate_hmm_config(alpha, cutoff)
        key = (training.fingerprint, alpha, cutoff)
        if use_cache and key in self._trigram:
            return self._trigram[key]
        counts = training.trigram_counts
        transition = _readonly(add_alpha_logs(counts.reshape(-1, counts.shape[-1]), alpha).reshape(counts.shape))
        result = HMMProbabilities(transition, self.emission_logs(
            training, emissions, alpha, cutoff, use_cache=use_cache,
        ))
        if use_cache:
            self._trigram[key] = result
        return result
