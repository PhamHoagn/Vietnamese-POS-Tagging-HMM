"""Supervised Bigram HMM with Add-alpha, train-only UNK and NumPy Viterbi."""

import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Self

import numpy as np

from .contracts import Corpus, UPOS_TAGS, validate_hmm_config, validate_tokens
from .hmm_common import CODE_VERSION, END, START, UNK, EmissionSymbol, HMMTrainingCache


def _validate_words(words: list[str]) -> None:
    if not isinstance(words, list):
        raise TypeError("Provide a list of pre-tokenized words, not raw text.")
    validate_tokens(words)


def _log_probability(count: int, total: int, alpha: float, size: int) -> float:
    """Scalar count-based Add-alpha; independent of decoder probability tables."""
    log_alpha = math.log(alpha)
    numerator = np.logaddexp(math.log(count) if count else -math.inf, log_alpha)
    denominator = np.logaddexp(math.log(total) if total else -math.inf,
                               log_alpha + math.log(size))
    return float(numerator - denominator)


class BigramHMMTagger:
    """Repository API: constructor config, fit(Corpus), predict(tokens).

    Tags are sorted UPOS observed in train. Transition support is S + END;
    emission support is retained FORM + UNK, with one shared alpha. Empty
    prediction returns [] for compatibility with the common project contract.
    """

    def __init__(
        self, alpha: float = 0.1, cutoff: int = 1, *,
        cache: HMMTrainingCache | None = None, use_cache: bool = True,
    ) -> None:
        self.alpha, self.cutoff = validate_hmm_config(alpha, cutoff)
        if not isinstance(use_cache, bool):
            raise ValueError("use_cache must be bool.")
        self.cache = cache if cache is not None else HMMTrainingCache()
        self.use_cache = use_cache

    def fit(self, train_data: Corpus) -> Self:
        return self._fit(train_data, self.alpha, self.cutoff)

    def _fit(self, train_data: Corpus, alpha: float, cutoff: int) -> Self:
        alpha, cutoff = validate_hmm_config(alpha, cutoff)
        training = self.cache.training(train_data, use_cache=self.use_cache)
        expected = getattr(self, "_expected_config", None)
        if expected is not None and (
            alpha != expected["alpha"] or cutoff != expected["cutoff"]
            or list(training.tags) != expected["tags"]
            or training.fingerprint != expected["training_sha256"]
        ):
            raise ValueError("Training data/config does not match the saved configuration.")
        emissions = self.cache.emissions(training, cutoff, use_cache=self.use_cache)
        probabilities = self.cache.bigram_probabilities(
            training, emissions, alpha, cutoff, use_cache=self.use_cache,
        )
        # Publish only after successful preparation, preserving a previous fit on error.
        self.alpha, self.cutoff = alpha, cutoff
        self.tags = training.tags
        self.tag_to_index_ = MappingProxyType({tag: index for index, tag in enumerate(self.tags)})
        self.word_counts_ = training.word_counts
        self.tag_counts_ = training.tag_counts
        self.train_vocab_ = frozenset(training.word_counts)
        self.retained_vocab_ = frozenset(emissions.words)
        self.vocab_ = self.retained_vocab_ | {UNK}
        self.word_to_index_ = emissions.word_to_index
        self.unk_map_ = emissions.unk_map
        self.unk_index_ = len(emissions.words)
        self.training_fingerprint_ = training.fingerprint
        self.transition_counts_ = training.bigram_counts
        self.emission_counts_ = emissions.counts
        self.transition_log_probs_ = probabilities.transition
        self.emission_log_probs_ = probabilities.emission
        self._fitted_parameters = (alpha, cutoff)
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "_fitted_parameters"):
            raise RuntimeError("BigramHMM must be fit before prediction/scoring.")
        if (self.alpha, self.cutoff) != self._fitted_parameters:
            raise RuntimeError("Configuration changed after fit; call fit again.")

    def map_token(self, word: str) -> EmissionSymbol:
        self._require_fitted()
        validate_tokens([word])
        return word if word in self.retained_vocab_ else UNK

    def _target_index(self, tag: str) -> int:
        if tag == END:
            return len(self.tags)
        if tag not in self.tag_to_index_:
            raise ValueError(f"Tag {tag!r} is not in the training tagset.")
        return self.tag_to_index_[tag]

    def _transition_log(self, previous: str, following: str) -> float:
        column = self._target_index(following)
        if previous == END or not isinstance(previous, str) or not previous:
            raise ValueError("A transition context must be a non-empty tag or START, never END.")
        row = len(self.tags) if previous == START else self.tag_to_index_.get(previous)
        if row is None:
            return -math.log(len(self.tags) + 1)  # Unobserved context: uniform Add-alpha.
        counts = self.transition_counts_[row]
        return _log_probability(int(counts[column]), int(counts.sum()), self.alpha, len(self.tags) + 1)

    def transition_probability(self, previous: str, following: str) -> float:
        """q(following | previous); an unseen context is uniform over S + END."""
        self._require_fitted()
        return math.exp(self._transition_log(previous, following))

    def _emission_log(self, word: EmissionSymbol, tag: str) -> float:
        row = self._target_index(tag)
        if row == len(self.tags):
            raise ValueError("END has no emission distribution.")
        if word is UNK:
            column = self.unk_index_
        else:
            validate_tokens([word])
            column = self.word_to_index_.get(word, self.unk_index_)
        return _log_probability(int(self.emission_counts_[row, column]), int(self.tag_counts_[row]),
                                self.alpha, self.unk_index_ + 1)

    def emission_probability(self, word: EmissionSymbol, tag: str) -> float:
        """e(map_cutoff(word) | tag), accepting the explicit UNK sentinel too."""
        self._require_fitted()
        return math.exp(self._emission_log(word, tag))

    def predict(self, tokens: list[str]) -> list[str]:
        self._require_fitted()
        _validate_words(tokens)
        if not tokens:
            return []
        indices = [self.word_to_index_.get(word, self.unk_index_) for word in tokens]
        # Gather the full sentence before the time loop; all logs were prepared in fit.
        emissions = np.ascontiguousarray(self.emission_log_probs_[:, indices].T)
        return self._viterbi(emissions)

    def _viterbi(self, emissions: np.ndarray) -> list[str]:
        """Viterbi Bigram trong log-space, O(T*S^2), backpointer O(T*S).

        Mỗi cột v chọn tiền nhiệm u qua ma trận score[u,v], không nhân xác suất.
        Rank lưu thứ tự từ điển của prefix tối ưu; dùng rank để xử lý hòa điểm
        trên cả chuỗi, thay vì chỉ chọn nhãn cuối nhỏ nhất. Cộng EOS trước truy vết.
        """
        length, size = emissions.shape
        transitions = self.transition_log_probs_
        scores = transitions[size, :size] + emissions[0]
        ranks = np.arange(size)
        backpointers = np.full((length, size), -1, dtype=np.intp)
        for time in range(1, length):
            candidates = scores[:, None] + transitions[:size, :size]
            best = candidates.max(axis=0)
            previous = np.argmin(np.where(candidates == best, ranks[:, None], size), axis=0)
            backpointers[time] = previous
            scores = best + emissions[time]
            order = np.lexsort((np.arange(size), ranks[previous]))
            ranks = np.empty(size, dtype=np.intp)
            ranks[order] = np.arange(size)
        final_scores = scores + transitions[:size, size]
        current = int(np.argmin(np.where(final_scores == final_scores.max(), ranks, size)))
        path = np.empty(length, dtype=np.intp)
        path[-1] = current
        # Mỗi bước lùi đọc tiền nhiệm đã lưu, không tính lại hoặc tham lam chọn nhãn.
        # Đi ngược T-1 bước rồi ánh xạ chỉ số về UPOS, bảo toàn đúng số token.
        for time in range(length - 1, 0, -1):
            current = int(backpointers[time, current])
            path[time - 1] = current
        return [self.tags[index] for index in path]

    def predict_corpus(self, sentences: list[list[str]]) -> list[list[str]]:
        self._require_fitted()
        return [self.predict(words) for words in sentences]

    def log_joint(self, words: list[str], tags: list[str]) -> float:
        """Score a supplied path directly from counts, independently of Viterbi.

        Cộng log q(y_i|y_(i-1)) và log e(x_i|y_i) từ các count đã học.
        Bắt đầu ở START, cộng q(END|y_T) đúng một lần sau token cuối.
        Không gọi decoder hoặc đọc bảng log-prob của decoder; câu rỗng bị từ chối.
        """
        self._require_fitted()
        _validate_words(words)
        if not words or len(words) != len(tags):
            raise ValueError("Scoring requires non-empty, aligned words and tags.")
        if any(tag not in self.tag_to_index_ for tag in tags):
            raise ValueError("Scored tags must belong to the training tagset.")
        result = 0.0
        previous = START
        for word, tag in zip(words, tags):
            result += self._transition_log(previous, tag)
            result += self._emission_log(word, tag)
            previous = tag
        return result + self._transition_log(previous, END)

    def save_config(self, path: str | Path) -> None:
        """Save locked metadata only, not weights; load_config requires refitting."""
        self._require_fitted()
        config = {
            "schema_version": 1, "model": "bigram", "code_version": CODE_VERSION,
            "alpha": self.alpha, "cutoff": self.cutoff, "tags": list(self.tags),
            "training_sha256": self.training_fingerprint_,
        }
        Path(path).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_config(
        cls, path: str | Path, *, cache: HMMTrainingCache | None = None, use_cache: bool = True,
    ) -> Self:
        """Load config into an unfitted model; fit verifies training hash and tags."""
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"schema_version", "model", "code_version", "alpha", "cutoff", "tags", "training_sha256"}
        if (not isinstance(config, dict) or set(config) != required
                or config["schema_version"] != 1 or config["model"] != "bigram"
                or config["code_version"] != CODE_VERSION):
            raise ValueError("Unsupported Bigram configuration/schema/code version.")
        tags = config["tags"]
        fingerprint = config["training_sha256"]
        if (not isinstance(tags, list) or not tags or any(tag not in UPOS_TAGS for tag in tags)
                or tags != sorted(set(tags)) or not isinstance(fingerprint, str)
                or len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint)):
            raise ValueError("Invalid saved tag order or training SHA-256.")
        model = cls(config["alpha"], config["cutoff"], cache=cache, use_cache=use_cache)
        model._expected_config = config
        return model


class BigramHMM(BigramHMMTagger):
    """New TV2 API using the existing parser's Corpus, without changing TV1 code.

    fit(train_sentences, alpha, cutoff) accepts positional or keyword config.
    Unlike the compatibility Tagger, predict([]) raises a clear ValueError.
    """

    def fit(self, train_sentences: Corpus, alpha: float | None = None, cutoff: int | None = None) -> Self:
        return self._fit(train_sentences, self.alpha if alpha is None else alpha,
                         self.cutoff if cutoff is None else cutoff)

    def predict(self, words: list[str]) -> list[str]:
        self._require_fitted()
        if not words:
            raise ValueError("BigramHMM.predict requires at least one token.")
        return super().predict(words)
