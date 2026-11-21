"""Second-order HMM: NumPy pair-state Viterbi, two STARTs and one END."""

import math
from typing import Self

import numpy as np

from .contracts import Corpus
from .hmm_bigram import BigramHMMTagger, _log_probability, _validate_words
from .hmm_common import END, START, EmissionCounts, HMMProbabilities, TrainingCounts


class TrigramHMMTagger(BigramHMMTagger):
    """Reuse train-only emissions/config lifecycle; replace transition order.

    Unseen histories receive uniform 1/(|S|+1), exactly as Add-alpha implies;
    they do not back off to Bigram. predict([]) remains [] for the shared API.
    """

    model_kind = "trigram"
    code_version = "tv2-trigram-1"

    def _prepare_model(
        self, training: TrainingCounts, emissions: EmissionCounts, alpha: float, cutoff: int,
    ) -> tuple[np.ndarray, HMMProbabilities]:
        return training.trigram_counts, self.cache.trigram_probabilities(
            training, emissions, alpha, cutoff, use_cache=self.use_cache,
        )

    def _transition_log(self, earlier: str, previous: str, following: str) -> float:
        target = self._target_index(following)
        indices = []
        for context in (earlier, previous):
            if not isinstance(context, str) or not context or context == END:
                raise ValueError("Contexts must be non-empty tags or START, never END.")
            indices.append(len(self.tags) if context == START else self.tag_to_index_.get(context))
        if getattr(self, "_table_mode", False):
            if (earlier, previous) not in self._provided_histories:
                raise KeyError((earlier, previous))
            probability = self._provided_transition_probs[indices[0], indices[1], target]
            return math.log(probability) if probability else -math.inf
        if None in indices:
            return -math.log(len(self.tags) + 1)
        counts = self.transition_counts_[indices[0], indices[1]]
        return _log_probability(int(counts[target]), int(counts.sum()), self.alpha, len(self.tags) + 1)

    def transition_probability(self, earlier: str, previous: str, following: str) -> float:
        """q(following | earlier, previous), including END and unseen contexts."""
        self._require_fitted()
        return math.exp(self._transition_log(earlier, previous, following))

    def _viterbi(self, emissions: np.ndarray) -> list[str]:
        """Viterbi cặp nhãn, thời gian O(T*S^3), backpointer O(T*S^2).

        Khởi tạo hai START; tách riêng câu một token và bước tạo cặp đầu tiên.
        Tensor score[u,v,w] được max/argmax theo u, không dùng ba vòng for Python.
        Rank prefix xử lý hòa điểm toàn chuỗi; cộng EOS trước khi chọn cặp cuối.
        """
        length, size = emissions.shape
        transitions = self.transition_log_probs_
        first = transitions[size, size, :size] + emissions[0]
        if length == 1:
            final = first + transitions[size, :size, size]
            return [self.tags[int(np.argmax(final))]]
        scores = first[:, None] + transitions[size, :size, :size] + emissions[1][None, :]
        ranks = np.arange(size * size).reshape(size, size)
        backpointers = np.full((length, size, size), -1, dtype=np.intp)
        previous_axis = np.arange(size)[:, None]
        last_tags = np.tile(np.arange(size), size)
        for time in range(2, length):
            candidates = scores[:, :, None] + transitions[:size, :size, :size]
            best = candidates.max(axis=0)
            predecessors = np.argmin(
                np.where(candidates == best, ranks[:, :, None], size * size), axis=0,
            )
            backpointers[time] = predecessors
            scores = best + emissions[time][None, :]
            prefix_ranks = ranks[predecessors, previous_axis]
            order = np.lexsort((last_tags, prefix_ranks.ravel()))
            ranks = np.empty((size, size), dtype=np.intp)
            ranks.flat[order] = np.arange(size * size)
        final = scores + transitions[:size, :size, size]
        best_pair = int(np.argmin(np.where(final == final.max(), ranks, size * size)))
        previous, current = np.unravel_index(best_pair, (size, size))
        path = np.empty(length, dtype=np.intp)
        path[-2:] = previous, current
        # Ở bước time, (previous,current) là cặp nhãn đã chọn tại time-1,time.
        # Backpointer lưu earlier; ghi nó vào time-2 rồi dịch cặp lùi một bước.
        # Hai START chỉ phục vụ khởi tạo, không xuất hiện trong path trả về.
        for time in range(length - 1, 1, -1):
            earlier = int(backpointers[time, previous, current])
            path[time - 2] = earlier
            previous, current = earlier, previous
        return [self.tags[index] for index in path]

    def log_joint(self, words: list[str], tags: list[str]) -> float:
        """Tính trực tiếp từ counts, độc lập với tensor Viterbi/log tables.

        Bắt đầu lịch sử (START,START), mỗi token cộng q2 và emission đúng nhãn.
        Dịch hai nhãn lịch sử sau từng token; kết thúc bằng một q2 đến END.
        Câu một token có EOS tại (START,y1); không giả định câu có ít nhất hai từ.
        """
        self._require_fitted()
        _validate_words(words)
        if not words and not tags and getattr(self, "_table_mode", False):
            return self._transition_log(START, START, END)
        if not words or len(words) != len(tags):
            raise ValueError("Scoring requires non-empty, aligned words and tags.")
        if any(tag not in self.tag_to_index_ for tag in tags):
            raise ValueError("Scored tags must belong to the training tagset.")
        earlier = previous = START
        result = 0.0
        for word, tag in zip(words, tags):
            result += self._transition_log(earlier, previous, tag)
            result += self._emission_log(word, tag)
            earlier, previous = previous, tag
        return result + self._transition_log(earlier, previous, END)


class TrigramHMM(TrigramHMMTagger):
    """TV2 API: fit(train_sentences, alpha, cutoff); empty prediction is an error."""

    def fit(self, train_sentences: Corpus, alpha: float | None = None, cutoff: int | None = None) -> Self:
        return self._fit(train_sentences, self.alpha if alpha is None else alpha,
                         self.cutoff if cutoff is None else cutoff)

    def predict(self, words: list[str]) -> list[str]:
        self._require_fitted()
        if not words:
            raise ValueError("TrigramHMM.predict requires at least one token.")
        return super().predict(words)
