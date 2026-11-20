"""Most-frequent-tag baseline trained exclusively on training sentences."""

from collections import Counter, defaultdict
from typing import Self

from .contracts import Corpus, validate_corpus, validate_tokens


def _most_frequent(counts: Counter[str]) -> str:
    # Resolve ties by ascending tag name, independent of corpus ordering.
    return min(counts, key=lambda tag: (-counts[tag], tag))


class MostFrequentTagger:
    def fit(self, train_data: Corpus) -> Self:
        validate_corpus(train_data)
        word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        tag_counts: Counter[str] = Counter()
        for tokens, tags in train_data:
            for token, tag in zip(tokens, tags):
                word_counts[token][tag] += 1
                tag_counts[tag] += 1
        if not tag_counts:
            raise ValueError("Training corpus must contain at least one token.")
        self.word_tags_ = {word: _most_frequent(counts) for word, counts in word_counts.items()}
        self.default_tag_ = _most_frequent(tag_counts)
        self.train_vocab_ = frozenset(word_counts)
        return self

    def predict(self, tokens: list[str]) -> list[str]:
        if not hasattr(self, "word_tags_"):
            raise RuntimeError("MostFrequentTagger must be fit before predict.")
        validate_tokens(tokens)
        return [self.word_tags_.get(token, self.default_tag_) for token in tokens]
