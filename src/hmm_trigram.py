"""Trigram HMM interface; probability estimation and Viterbi are pending."""

from typing import Self

from .contracts import Corpus, validate_hmm_config


class TrigramHMMTagger:
    def __init__(self, alpha: float = 0.1, cutoff: int = 1) -> None:
        self.alpha, self.cutoff = validate_hmm_config(alpha, cutoff)

    def fit(self, train_data: Corpus) -> Self:
        raise NotImplementedError("Trigram HMM training is not implemented (TV2).")

    def predict(self, tokens: list[str]) -> list[str]:
        raise NotImplementedError("Trigram Viterbi decoding is not implemented (TV2).")
