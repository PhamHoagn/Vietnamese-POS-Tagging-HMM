"""Types, tagset and validation shared by all project modules."""

from math import isfinite
from numbers import Integral, Real
from typing import Protocol, Self, runtime_checkable

Sentence = tuple[list[str], list[str]]
Corpus = list[Sentence]
UPOS_TAGS: tuple[str, ...] = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)


@runtime_checkable
class POSTagger(Protocol):
    def fit(self, train_data: Corpus) -> Self:
        """Learn only from train_data and return self."""
        ...

    def predict(self, tokens: list[str]) -> list[str]:
        """Return one UPOS tag per token, without receiving gold labels."""
        ...


def validate_tokens(tokens: list[str]) -> None:
    if any(not isinstance(token, str) or not token for token in tokens):
        raise ValueError("Tokens must be non-empty strings.")


def validate_corpus(data: Corpus) -> None:
    for index, (tokens, tags) in enumerate(data):
        if len(tokens) != len(tags):
            raise ValueError(f"Sentence {index}: token/tag lengths differ.")
        validate_tokens(tokens)
        if any(tag not in UPOS_TAGS for tag in tags):
            raise ValueError(f"Sentence {index}: tags must be valid UPOS labels.")


def validate_hmm_config(alpha: float, cutoff: int) -> tuple[float, int]:
    if (isinstance(alpha, bool) or not isinstance(alpha, Real)
            or not isfinite(alpha) or alpha <= 0):
        raise ValueError("alpha must be a finite number > 0.")
    if isinstance(cutoff, bool) or not isinstance(cutoff, Integral) or cutoff < 0:
        raise ValueError("cutoff must be an integer >= 0.")
    return float(alpha), int(cutoff)
