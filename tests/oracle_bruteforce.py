"""Oracle toy độc lập: bảng xác suất cho trước, không import src hay decoder."""

from itertools import product
from math import isclose, log
from typing import Callable

LABELS = ("A", "B", "C")
TOKENS = ("x", "y", "z")
START = "<s>"
END = "</s>"

PI = {"A": 0.4, "B": 0.4, "C": 0.2}
Q1 = {
    "A": {"A": 0.3, "B": 0.3, "C": 0.1, END: 0.3},
    "B": {"A": 0.3, "B": 0.3, "C": 0.1, END: 0.3},
    "C": {"A": 0.2, "B": 0.2, "C": 0.4, END: 0.2},
}
# Mỗi hàng có đích A/B/C/EOS; hai lịch sử start chỉ xuất hiện lúc đầu câu.
# Hàng start của q2 khác PI vì q2 còn dành xác suất cho EOS.
Q2 = {
    (START, START): {"A": 0.4, "B": 0.4, "C": 0.1, END: 0.1},
    (START, "A"): {"A": 0.3, "B": 0.3, "C": 0.1, END: 0.3},
    (START, "B"): {"A": 0.3, "B": 0.3, "C": 0.1, END: 0.3},
    (START, "C"): {"A": 0.2, "B": 0.2, "C": 0.4, END: 0.2},
    ("A", "A"): {"A": 0.3, "B": 0.3, "C": 0.1, END: 0.3},
    ("A", "B"): {"A": 0.3, "B": 0.3, "C": 0.1, END: 0.3},
    ("A", "C"): {"A": 0.2, "B": 0.2, "C": 0.4, END: 0.2},
    ("B", "A"): {"A": 0.3, "B": 0.3, "C": 0.1, END: 0.3},
    ("B", "B"): {"A": 0.3, "B": 0.3, "C": 0.1, END: 0.3},
    ("B", "C"): {"A": 0.25, "B": 0.25, "C": 0.25, END: 0.25},
    ("C", "A"): {"A": 0.2, "B": 0.2, "C": 0.4, END: 0.2},
    ("C", "B"): {"A": 0.2, "B": 0.2, "C": 0.4, END: 0.2},
    ("C", "C"): {"A": 0.2, "B": 0.2, "C": 0.4, END: 0.2},
}
E = {
    "A": {"x": 0.5, "y": 0.3, "z": 0.2},
    "B": {"x": 0.5, "y": 0.3, "z": 0.2},
    "C": {"x": 0.2, "y": 0.3, "z": 0.5},
}

Sequence = tuple[str, ...]
OracleResult = tuple[list[tuple[Sequence, float]], float, set[Sequence]]


def _validate(seq: Sequence) -> None:
    """Chỉ nhận chuỗi ba nhãn thuộc toy A/B/C."""
    if len(seq) != len(TOKENS) or any(tag not in LABELS for tag in seq):
        raise ValueError("Cần đúng 3 nhãn thuộc A/B/C.")


def bigram_score(seq: Sequence) -> float:
    """Tính trực tiếp log-joint của chuỗi nhãn và ba token, có EOS."""
    _validate(seq)
    score = log(PI[seq[0]])
    for index, tag in enumerate(seq):
        score += log(E[tag][TOKENS[index]])
        if index > 0:
            score += log(Q1[seq[index - 1]][tag])
    return score + log(Q1[seq[-1]][END])


def trigram_score(seq: Sequence) -> float:
    """Tính trực tiếp từ hai start và từng transition q2, bao gồm EOS."""
    _validate(seq)
    first, second = START, START
    score = 0.0
    for index, tag in enumerate(seq):
        score += log(Q2[first, second][tag])
        score += log(E[tag][TOKENS[index]])
        first, second = second, tag
    return score + log(Q2[first, second][END])


def enumerate_sequences(score_fn: Callable[[Sequence], float], *, atol: float = 1e-12) -> OracleResult:
    """Vét cạn; hòa nếu lệch log-score ≤ atol so với max cuối, rel_tol=0."""
    if not 0 <= atol < float("inf"):
        raise ValueError("atol phải hữu hạn và không âm.")
    scored: list[tuple[Sequence, float]] = []
    best_score = float("-inf")
    for seq in product(LABELS, repeat=len(TOKENS)):
        score = score_fn(seq)
        scored.append((seq, score))
        if score > best_score:
            best_score = score
    winners: set[Sequence] = set()
    for seq, score in scored:
        if isclose(score, best_score, rel_tol=0.0, abs_tol=atol):
            winners.add(seq)
    return scored, best_score, winners


def brute_force_bigram(*, atol: float = 1e-12) -> OracleResult:
    """Trả đủ 27 score, score tốt nhất và tất cả nghiệm hòa của Bigram."""
    return enumerate_sequences(bigram_score, atol=atol)


def brute_force_trigram(*, atol: float = 1e-12) -> OracleResult:
    """Trả đủ 27 score, score tốt nhất và tất cả nghiệm hòa của Trigram."""
    return enumerate_sequences(trigram_score, atol=atol)
