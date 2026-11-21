"""Kiểm tra oracle bằng xác suất tính tay, không dùng code src/decoder."""

from itertools import product
from math import log
from typing import Callable

import pytest

from oracle_bruteforce import (
    E, END, LABELS, PI, Q1, Q2, START, TOKENS, OracleResult,
    bigram_score, trigram_score, brute_force_bigram, brute_force_trigram,
)


def test_probability_tables_normalized() -> None:
    """Mỗi hàng là phân phối dương, transition có đủ A/B/C/EOS."""
    assert set(PI) == set(LABELS)
    assert set(Q1) == set(LABELS)
    assert set(Q2) == {(START, START), *((START, tag) for tag in LABELS), *product(LABELS, repeat=2)}
    for row in [PI, *Q1.values(), *Q2.values(), *E.values()]:
        assert all(0 < p <= 1 for p in row.values())
        assert sum(row.values()) == pytest.approx(1, abs=1e-12)
    for row in [*Q1.values(), *Q2.values()]:
        assert set(row) == {*LABELS, END}
    for row in E.values():
        assert set(row) == set(TOKENS)


def test_manual_scores_include_end_and_second_order() -> None:
    """ABC: q2(EOS|B,C)=.25 khác q1(EOS|C)=.2, không được bỏ EOS."""
    assert bigram_score(("A", "B", "C")) == pytest.approx(log(.4 * .5 * .3 * .3 * .1 * .5 * .2))
    assert trigram_score(("A", "B", "C")) == pytest.approx(log(.4 * .5 * .3 * .3 * .1 * .5 * .25))
    assert trigram_score(("C", "C", "C")) == pytest.approx(log(.1 * .2 * .4 * .3 * .4 * .5 * .2))


@pytest.mark.parametrize("oracle", [brute_force_bigram, brute_force_trigram])
def test_exhaustive_results_and_all_ties(oracle: Callable[[], OracleResult]) -> None:
    """Tám chuỗi chỉ chứa A/B cùng đạt .000324; không được chỉ trả một nghiệm."""
    scored, best, winners = oracle()
    assert len(scored) == 27
    assert {seq for seq, _ in scored} == set(product(LABELS, repeat=3))
    assert best == pytest.approx(log(.4 * .5 * .3 * .3 * .3 * .2 * .3))
    assert winners == set(product(("A", "B"), repeat=3))
    assert all(score <= best for _, score in scored)


@pytest.mark.parametrize("seq", [(), ("A",), ("A", "B", "X")])
def test_invalid_sequence(seq: tuple[str, ...]) -> None:
    """Oracle toy chỉ định nghĩa cho đúng ba token và nhãn A/B/C."""
    for score in (bigram_score, trigram_score):
        with pytest.raises(ValueError):
            score(seq)
