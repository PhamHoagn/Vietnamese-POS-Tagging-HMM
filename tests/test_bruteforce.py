"""Kiểm tra oracle bằng xác suất tính tay, không dùng code src/decoder."""

from itertools import product
from math import log
from typing import Callable

import pytest

from tests.oracle_bruteforce import (
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


@pytest.mark.parametrize('order', [1, 2])
def test_decoder_matches_oracle(order: int) -> None:
    """Decoder TV2 phải đạt score oracle và trả một trong mọi nghiệm hòa."""
    from tests import adapters
    from tests.test_normalization import toy_tables

    model = adapters.build_toy_model(order, toy_tables())
    _, best, winners = (brute_force_bigram if order == 1 else brute_force_trigram)()
    tags, score = adapters.decode(model, list(TOKENS))
    assert tuple(tags) in winners
    assert score == pytest.approx(best, abs=1e-12)


def random_cases() -> list[tuple[dict, tuple[str, ...]]]:
    """Sinh 200 bộ bảng dương chuẩn hóa, seed 42 cố định, không dùng RNG toàn cục."""
    import numpy as np

    rng = np.random.default_rng(42)
    cases = []

    def distribution(keys: tuple[str, ...]) -> dict[str, float]:
        # Trọng số trong [0.01,1) nên xác suất luôn >0, tránh log(0).
        weights = rng.uniform(.01, 1.0, size=len(keys))
        weights /= weights.sum()
        return {key: float(value) for key, value in zip(keys, weights)}

    for _ in range(200):
        labels = ('A', 'B', 'C')[:int(rng.integers(2, 4))]
        tokens = tuple(str(x) for x in rng.choice(('x', 'y', 'z'), size=int(rng.integers(1, 5))))
        targets = (*labels, END)
        tables = dict(labels=labels, tokens=('x', 'y', 'z'), pi=distribution(labels),
                      q1={tag: distribution(targets) for tag in labels},
                      q2={history: distribution(targets) for history in product((*labels, START), repeat=2)},
                      e={tag: distribution(('x', 'y', 'z')) for tag in labels})
        cases.append((tables, tokens))
    return cases


RANDOM_CASES = random_cases()


@pytest.mark.parametrize('case_id', range(200), ids=lambda i: f'seed42-case{i:03d}')
@pytest.mark.parametrize('order', [1, 2], ids=['bigram', 'trigram'])
def test_random_decoder_against_oracle(order: int, case_id: int) -> None:
    """Chặn sai argmax, log-score hoặc độ dài trên 200 HMM cho mỗi bậc."""
    from copy import deepcopy
    from math import isfinite
    from tests import adapters
    from tests.oracle_bruteforce import brute_force

    tables, tokens = deepcopy(RANDOM_CASES[case_id])
    _, best, winners = brute_force(order, tokens, tables['labels'], tables['pi'],
                                   tables['q1'], tables['q2'], tables['e'])
    model = adapters.build_toy_model(order, tables)
    tags, score = adapters.decode(model, list(tokens))
    assert len(tags) == len(tokens)
    assert tuple(tags) in winners
    assert isfinite(score) and abs(score - best) < 1e-9


@pytest.mark.parametrize('order', [1, 2])
def test_deliberate_tie_accepts_any_argmax(order: int) -> None:
    """Tie-break: chấp nhận BẤT KỲ nghiệm argmax, không ép nhãn đầu hay thứ tự từ điển."""
    from tests import adapters
    from tests.oracle_bruteforce import brute_force

    labels, tokens = ('B', 'A'), ('x', 'x', 'x')
    targets = (*labels, END)
    tables = dict(labels=labels, tokens=('x',), pi=dict.fromkeys(labels, .5),
                  q1={tag: dict.fromkeys(targets, 1 / 3) for tag in labels},
                  q2={h: dict.fromkeys(targets, 1 / 3) for h in product((*labels, START), repeat=2)},
                  e={tag: {'x': 1.0} for tag in labels})
    _, best, winners = brute_force(order, tokens, labels, tables['pi'], tables['q1'], tables['q2'], tables['e'])
    # Bigram: .5*(1/3)^3; Trigram: (1/3)^4. Cả 2^3=8 đường đi hòa.
    assert winners == set(product(labels, repeat=3))
    assert best == pytest.approx(log(.5 * (1 / 3)**3 if order == 1 else (1 / 3)**4))
    model = adapters.build_toy_model(order, tables)
    tags, score = adapters.decode(model, list(tokens))
    assert tuple(tags) in winners
    assert len(tags) == 3
    assert abs(score - best) < 1e-9


def test_random_generator_coverage_and_normalization() -> None:
    """Chặn fixture thiếu kích thước mong muốn, không chuẩn hóa hoặc không tái lập."""
    assert RANDOM_CASES == random_cases()
    assert len(RANDOM_CASES) == 200
    assert {len(t['labels']) for t, _ in RANDOM_CASES} == {2, 3}
    assert {len(tokens) for _, tokens in RANDOM_CASES} == {1, 2, 3, 4}
    for tables, _ in RANDOM_CASES:
        for row in [tables['pi'], *tables['q1'].values(), *tables['q2'].values(), *tables['e'].values()]:
            assert all(0 < p <= 1 for p in row.values())
            assert sum(row.values()) == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize('order', [1, 2])
def test_general_oracle_matches_hand_checked_toy(order: int) -> None:
    """Chặn sai công thức oracle tổng quát bằng toy cố định đã tính tay."""
    from tests.oracle_bruteforce import brute_force

    scored, best, winners = brute_force(order, TOKENS, LABELS, PI, Q1, Q2, E)
    reference, reference_best, reference_winners = (brute_force_bigram if order == 1 else brute_force_trigram)()
    assert dict(scored) == pytest.approx(dict(reference), abs=1e-12)
    assert best == pytest.approx(reference_best, abs=1e-12)
    assert winners == reference_winners
