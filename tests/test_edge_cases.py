"""Các ca biên decoder chỉ gọi TV2 qua adapter, không import src."""

from itertools import product
from math import isfinite, log

import pytest

import adapters
from oracle_bruteforce import END, LABELS, START
from test_normalization import pending_api, toy_tables


def direct_score(order: int, tags: tuple[str, ...], tokens: list[str], tables: adapters.ToyTables) -> float:
    """Nhân xác suất cho đường đi cố định; không quy hoạch động hay gọi decoder."""
    score = 0.0
    history = (START, START)
    for i, tag in enumerate(tags):
        if order == 1:
            p = tables['pi'][tag] if i == 0 else tables['q1'][tags[i - 1]][tag]
        else:
            p = tables['q2'][history][tag]
        score += log(p) + log(tables['e'][tag][tokens[i]])
        history = history[1], tag
    end = tables['q1'][tags[-1]][END] if order == 1 else tables['q2'][history][END]
    return score + log(end)


@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('tokens', [['x'], ['x', 'y'], ['x', 'y', 'z']])
@pending_api
def test_short_sequences_against_enumeration(order: int, tokens: list[str]) -> None:
    """Câu 1/2/3 token phải đạt max vét cạn và score ứng với đường đi trả về."""
    tables = toy_tables()
    scores = [(seq, direct_score(order, seq, tokens, tables)) for seq in product(LABELS, repeat=len(tokens))]
    best = max(score for _, score in scores)
    model = adapters.build_toy_model(order, tables)
    tags, score = adapters.decode(model, tokens)
    assert len(tags) == len(tokens)
    assert tuple(tags) in {seq for seq, value in scores if abs(value - best) <= 1e-12}
    assert score == pytest.approx(best, abs=1e-12)
    assert score == pytest.approx(direct_score(order, tuple(tags), tokens, tables), abs=1e-12)


@pytest.mark.parametrize('order', [1, 2])
@pending_api
def test_long_sequence_repeatable(order: int) -> None:
    """Câu dài không underflow, nhãn đủ độ dài và chạy lại cho cùng kết quả."""
    tables = toy_tables()
    tokens = ['x', 'y', 'z'] * 100
    model = adapters.build_toy_model(order, tables)
    tags, score = adapters.decode(model, tokens)
    assert len(tags) == len(tokens) and set(tags) <= set(LABELS)
    assert isfinite(score)
    assert (tags, score) == adapters.decode(model, tokens)
    assert score == pytest.approx(direct_score(order, tuple(tags), tokens, tables), abs=1e-10)


@pytest.mark.parametrize('order', [1, 2])
@pending_api
def test_oov_with_and_without_unk(order: int) -> None:
    """Toy thiếu UNK báo lỗi; khi có UNK, từ lạ tương đương token UNK."""
    tables = toy_tables()
    model = adapters.build_toy_model(order, tables)
    with pytest.raises(ValueError):
        adapters.decode(model, ['unseen'])
    tables['tokens'] = ('x', 'y', 'z', '<UNK>')
    tables['e'] = {tag: {'x': .4, 'y': .2, 'z': .3, '<UNK>': .1} for tag in LABELS}
    model = adapters.build_toy_model(order, tables)
    assert adapters.decode(model, ['unseen']) == adapters.decode(model, ['<UNK>'])


@pytest.mark.parametrize('order', [1, 2])
@pending_api
def test_empty_sequence_contract(order: int) -> None:
    """Trigram rỗng đi thẳng EOS; pi của Bigram toy không định nghĩa câu rỗng."""
    tables = toy_tables()
    model = adapters.build_toy_model(order, tables)
    if order == 1:
        with pytest.raises(ValueError):
            adapters.decode(model, [])
    else:
        tags, score = adapters.decode(model, [])
        assert tags == []
        assert score == pytest.approx(log(tables['q2'][START, START][END]))
