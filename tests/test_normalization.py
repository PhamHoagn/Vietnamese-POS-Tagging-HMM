"""Kiểm tra phân phối qua adapter; chưa xác nhận HMM khi API TV2 còn thiếu."""

from copy import deepcopy
from math import isfinite

import pytest

import adapters
from oracle_bruteforce import E, END, LABELS, PI, Q1, Q2, START, TOKENS


def toy_tables() -> adapters.ToyTables:
    """Sao chép bảng oracle để mỗi test không sửa bảng dùng chung."""
    return deepcopy(dict(labels=LABELS, tokens=TOKENS, pi=PI, q1=Q1, q2=Q2, e=E))


pending_api = pytest.mark.xfail(raises=NotImplementedError, strict=True,
                               reason="Chờ TV2 và cài tests/adapters.py; không phải HMM đã pass")


@pytest.mark.parametrize("order", [1, 2])
@pending_api
def test_transition_normalization(order: int) -> None:
    """Mỗi hàng có đủ |S|+1 đích, gồm EOS, và giữ đúng xác suất cho trước."""
    tables = toy_tables()
    model = adapters.build_toy_model(order, tables)
    histories = [(tag,) for tag in LABELS] if order == 1 else list(Q2)
    for history in histories:
        row = adapters.get_transition_row(model, history)
        expected = Q1[history[0]] if order == 1 else Q2[history]
        assert set(row) == {*LABELS, END}
        assert all(isfinite(p) and 0 <= p <= 1 for p in row.values())
        assert sum(row.values()) == pytest.approx(1, abs=1e-12)
        assert row == pytest.approx(expected)
        row[END] = -1
        assert adapters.get_transition_row(model, history) == pytest.approx(expected)


@pytest.mark.parametrize("order", [1, 2])
@pending_api
def test_emission_normalization(order: int) -> None:
    """Emission đủ vocabulary, không thêm start/end; bản sao không sửa model."""
    model = adapters.build_toy_model(order, toy_tables())
    for tag in LABELS:
        row = adapters.get_emission_row(model, tag)
        assert set(row) == set(TOKENS)
        assert all(isfinite(p) and 0 <= p <= 1 for p in row.values())
        assert sum(row.values()) == pytest.approx(1, abs=1e-12)
        assert row == pytest.approx(E[tag])
        row['x'] = -1
        assert adapters.get_emission_row(model, tag) == pytest.approx(E[tag])


@pytest.mark.parametrize("order", [1, 2])
@pending_api
def test_uniform_and_unk_rows(order: int) -> None:
    """Hàng uniform 1/4 và UNK phải được giữ nguyên, không bỏ EOS/UNK."""
    tables = toy_tables()
    tables['tokens'] = (*TOKENS, '<UNK>')
    tables['e'] = {tag: {'x': .4, 'y': .2, 'z': .3, '<UNK>': .1} for tag in LABELS}
    tables['q1']['C'] = {tag: .25 for tag in (*LABELS, END)}
    tables['q2']['C', 'C'] = {tag: .25 for tag in (*LABELS, END)}
    model = adapters.build_toy_model(order, tables)
    history = ('C',) if order == 1 else ('C', 'C')
    assert adapters.get_transition_row(model, history) == pytest.approx({tag: .25 for tag in (*LABELS, END)})
    for tag in LABELS:
        assert adapters.get_emission_row(model, tag) == pytest.approx(tables['e'][tag])
