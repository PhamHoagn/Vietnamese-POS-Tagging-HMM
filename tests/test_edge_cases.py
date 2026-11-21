"""Ca biên HMM qua adapter: score độc lập, thời gian và hash trạng thái train."""

from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
from math import isfinite, log
from time import perf_counter

import pytest

import adapters
from oracle_bruteforce import END, LABELS, START
from test_normalization import TRAIN, toy_tables


def build_model(order: int, tables: adapters.ToyTables) -> object:
    """Báo chờ khi adapter nạp toy chưa cài, không che lỗi assertion."""
    try:
        return adapters.build_toy_model(order, tables)
    except NotImplementedError as exc:
        pytest.xfail(f'Chờ TV2/adapter: {exc}')


def direct_score(order: int, tags: tuple[str, ...], tokens: list[str], tables: adapters.ToyTables) -> float:
    """Tính score cho đường đi cố định từ bảng, không gọi decoder."""
    score, history = 0.0, (START, START)
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
def test_short_sequences_against_enumeration(order: int, tokens: list[str]) -> None:
    """Chặn lỗi khởi tạo/backpointer ở câu 1/2/3 token bằng toàn bộ đường đi vét cạn."""
    tables = toy_tables()
    scores = [(seq, direct_score(order, seq, tokens, tables)) for seq in product(LABELS, repeat=len(tokens))]
    best = max(score for _, score in scores)
    tags, score = adapters.decode(build_model(order, tables), tokens)
    assert len(tags) == len(tokens)
    assert tuple(tags) in {seq for seq, value in scores if abs(value - best) <= 1e-12}
    assert score == pytest.approx(best, abs=1e-12)
    assert score == pytest.approx(direct_score(order, tuple(tags), tokens, tables), abs=1e-12)


@pytest.mark.parametrize('order', [1, 2])
def test_sixty_tokens_under_two_seconds(order: int) -> None:
    """Chặn decoder chậm, thiếu nhãn hoặc underflow trên đúng 60 token."""
    tables, tokens = toy_tables(), ['x', 'y', 'z'] * 20
    model = build_model(order, tables)
    start = perf_counter()
    tags, score = adapters.decode(model, tokens)
    elapsed = perf_counter() - start
    assert elapsed <= 2.0, f'Decode 60 token mất {elapsed:.6f}s > 2s (không tính build).'
    assert len(tags) == len(tokens) == 60 and set(tags) <= set(LABELS)
    assert isfinite(score)
    assert score == pytest.approx(direct_score(order, tuple(tags), tokens, tables), abs=1e-10)


@pytest.mark.parametrize('order', [1, 2])
def test_repeatable_same_input_and_config(order: int) -> None:
    """Chặn trạng thái cache/tie-break không ổn định giữa hai lần chạy và hai model."""
    tables, tokens = toy_tables(), ['x', 'y', 'z']
    before = deepcopy(tables), list(tokens)
    model = build_model(order, tables)
    first = adapters.decode(model, tokens)
    assert adapters.decode(model, tokens) == first
    assert adapters.decode(build_model(order, deepcopy(tables)), tokens) == first
    assert (tables, tokens) == before


@pytest.mark.parametrize('order', [1, 2])
def test_oov_with_and_without_unk(order: int) -> None:
    """Chặn tra cứu OOV sai: thiếu UNK báo lỗi, có UNK phải dùng emission UNK."""
    tables = toy_tables()
    model = build_model(order, tables)
    with pytest.raises(ValueError):
        adapters.decode(model, ['unseen'])
    tables['tokens'] = ('x', 'y', 'z', '<UNK>')
    tables['e'] = {tag: {'x': .4, 'y': .2, 'z': .3, '<UNK>': .1} for tag in LABELS}
    model = build_model(order, tables)
    tags, score = adapters.decode(model, ['unseen'])
    assert len(tags) == 1 and isfinite(score)
    assert (tags, score) == adapters.decode(model, ['<UNK>'])
    assert score == pytest.approx(direct_score(order, tuple(tags), ['<UNK>'], tables))


def test_two_start_states() -> None:
    """Chặn dùng một start hoặc PI thay q2 bằng bảng start ưu tiên C trái với PI."""
    tables = toy_tables()
    tables['q2'][START, START] = {'A': .05, 'B': .05, 'C': .8, END: .1}
    tags, score = adapters.decode(build_model(2, tables), ['x'])
    # C: .8*.2*.2=.032; A/B: .05*.5*.3=.0075 -> C thắng duy nhất.
    assert tags == ['C']
    assert score == pytest.approx(log(.8 * .2 * .2))


@pytest.mark.parametrize('order', [1, 2])
def test_end_state_changes_optimal_path(order: int) -> None:
    """Chặn bỏ EOS: A/B thắng trước EOS nhưng C thắng khi nhân xác suất kết thúc."""
    tables = toy_tables()
    for tag in LABELS:
        row = {'A': .33, 'B': .33, 'C': .33, END: .01} if tag != 'C' else {'A': .05, 'B': .05, 'C': .1, END: .8}
        tables['q1'][tag] = dict(row)
        tables['q2'][START, tag] = dict(row)
    tags, score = adapters.decode(build_model(order, tables), ['x'])
    # Bigram C=.2*.2*.8=.032; Trigram C=.1*.2*.8=.016; A/B=.4*.5*.01=.002.
    assert tags == ['C']
    assert score == pytest.approx(log((.2 if order == 1 else .1) * .2 * .8))


@pytest.mark.parametrize('order', [1, 2])
def test_empty_sequence_contract(order: int) -> None:
    """Chặn truy cập nhãn cuối của câu rỗng; giữ quy ước empty của toy adapter."""
    tables = toy_tables()
    model = build_model(order, tables)
    if order == 1:
        with pytest.raises(ValueError):
            adapters.decode(model, [])
    else:
        tags, score = adapters.decode(model, [])
        assert tags == []
        assert score == pytest.approx(log(tables['q2'][START, START][END]))


def training_hashes(model: object) -> tuple[str, str]:
    """Hash counts thật và vocab thật từ adapter, không tái dựng từ train."""
    state = adapters.get_training_snapshot(model)
    assert state['counts'], 'Snapshot counts rỗng không chứng minh chống leakage.'
    counts = json.dumps(state['counts'], sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    vocabs = json.dumps({k: sorted(state[k]) for k in ('train_vocab', 'emission_vocab')}, ensure_ascii=False, sort_keys=True)
    return sha256(counts.encode()).hexdigest(), sha256(vocabs.encode()).hexdigest()


@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('cutoff', [0, 1, 2])
def test_changed_dev_test_do_not_change_training_counts(order: int, cutoff: int) -> None:
    """Chặn leakage từ dev/test hoặc predict làm thay đổi counts/vocab đã fit train."""
    train = [(list(tokens), list(tags)) for tokens, tags in TRAIN]
    train_before = deepcopy(train)
    dev = [(['a', 'dev_only'], ['NOUN', 'VERB'])]
    test = [(['b', 'test_only'], ['VERB', 'NOUN'])]
    try:
        model = adapters.fit_train_model(order, train, alpha=.1, cutoff=cutoff)
        before = training_hashes(model)
    except NotImplementedError as exc:
        pytest.xfail(f'Chờ API fit/snapshot counts TV2: {exc}')
    for corpus in (dev, test):
        for tokens, _ in corpus:
            adapters.decode(model, tokens)
    assert training_hashes(model) == before
    dev[:] = [(['dev_only'] * 20, ['ADJ'] * 20)]
    test[:] = [(['test_only'] * 30, ['ADJ'] * 30)]
    for corpus in (dev, test):
        for tokens, _ in corpus:
            adapters.decode(model, tokens)
    assert training_hashes(model) == before
    # Fit lại chỉ train, tuyệt đối không truyền dev/test hoặc gold của chúng vào model.
    refit = adapters.fit_train_model(order, train, alpha=.1, cutoff=cutoff)
    assert training_hashes(refit) == before
    state = adapters.get_training_snapshot(refit)
    assert set(state['train_vocab']) == {'a', 'b', 'c', 'd'}
    assert 'dev_only' not in state['emission_vocab'] and 'test_only' not in state['emission_vocab']
    assert train == train_before
