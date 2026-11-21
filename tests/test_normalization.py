"""Kỳ vọng Add-alpha từ đếm tay, độc lập với decoder và bộ ước lượng TV2."""

from collections import Counter
from copy import deepcopy
from itertools import product
from math import isfinite, log

import pytest

import adapters
from oracle_bruteforce import E, END, LABELS, PI, Q1, Q2, START, TOKENS

# Giữ hai helper đang được test_edge_cases/test_bruteforce sử dụng.
def toy_tables() -> adapters.ToyTables:
    """Sao chép bảng xác suất oracle, chỉ dành cho test decoder cũ."""
    return deepcopy(dict(labels=LABELS, tokens=TOKENS, pi=PI, q1=Q1, q2=Q2, e=E))


pending_api = pytest.mark.xfail(raises=NotImplementedError, strict=True,
                               reason="Chờ adapter TV2; chưa xác nhận decoder")

S = ('ADJ', 'NOUN', 'VERB')
DESTINATIONS = (*S, END)
ALPHAS = (0.01, 0.1, 1.0)
TRAIN = (
    (('a', 'b'), ('NOUN', 'VERB')),
    (('a', 'c'), ('NOUN', 'NOUN')),
    (('b', 'a'), ('VERB', 'NOUN')),
    (('d',), ('ADJ',)),
    (('a', 'b'), ('NOUN', 'VERB')),
    (('c',), ('NOUN',)),
)
# Đếm tay 6 câu, 10 token: a=4, b=3, c=2, d=1; NOUN=6, VERB=3, ADJ=1.
# Mỗi câu thêm đúng 1 EOS -> có 16 transition, không đếm start là token.
FREQUENCIES = {'a': 4, 'b': 3, 'c': 2, 'd': 1}
TAG_COUNTS = {'ADJ': 1, 'NOUN': 6, 'VERB': 3}
BIGRAM_COUNTS = {
    START: {'NOUN': 4, 'VERB': 1, 'ADJ': 1},
    'NOUN': {'VERB': 2, 'NOUN': 1, END: 3},
    'VERB': {'NOUN': 1, END: 2},
    'ADJ': {END: 1},
}
# Hai câu NV: (s,s)->N, (s,N)->V, (N,V)->EOS, mỗi loại xuất hiện 2 lần.
# Câu NN: (s,s)->N, (s,N)->N, (N,N)->EOS.
# Câu VN: (s,s)->V, (s,V)->N, (V,N)->EOS.
# Câu ADJ: (s,s)->ADJ, (s,ADJ)->EOS. Câu N: (s,s)->N, (s,N)->EOS.
TRIGRAM_COUNTS = {
    (START, START): {'NOUN': 4, 'VERB': 1, 'ADJ': 1},
    (START, 'NOUN'): {'VERB': 2, 'NOUN': 1, END: 1},
    (START, 'VERB'): {'NOUN': 1},
    (START, 'ADJ'): {END: 1},
    ('NOUN', 'VERB'): {END: 2},
    ('NOUN', 'NOUN'): {END: 1},
    ('VERB', 'NOUN'): {END: 1},
}
# Vk giữ f>k. k=0: a,b,c,d + UNK; k=1: a,b,c + UNK; k=2: a,b + UNK.
VOCABULARIES = {0: ('a', 'b', 'c', 'd'), 1: ('a', 'b', 'c'), 2: ('a', 'b')}
EMISSION_COUNTS = {
    0: {'ADJ': {'d': 1}, 'NOUN': {'a': 4, 'c': 2}, 'VERB': {'b': 3}},
    1: {'ADJ': {'<UNK>': 1}, 'NOUN': {'a': 4, 'c': 2}, 'VERB': {'b': 3}},
    2: {'ADJ': {'<UNK>': 1}, 'NOUN': {'a': 4, '<UNK>': 2}, 'VERB': {'b': 3}},
}


def fit_model(order: int, alpha: float, cutoff: int) -> object:
    """Chỉ đánh dấu chờ ở bước fit; lỗi truy xuất/kiểm tra không bị che."""
    train = [(list(tokens), list(tags)) for tokens, tags in TRAIN]
    try:
        return adapters.fit_train_model(order, train, alpha=alpha, cutoff=cutoff)
    except NotImplementedError as exc:
        pytest.xfail(f'Chưa có fit TV2/adapter: {exc}')


def check_row(actual: dict[str, float], expected: dict[str, float]) -> None:
    """Kiểm tra cả giá trị, miền xác suất và log hữu hạn, không chỉ tổng hàng."""
    assert set(actual) == set(expected)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert sum(actual.values()) == pytest.approx(1.0, rel=0, abs=1e-12)
    for p in actual.values():
        assert isfinite(p) and 0 < p <= 1
        assert isfinite(log(p))  # Chỉ log sau khi xác nhận p>0, bắt được log(0).


def test_hand_counts_match_six_sentences() -> None:
    """Đối chiếu bảng đếm tay với corpus gốc, không đọc count từ model."""
    assert len(TRAIN) == 6
    assert Counter(x for tokens, _ in TRAIN for x in tokens) == FREQUENCIES
    assert Counter(y for _, tags in TRAIN for y in tags) == TAG_COUNTS
    q1, q2 = {}, {}
    for _, tags in TRAIN:
        previous, history = START, (START, START)
        for target in (*tags, END):
            q1.setdefault(previous, Counter())[target] += 1
            q2.setdefault(history, Counter())[target] += 1
            previous, history = target, (history[1], target)
    assert q1 == BIGRAM_COUNTS
    assert q2 == TRIGRAM_COUNTS


@pytest.mark.parametrize('alpha', ALPHAS)
@pytest.mark.parametrize('history', list(product((*S, START), repeat=2)))
def test_trigram_add_alpha(alpha: float, history: tuple[str, str]) -> None:
    """q2=(C(u,v,w)+alpha)/(C(u,v)+4alpha), gồm mọi cặp và EOS."""
    counts = TRIGRAM_COUNTS.get(history, {})
    total = sum(counts.values())
    expected = {w: (counts.get(w, 0) + alpha) / (total + alpha * 4) for w in DESTINATIONS}
    if not counts:
        assert expected == pytest.approx(dict.fromkeys(DESTINATIONS, 1 / 4))
    # (<s>,<s>): C=6, q(N)=(4+alpha)/(6+4alpha), q(EOS)=alpha/(6+4alpha).
    model = fit_model(2, alpha, 1)
    check_row(adapters.get_transition_row(model, history), expected)


@pytest.mark.parametrize('alpha', ALPHAS)
@pytest.mark.parametrize('previous', (*S, START))
def test_bigram_add_alpha(alpha: float, previous: str) -> None:
    """q1 dùng cùng 4 đích; hàng start bao gồm xác suất EOS sau smoothing."""
    counts = BIGRAM_COUNTS[previous]
    expected = {w: (counts.get(w, 0) + alpha) / (sum(counts.values()) + alpha * 4)
                for w in DESTINATIONS}
    check_row(adapters.get_transition_row(fit_model(1, alpha, 1), (previous,)), expected)


@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('alpha', ALPHAS)
@pytest.mark.parametrize('cutoff', [0, 1, 2])
@pytest.mark.parametrize('tag', S)
def test_emission_add_alpha(order: int, alpha: float, cutoff: int, tag: str) -> None:
    """e=(Ck(y,x)+alpha)/(C(y)+alpha*(|Vk|+1)); ADJ chỉ có 1 token."""
    vocabulary = (*VOCABULARIES[cutoff], '<UNK>')
    denominator = TAG_COUNTS[tag] + alpha * len(vocabulary)
    counts = EMISSION_COUNTS[cutoff][tag]
    expected = {x: (counts.get(x, 0) + alpha) / denominator for x in vocabulary}
    # k=2: NOUN a=4/UNK=2, ADJ UNK=1; mẫu số lần lượt 6+3alpha, 1+3alpha.
    check_row(adapters.get_emission_row(fit_model(order, alpha, cutoff), tag), expected)


@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('alpha', ALPHAS)
@pytest.mark.parametrize('tag', S)
def test_cutoff_zero_unk_exact(order: int, alpha: float, tag: str) -> None:
    """k=0 có 4 từ giữ lại và UNK chưa thấy; e(UNK|y)=alpha/(C(y)+5alpha)."""
    row = adapters.get_emission_row(fit_model(order, alpha, 0), tag)
    expected = alpha / (TAG_COUNTS[tag] + alpha * 5)
    assert row['<UNK>'] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert isfinite(row['<UNK>']) and row['<UNK>'] > 0
    assert isfinite(log(row['<UNK>']))


@pytest.mark.parametrize('alpha', ALPHAS)
@pytest.mark.parametrize('cutoff', [0, 1, 2])
def test_hand_emission_counts_and_formula(alpha: float, cutoff: int) -> None:
    """Kiểm tra fixture tính tay: ánh xạ rare không mất token, mẫu số đúng tập đích."""
    for tag in S:
        counted = Counter()
        for tokens, tags in TRAIN:
            for token, label in zip(tokens, tags):
                if label == tag:
                    counted[token if FREQUENCIES[token] > cutoff else '<UNK>'] += 1
        assert counted == EMISSION_COUNTS[cutoff][tag]
        assert sum(counted.values()) == TAG_COUNTS[tag]
        vocabulary = (*VOCABULARIES[cutoff], '<UNK>')
        denominator = TAG_COUNTS[tag] + alpha * len(vocabulary)
        assert sum((counted[x] + alpha) / denominator for x in vocabulary) == pytest.approx(1)
