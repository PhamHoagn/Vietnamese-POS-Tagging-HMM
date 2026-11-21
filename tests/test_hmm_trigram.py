"""TV2 self-checks for second-order counts, NumPy Viterbi and caching."""

from copy import deepcopy
from itertools import product
import json
import math

import numpy as np
import pytest

from src.hmm_bigram import BigramHMMTagger
from src.hmm_common import END, START, UNK, HMMTrainingCache
from src.hmm_trigram import TrigramHMM, TrigramHMMTagger
from tests.toy_corpus import TOY_CORPUS, TRIGRAM_HAND_CASES, TRIGRAM_NONZERO_COUNTS


def test_hand_trigram_counts_include_two_starts_and_one_end():
    model = TrigramHMMTagger(1, 1).fit(TOY_CORPUS)
    expected = np.zeros((3, 3, 3), dtype=np.int64)
    for indices, count in TRIGRAM_NONZERO_COUNTS.items():
        expected[indices] = count
    np.testing.assert_array_equal(model.transition_counts_, expected)
    assert expected.sum() == 10  # Six tokens and four END transitions.
    assert expected[:, :, -1].sum() == 4
    assert model.transition_probability(START, START, "NOUN") == pytest.approx(4 / 7)
    assert model.transition_probability(START, "VERB", END) == pytest.approx(1 / 2)
    assert model.transition_probability("NOUN", "VERB", END) == pytest.approx(3 / 5)


@pytest.mark.parametrize("words,tags,probability", TRIGRAM_HAND_CASES)
def test_joint_and_decode_match_hand_fractions(words, tags, probability):
    model = TrigramHMM().fit(TOY_CORPUS, 1, 1)
    assert model.predict(words) == tags
    assert model.log_joint(words, tags) == pytest.approx(math.log(probability), abs=1e-13)


@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.1, 1.0, 3.0])
@pytest.mark.parametrize("cutoff", [0, 1, 2, 3])
def test_every_history_and_emission_normalizes(alpha, cutoff):
    model = TrigramHMMTagger(alpha, cutoff).fit(TOY_CORPUS)
    np.testing.assert_allclose(np.exp(model.transition_log_probs_).sum(axis=-1), 1, atol=1e-14)
    np.testing.assert_allclose(np.exp(model.emission_log_probs_).sum(axis=-1), 1, atol=1e-14)
    for first, second in product((*model.tags, START, "ADJ"), repeat=2):
        values = [model.transition_probability(first, second, tag) for tag in (*model.tags, END)]
        assert sum(values) == pytest.approx(1)
    for tag in model.tags:
        assert sum(model.emission_probability(word, tag) for word in model.vocab_) == pytest.approx(1)
    # Unseen (VERB,NOUN) must be uniform, not Bigram NOUN's non-uniform row.
    assert model.transition_probability("VERB", "NOUN", "VERB") == pytest.approx(1 / 3)
    assert model.transition_probability("VERB", "NOUN", END) == pytest.approx(1 / 3)


@pytest.mark.parametrize("alpha", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("cutoff", [0, 1, 2])
@pytest.mark.parametrize("words", [["a"], ["a", "b"], ["c", "a", "unseen"]])
def test_short_decode_maximizes_log_joint(alpha, cutoff, words):
    model = TrigramHMMTagger(alpha, cutoff).fit(TOY_CORPUS + [(["d"], ["ADJ"])])
    best = max(model.log_joint(words, list(tags)) for tags in product(model.tags, repeat=len(words)))
    predicted = model.predict(words)
    assert model.log_joint(words, predicted) == pytest.approx(best, abs=1e-12)
    assert len(predicted) == len(words)
    assert set(predicted) <= set(model.tags)


def test_random_training_corpora_against_exhaustive_self_check():
    rng = np.random.default_rng(2026)
    labels = ("ADJ", "NOUN", "VERB")
    for _ in range(60):
        train = [(["a", "b", "c"], list(labels))]
        for _ in range(4):
            length = int(rng.integers(1, 5))
            train.append((rng.choice(["a", "b", "c"], length).tolist(), rng.choice(labels, length).tolist()))
        model = TrigramHMMTagger(float(rng.choice([.01, .1, 1.])), int(rng.integers(0, 3))).fit(train)
        words = rng.choice(["a", "b", "unseen"], int(rng.integers(1, 5))).tolist()
        best = max(model.log_joint(words, list(tags)) for tags in product(model.tags, repeat=len(words)))
        assert model.log_joint(words, model.predict(words)) == pytest.approx(best, abs=1e-11)


def test_global_lexical_tie_and_long_backtracking():
    train = [(["a", "a"], ["NOUN", "VERB"]), (["a", "a"], ["VERB", "NOUN"])]
    model = TrigramHMMTagger(1, 0).fit(train)
    words = ["a"] * 4
    scored = [(model.log_joint(words, list(tags)), tags) for tags in product(model.tags, repeat=4)]
    best = max(score for score, _ in scored)
    winners = [tags for score, tags in scored if abs(score - best) < 1e-12]
    assert tuple(model.predict(words)) == min(winners)
    long_words = ["a"] * 500
    assert model.predict(long_words) == model.predict(long_words)
    assert len(model.predict(long_words)) == 500


def test_single_tag_and_single_token_have_correct_end_history():
    model = TrigramHMMTagger(1, 0).fit([(["x"], ["NOUN"])])
    assert model.predict(["unknown"]) == ["NOUN"]
    # start->N=2/3, UNK|N=1/3, (start,N)->END=2/3.
    assert model.log_joint(["unknown"], ["NOUN"]) == pytest.approx(math.log(4 / 27))
    assert model.predict(["x"] * 10) == ["NOUN"] * 10


def test_single_token_end_transition_changes_the_optimal_tag():
    train = [(["x"] * 10, ["NOUN"] * 10)] * 3 + [(["x"], ["VERB"])] * 2
    model = TrigramHMMTagger(1, 0).fit(train)
    without_end = model.transition_log_probs_[-1, -1, :-1] + model.emission_log_probs_[:, 0]
    assert model.tags[int(np.argmax(without_end))] == "NOUN"
    assert model.predict(["x"]) == ["VERB"]
    assert model.log_joint(["x"], ["VERB"]) > model.log_joint(["x"], ["NOUN"])


def test_cross_order_cache_shares_raw_counts_and_identical_emissions():
    cache = HMMTrainingCache()
    bigram = BigramHMMTagger(.1, 1, cache=cache).fit(TOY_CORPUS)
    trigram = TrigramHMMTagger(.1, 1, cache=cache).fit(TOY_CORPUS)
    repeat = TrigramHMMTagger(.1, 1, cache=cache).fit(deepcopy(TOY_CORPUS))
    reference = TrigramHMMTagger(.1, 1, use_cache=False).fit(TOY_CORPUS)
    assert bigram.word_counts_ is trigram.word_counts_
    assert bigram.emission_counts_ is trigram.emission_counts_
    assert bigram.emission_log_probs_ is trigram.emission_log_probs_
    assert trigram.transition_log_probs_ is repeat.transition_log_probs_
    np.testing.assert_array_equal(trigram.transition_log_probs_, reference.transition_log_probs_)
    np.testing.assert_array_equal(trigram.emission_log_probs_, reference.emission_log_probs_)
    words = ["a", "c", "unseen", "b"]
    assert trigram.predict(words) == repeat.predict(words) == reference.predict(words)
    assert trigram.log_joint(words, trigram.predict(words)) == reference.log_joint(words, reference.predict(words))


def test_predict_does_not_learn_and_refit_replaces_counts():
    train = deepcopy(TOY_CORPUS)
    model = TrigramHMMTagger().fit(train)
    counts = model.transition_counts_.copy()
    vocab = model.train_vocab_
    original = deepcopy(train)
    model.predict(["dev_only"] * 5)
    assert model.train_vocab_ == vocab
    np.testing.assert_array_equal(counts, model.transition_counts_)
    assert train == original
    with pytest.raises(ValueError):
        model.fit([])
    np.testing.assert_array_equal(counts, model.transition_counts_)
    model.fit([(["new"], ["ADJ"])])
    assert model.tags == ("ADJ",) and model.train_vocab_ == {"new"}
    assert model.predict(["old"]) == ["ADJ"]


def test_log_joint_is_independent_of_decoder_log_arrays(monkeypatch):
    model = TrigramHMMTagger(1, 1).fit(TOY_CORPUS)
    def fail(*args):
        raise AssertionError("Scoring must not call decoder.")
    monkeypatch.setattr(model, "_viterbi", fail)
    model.transition_log_probs_ = None
    model.emission_log_probs_ = None
    assert model.log_joint(["a", "b"], ["NOUN", "VERB"]) == pytest.approx(math.log(2 / 35))


def test_decoder_never_calls_log_after_fit(monkeypatch):
    model = TrigramHMMTagger().fit(TOY_CORPUS)
    expected = model.predict(["a", "b", "unseen"])
    def fail(*args, **kwargs):
        raise AssertionError("No np.log inside decode.")
    monkeypatch.setattr(np, "log", fail)
    assert model.predict(["a", "b", "unseen"]) == expected


def test_empty_unfitted_invalid_inputs_and_unk_collision():
    with pytest.raises(RuntimeError):
        TrigramHMMTagger().predict([])
    model = TrigramHMMTagger(1, 0).fit([(["<UNK>"], ["NOUN"])])
    assert model.predict([]) == []
    assert model.predict_corpus([[], ["unknown"]]) == [[], ["NOUN"]]
    assert model.map_token("<UNK>") == "<UNK>" and model.map_token("other") is UNK
    with pytest.raises(ValueError):
        TrigramHMM().fit(TOY_CORPUS).predict([])
    with pytest.raises(TypeError):
        model.predict("raw text")
    for words, tags in [([], []), (["x"], []), (["x"], ["VERB"]), ([""], ["NOUN"])]:
        with pytest.raises(ValueError):
            model.log_joint(words, tags)
    with pytest.raises(ValueError):
        model.transition_probability(END, START, "NOUN")
    with pytest.raises(ValueError):
        model.transition_probability(START, START, START)
    model.cutoff = 2
    with pytest.raises(RuntimeError):
        model.predict(["x"])


@pytest.mark.parametrize("alpha", [.001, 3.0, 1e308, 5e-324])
def test_extreme_positive_alpha_has_finite_log_probabilities(alpha):
    model = TrigramHMMTagger(alpha, 0).fit(TOY_CORPUS)
    assert np.isfinite(model.transition_log_probs_).all()
    assert np.isfinite(model.emission_log_probs_).all()
    words = ["unknown", "a", "b"] * 20
    assert math.isfinite(model.log_joint(words, model.predict(words)))


@pytest.mark.parametrize("model_class", [TrigramHMMTagger, TrigramHMM])
def test_trigram_config_roundtrip_and_wrong_order_rejected(tmp_path, model_class):
    model = model_class(1, 1).fit(TOY_CORPUS)
    path = tmp_path / "trigram.json"
    model.save_config(path)
    config = json.loads(path.read_text())
    assert config["model"] == "trigram"
    assert config["code_version"] == "tv2-trigram-1"
    with pytest.raises(ValueError):
        BigramHMMTagger.load_config(path)
    restored = model_class.load_config(path)
    with pytest.raises(RuntimeError):
        restored.predict(["a"])
    restored.fit(TOY_CORPUS)
    assert restored.predict(["a", "b"]) == model.predict(["a", "b"])
    with pytest.raises(ValueError):
        model_class.load_config(path).fit([(["changed"], ["NOUN"])])
