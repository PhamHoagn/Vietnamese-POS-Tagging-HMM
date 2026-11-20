"""TV2 developer checks; TV1 must supply a separate independent oracle later."""

from copy import deepcopy
from itertools import product
import json
import math

import numpy as np
import pytest

from src.data_loader import load_conllu
from src.hmm_bigram import BigramHMM, BigramHMMTagger
from src.hmm_common import CODE_VERSION, END, START, UNK, HMMTrainingCache
from tests.toy_corpus import (
    EXPECTED_EMISSION_COUNTS, EXPECTED_EMISSIONS,
    EXPECTED_TRANSITION_COUNTS, EXPECTED_TRANSITIONS, HAND_CASES, TOY_CORPUS,
)


def test_hand_calculated_counts_and_probabilities():
    model = BigramHMMTagger(alpha=1, cutoff=1).fit(TOY_CORPUS)
    assert model.tags == ("NOUN", "VERB")
    assert dict(model.word_counts_) == {"a": 2, "b": 3, "c": 1}
    np.testing.assert_array_equal(model.tag_counts_, [3, 3])
    np.testing.assert_array_equal(model.transition_counts_, EXPECTED_TRANSITION_COUNTS)
    np.testing.assert_array_equal(model.emission_counts_, EXPECTED_EMISSION_COUNTS)
    np.testing.assert_allclose(np.exp(model.transition_log_probs_),
                               np.array(EXPECTED_TRANSITIONS, dtype=float), rtol=1e-14)
    np.testing.assert_allclose(np.exp(model.emission_log_probs_),
                               np.array(EXPECTED_EMISSIONS, dtype=float), rtol=1e-14)


@pytest.mark.parametrize("words,tags,probability", HAND_CASES)
def test_hand_calculated_joint_and_predictions(words, tags, probability):
    model = BigramHMM().fit(TOY_CORPUS, 1.0, 1)
    assert model.predict(words) == tags
    assert model.log_joint(words, tags) == pytest.approx(math.log(probability), abs=1e-13)


@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.1, 1.0, 3.0])
@pytest.mark.parametrize("cutoff", [0, 1, 2, 3])
def test_transition_emission_normalization_for_all_contexts(alpha, cutoff):
    model = BigramHMMTagger(alpha, cutoff).fit(TOY_CORPUS)
    np.testing.assert_allclose(np.exp(model.transition_log_probs_).sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(np.exp(model.emission_log_probs_).sum(axis=1), 1, atol=1e-14)
    for previous in (START, *model.tags, "ADJ"):
        probabilities = [model.transition_probability(previous, tag) for tag in (*model.tags, END)]
        assert sum(probabilities) == pytest.approx(1)
    for tag in model.tags:
        assert sum(model.emission_probability(word, tag) for word in model.vocab_) == pytest.approx(1)
    assert model.transition_probability("ADJ", END) == pytest.approx(1 / 3)
    assert np.isfinite(model.transition_log_probs_).all()
    assert np.isfinite(model.emission_log_probs_).all()


@pytest.mark.parametrize("alpha", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("cutoff", [0, 1, 2])
@pytest.mark.parametrize("words", [["a"], ["a", "b"], ["a", "b", "unseen"]])
def test_viterbi_matches_short_exhaustive_log_joint_self_check(alpha, cutoff, words):
    # This is TV2's quick check using log_joint, not TV1's independent oracle.
    model = BigramHMMTagger(alpha, cutoff).fit(TOY_CORPUS)
    all_scores = [(model.log_joint(words, list(tags)), tags)
                  for tags in product(model.tags, repeat=len(words))]
    best_score = max(score for score, _ in all_scores)
    predicted = model.predict(words)
    assert model.log_joint(words, predicted) == pytest.approx(best_score, abs=1e-12)
    assert len(predicted) == len(words)
    assert set(predicted) <= set(model.tags)


def test_end_transition_changes_single_token_decision():
    train = [(["x"] * 10, ["NOUN"] * 10)] * 3 + [(["x"], ["VERB"])] * 2
    model = BigramHMMTagger(1, 0).fit(train)
    score_without_end = model.transition_log_probs_[-1, :-1] + model.emission_log_probs_[:, 0]
    assert model.tags[int(np.argmax(score_without_end))] == "NOUN"
    assert model.predict(["x"]) == ["VERB"]
    assert model.log_joint(["x"], ["VERB"]) > model.log_joint(["x"], ["NOUN"])


def test_global_lexical_tie_break_not_just_smallest_final_tag():
    train = [(["a", "a"], ["NOUN", "VERB"]), (["a", "a"], ["VERB", "NOUN"])]
    model = BigramHMMTagger(1, 0).fit(train)
    assert model.log_joint(["a", "a"], ["NOUN", "VERB"]) == pytest.approx(
        model.log_joint(["a", "a"], ["VERB", "NOUN"]))
    # Smallest final tag alone would choose VERB,NOUN, the wrong lexical tie-break.
    for _ in range(2):
        assert model.predict(["a", "a"]) == ["NOUN", "VERB"]
        assert model.predict(["a"] * 3) == ["NOUN", "VERB", "NOUN"]


@pytest.mark.parametrize("cutoff,retained", [(0, {"a", "b", "c"}), (1, {"a", "b"}),
                                           (2, {"b"}), (3, set())])
def test_cutoff_inclusive_and_raw_vocab_distinct_from_emission_vocab(cutoff, retained):
    model = BigramHMMTagger(0.1, cutoff).fit(TOY_CORPUS)
    assert model.train_vocab_ == {"a", "b", "c"}
    assert model.retained_vocab_ == retained
    assert model.vocab_ == retained | {UNK}
    for word in model.train_vocab_:
        assert model.unk_map_[word] == (word if word in retained else UNK)
    assert model.map_token("unseen") is UNK
    assert "unseen" not in model.train_vocab_
    if cutoff == 0:
        assert model.emission_counts_[:, model.unk_index_].sum() == 0


def test_literal_unk_form_does_not_collide_with_internal_sentinel():
    model = BigramHMMTagger(1, 0).fit([(["<UNK>"], ["NOUN"])])
    assert model.map_token("<UNK>") == "<UNK>"
    assert model.map_token("unknown") is UNK
    assert model.word_to_index_["<UNK>"] != model.word_to_index_[UNK]
    assert model.emission_probability("<UNK>", "NOUN") == pytest.approx(2 / 3)
    assert model.emission_probability(UNK, "NOUN") == pytest.approx(1 / 3)


def test_cached_and_uncached_models_agree_and_share_only_matching_work():
    cache = HMMTrainingCache()
    first = BigramHMMTagger(0.1, 1, cache=cache).fit(TOY_CORPUS)
    same = BigramHMMTagger(0.1, 1, cache=cache).fit(deepcopy(TOY_CORPUS))
    other_alpha = BigramHMMTagger(1, 1, cache=cache).fit(TOY_CORPUS)
    assert first.transition_counts_ is same.transition_counts_ is other_alpha.transition_counts_
    assert first.emission_counts_ is same.emission_counts_ is other_alpha.emission_counts_
    assert first.transition_log_probs_ is same.transition_log_probs_
    assert first.emission_log_probs_ is same.emission_log_probs_
    assert first.transition_log_probs_ is not other_alpha.transition_log_probs_
    uncached = BigramHMMTagger(0.1, 1, cache=cache, use_cache=False).fit(TOY_CORPUS)
    assert uncached.transition_log_probs_ is not first.transition_log_probs_
    np.testing.assert_array_equal(first.transition_log_probs_, uncached.transition_log_probs_)
    np.testing.assert_array_equal(first.emission_log_probs_, uncached.emission_log_probs_)
    words = ["c", "a", "unseen", "b"]
    assert first.predict(words) == same.predict(words) == uncached.predict(words)
    assert first.log_joint(words, first.predict(words)) == uncached.log_joint(words, uncached.predict(words))


def test_cache_isolation_refit_and_caller_mutations():
    train = deepcopy(TOY_CORPUS)
    original = deepcopy(train)
    cache = HMMTrainingCache()
    first = BigramHMMTagger(cache=cache).fit(train)
    first.predict(["dev_only"])
    assert train == original
    assert "dev_only" not in first.train_vocab_
    train[0][0][0] = "changed"
    assert "changed" not in first.train_vocab_
    second = BigramHMMTagger(cache=cache).fit(train)
    assert first.training_fingerprint_ != second.training_fingerprint_
    assert first.transition_log_probs_ is not second.transition_log_probs_
    first.fit([(["new"], ["ADJ"])])
    assert first.tags == ("ADJ",)
    assert first.train_vocab_ == {"new"}
    assert first.predict(["a"]) == ["ADJ"]


def test_empty_sentences_do_not_add_boundary_counts():
    plain = BigramHMMTagger().fit(TOY_CORPUS)
    padded = BigramHMMTagger().fit([([], []), *TOY_CORPUS, ([], [])])
    np.testing.assert_array_equal(plain.transition_counts_, padded.transition_counts_)
    assert plain.training_fingerprint_ == padded.training_fingerprint_


def test_empty_predictions_and_unfitted_calls_are_explicit():
    for model in (BigramHMMTagger(), BigramHMM()):
        for action in (lambda: model.predict([]), lambda: model.predict_corpus([]),
                       lambda: model.log_joint(["a"], ["NOUN"])):
            with pytest.raises(RuntimeError, match="fit"):
                action()
    model = BigramHMMTagger().fit(TOY_CORPUS)
    assert model.predict([]) == []
    assert model.predict_corpus([[], ["a"]]) == [[], model.predict(["a"])]
    assert model.predict_corpus([]) == []
    with pytest.raises(ValueError, match="at least one"):
        BigramHMM().fit(TOY_CORPUS).predict([])
    with pytest.raises(ValueError, match="non-empty"):
        model.log_joint([], [])


@pytest.mark.parametrize("train", [[], [([], [])], [(["a"], [])],
                                      [(["a"], ["BAD"])], [([""], ["NOUN"])]])
def test_invalid_refit_preserves_previous_working_model(train):
    model = BigramHMMTagger().fit(TOY_CORPUS)
    before = model.predict(["a", "b"])
    with pytest.raises(ValueError):
        model.fit(train)
    assert model.predict(["a", "b"]) == before


def test_invalid_prediction_and_scoring_inputs():
    model = BigramHMMTagger().fit(TOY_CORPUS)
    with pytest.raises(TypeError, match="pre-tokenized"):
        model.predict("a b")
    for tokens in ([""], [None]):
        with pytest.raises(ValueError):
            model.predict(tokens)
    for words, tags in [(["a"], []), (["a"], [START]), (["a"], [END]), (["a"], ["ADJ"])]:
        with pytest.raises(ValueError):
            model.log_joint(words, tags)
    with pytest.raises(ValueError):
        model.emission_probability("a", END)
    with pytest.raises(ValueError):
        model.transition_probability(END, "NOUN")
    model.alpha = 3.0
    with pytest.raises(RuntimeError, match="Configuration changed"):
        model.predict(["a"])
    model.fit(TOY_CORPUS)
    assert len(model.predict(["a"])) == 1


def test_log_joint_works_without_decoder_or_cached_log_tables(monkeypatch):
    model = BigramHMMTagger(1, 1).fit(TOY_CORPUS)
    def fail(*args):
        raise AssertionError("Scoring must not call Viterbi.")
    monkeypatch.setattr(model, "_viterbi", fail)
    model.transition_log_probs_ = None
    model.emission_log_probs_ = None
    assert model.log_joint(["a", "b"], ["NOUN", "VERB"]) == pytest.approx(math.log(4 / 63))


@pytest.mark.parametrize("alpha", [0.001, 3.0, 1e308, 5e-324])
def test_long_sentence_and_extreme_finite_alpha_use_finite_logs(alpha):
    model = BigramHMMTagger(alpha, 0).fit(TOY_CORPUS)
    words = ["unseen", "a", "b"] * 334
    predictions = model.predict(words)
    assert len(predictions) == len(words)
    assert np.isfinite(model.transition_log_probs_).all()
    assert np.isfinite(model.emission_log_probs_).all()
    assert math.isfinite(model.log_joint(words, predictions))


@pytest.mark.parametrize("model_class", [BigramHMMTagger, BigramHMM])
def test_config_roundtrip_refits_and_checks_corpus(tmp_path, model_class):
    model = model_class(1, 1).fit(TOY_CORPUS)
    path = tmp_path / "config.json"
    model.save_config(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["alpha"] == 1
    assert config["cutoff"] == 1
    assert config["tags"] == ["NOUN", "VERB"]
    assert config["code_version"] == CODE_VERSION
    restored = model_class.load_config(path)
    with pytest.raises(RuntimeError):
        restored.predict(["a"])
    restored.fit(TOY_CORPUS)
    words = [["a"], ["a", "b"], ["unseen", "c"]]
    assert restored.predict_corpus(words) == model.predict_corpus(words)
    with pytest.raises(ValueError, match="does not match"):
        model_class.load_config(path).fit([(["other"], ["NOUN"])])


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("code_version", "different"), ("tags", ["VERB", "NOUN"]),
    ("tags", ["BAD"]), ("training_sha256", "invalid"), ("alpha", 0), ("cutoff", -1),
])
def test_reject_invalid_saved_configuration(tmp_path, field, value):
    path = tmp_path / "config.json"
    BigramHMMTagger().fit(TOY_CORPUS).save_config(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    config[field] = value
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        BigramHMMTagger.load_config(path)


def test_parse_fixture_train_only_and_predict_without_gold():
    from pathlib import Path
    corpus = load_conllu(Path(__file__).parent / "fixtures" / "tiny.conllu")
    model = BigramHMMTagger().fit(corpus[:2])
    dev_words = [words for words, _ in corpus[2:]]
    predictions = model.predict_corpus(dev_words)
    assert len(predictions) == len(dev_words)
    assert all(len(words) == len(tags) for words, tags in zip(dev_words, predictions))
    assert "học" not in model.train_vocab_


def test_cached_arrays_and_maps_reject_accidental_edits():
    model = BigramHMMTagger().fit(TOY_CORPUS)
    with pytest.raises(ValueError):
        model.transition_counts_[0, 0] = 99
    with pytest.raises(ValueError):
        model.emission_log_probs_[0, 0] = 0
    with pytest.raises(TypeError):
        model.unk_map_["a"] = UNK
