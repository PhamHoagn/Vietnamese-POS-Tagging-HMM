import inspect

import pytest

from scripts.smoke_test import main
from src.baseline import MostFrequentTagger
from src.contracts import POSTagger, UPOS_TAGS
from src.data_loader import load_conllu
from src.hmm_bigram import BigramHMMTagger
from src.hmm_trigram import TrigramHMMTagger
from src.model_selection import grid_search_pos


@pytest.mark.parametrize("tagger_class", [MostFrequentTagger, BigramHMMTagger, TrigramHMMTagger])
def test_shared_model_signatures(tagger_class):
    model = tagger_class()
    assert isinstance(model, POSTagger)
    assert list(inspect.signature(model.fit).parameters) == ["train_data"]
    assert list(inspect.signature(model.predict).parameters) == ["tokens"]


@pytest.mark.parametrize("tagger_class", [BigramHMMTagger, TrigramHMMTagger])
def test_hmm_training_now_produces_aligned_predictions(tagger_class):
    model = tagger_class(alpha=0.1, cutoff=1)
    assert model.alpha == 0.1
    assert model.cutoff == 1
    assert model.fit([(["a"], ["NOUN"])]) is model
    assert model.predict([]) == []
    assert model.predict(["a"]) == ["NOUN"]


@pytest.mark.parametrize("tagger_class", [BigramHMMTagger, TrigramHMMTagger])
@pytest.mark.parametrize("kwargs", [
    {"alpha": 0}, {"alpha": -1}, {"alpha": float("nan")}, {"alpha": float("inf")},
    {"alpha": True}, {"cutoff": -1}, {"cutoff": 1.5}, {"cutoff": True},
])
def test_hmm_config_rejects_invalid_values(tagger_class, kwargs):
    with pytest.raises(ValueError):
        tagger_class(**kwargs)


def test_grid_search_is_explicit_placeholder_with_no_test_input():
    assert "test_data" not in inspect.signature(grid_search_pos).parameters
    with pytest.raises(NotImplementedError, match="grid search"):
        grid_search_pos(BigramHMMTagger, [], [], {"alpha": [0.1], "cutoff": [1]})


def test_smoke_checks_integration(capsys):
    main()
    assert "PASS: synthetic fixture" in capsys.readouterr().out


def test_fixture_baseline_has_aligned_valid_predictions():
    from pathlib import Path

    corpus = load_conllu(Path(__file__).parent / "fixtures" / "tiny.conllu")
    model = MostFrequentTagger().fit(corpus[:2])
    for tokens, _ in corpus[2:]:
        predictions = model.predict(tokens)
        assert len(predictions) == len(tokens)
        assert set(predictions) <= set(UPOS_TAGS)
