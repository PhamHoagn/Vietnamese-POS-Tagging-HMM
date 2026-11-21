"""Check the integration boundary, without replacing TV1's oracle expectations."""

from copy import deepcopy
import json
import math

import pytest

from src.hmm_bigram import BigramHMMTagger
from src.hmm_common import END, START
from src.hmm_trigram import TrigramHMMTagger
from tests import adapters
from tests.test_normalization import TRAIN, toy_tables


@pytest.mark.parametrize("order", [1, 2])
def test_table_loading_copies_inputs_and_rows_are_independent(order):
    tables = toy_tables()
    before = deepcopy(tables)
    model = adapters.build_toy_model(order, tables)
    expected = adapters.decode(model, ["x", "y"])
    assert tables == before
    history = ("A",) if order == 1 else (START, START)
    row = adapters.get_transition_row(model, history)
    row["A"] = 999
    emission = adapters.get_emission_row(model, "A")
    emission["x"] = 999
    tables["e"]["A"]["x"] = 999
    assert adapters.decode(model, ["x", "y"]) == expected
    with pytest.raises(ValueError):
        adapters.get_transition_row(model, ())
    with pytest.raises(KeyError):
        adapters.get_transition_row(model, ("unknown",) * order)
    with pytest.raises(KeyError):
        adapters.get_emission_row(model, END)


@pytest.mark.parametrize("order", [1, 2])
def test_probability_zero_is_not_smoothed_and_impossible_path_errors(order):
    tables = toy_tables()
    tables["pi"] = {"A": 1., "B": 0., "C": 0.}
    tables["q1"]["A"] = {"A": 0., "B": 1., "C": 0., END: 0.}
    tables["q1"]["B"] = {"A": 0., "B": 0., "C": 0., END: 1.}
    tables["q2"][START, START] = {"A": 1., "B": 0., "C": 0., END: 0.}
    tables["q2"][START, "A"] = {"A": 0., "B": 1., "C": 0., END: 0.}
    tables["q2"]["A", "B"] = {"A": 0., "B": 0., "C": 0., END: 1.}
    model = adapters.build_toy_model(order, tables)
    tags, score = adapters.decode(model, ["x", "y"])
    assert tags == ["A", "B"]
    assert score == pytest.approx(math.log(.5 * .3))
    assert model.log_joint(["x", "y"], ["B", "A"]) == -math.inf
    with pytest.raises(ValueError, match="positive-probability"):
        adapters.decode(model, ["x"])
    with pytest.raises(ValueError, match="OOV"):
        model.map_token("unseen")


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("problem", ["bad_emission", "bad_transition", "missing_row", "negative"])
def test_bad_probability_tables_rejected(order, problem):
    tables = toy_tables()
    if problem == "bad_emission":
        tables["e"]["A"]["x"] = .9
    elif problem == "bad_transition":
        tables["q1"]["A"][END] = .9
        tables["q2"][START, START][END] = .9
    elif problem == "missing_row":
        del tables["q1"]["A"]
        del tables["q2"][START, START]
    else:
        tables["e"]["A"]["x"] = -.5
    with pytest.raises(ValueError):
        adapters.build_toy_model(order, tables)


@pytest.mark.parametrize("order", [1, 2])
def test_train_snapshot_copies_real_counts_and_preserves_symbol_identity(order):
    model = adapters.fit_train_model(order, [(["<UNK>"], ["NOUN"])], alpha=.1, cutoff=0)
    first = adapters.get_training_snapshot(model)
    assert first["counts"]["word"] == {"<UNK>": 1}
    assert {tuple(json.loads(word)) for word in first["emission_vocab"]} == {("FORM", "<UNK>"), ("UNK",)}
    first["counts"]["word"]["<UNK>"] = 999
    assert adapters.get_training_snapshot(model)["counts"]["word"] == {"<UNK>": 1}
    with pytest.raises(ValueError, match="literal"):
        adapters.get_emission_row(model, "NOUN")


@pytest.mark.parametrize("model_class", [BigramHMMTagger, TrigramHMMTagger])
def test_table_models_are_not_train_checkpoints_and_can_be_refit(tmp_path, model_class):
    order = 1 if model_class is BigramHMMTagger else 2
    model = adapters.build_toy_model(order, toy_tables())
    with pytest.raises(ValueError, match="checkpoint"):
        model.save_config(tmp_path / "model.json")
    with pytest.raises(ValueError, match="counts"):
        adapters.get_training_snapshot(model)
    model.fit([(list(words), list(tags)) for words, tags in TRAIN])
    assert model.tags == ("ADJ", "NOUN", "VERB")
    assert len(model.predict(["a"])) == 1
    model.save_config(tmp_path / "model.json")
