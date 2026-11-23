
import pytest

from src.model_selection import _rank_rows, grid_search_pos


class DummyTagger:
    """A tiny model used only to test the grid-search implementation."""

    def __init__(self, alpha, cutoff):
        self.alpha = alpha
        self.cutoff = cutoff

    def fit(self, train_data):
        self.train_vocab_ = frozenset(
            token
            for tokens, _ in train_data
            for token in tokens
        )
        return self

    def predict(self, tokens):
        if self.alpha == 0.2:
            return ["NOUN" if token == "known" else "VERB"
                    for token in tokens]
        return ["NOUN" for _ in tokens]


def test_grid_search_selects_best_dev_configuration():
    train = [
        (["known"], ["NOUN"]),
    ]

    dev = [
        (["known", "unknown"], ["NOUN", "VERB"]),
    ]

    results = grid_search_pos(
        DummyTagger,
        train,
        dev,
        {
            "alpha": [0.1, 0.2],
            "cutoff": [0, 1],
        },
    )

    assert len(results) == 4
    assert results[0]["param_alpha"] == 0.2
    assert results[0]["param_cutoff"] == 0
    assert results[0]["dev_token_accuracy"] == 1.0
    assert results[0]["rank"] == 1
    assert results[0]["selected"] is True

    assert sum(row["selected"] for row in results) == 1


def test_grid_search_tie_breaks_by_smaller_cutoff_then_larger_alpha():
    train = [(["known"], ["NOUN"])]
    dev = [(["known"], ["NOUN"])]

    results = grid_search_pos(
        DummyTagger,
        train,
        dev,
        {
            "alpha": [0.2, 0.1],
            "cutoff": [1, 0],
        },
    )

    assert [
        (row["param_alpha"], row["param_cutoff"])
        for row in results
    ] == [
        (0.2, 0),
        (0.1, 0),
        (0.2, 1),
        (0.1, 1),
    ]


def test_accuracy_tie_prefers_macro_f1_before_parameters():
    class TieTagger(DummyTagger):
        def predict(self, tokens):
            if self.alpha == 0.1:
                return ["NOUN", "NOUN", "VERB", "VERB"]
            return ["NOUN"] * len(tokens)

    train = [(["known"], ["NOUN"])]
    dev = [(["a", "b", "c", "d"], ["NOUN", "NOUN", "NOUN", "VERB"])]
    rows = grid_search_pos(TieTagger, train, dev, {"alpha": [0.1, 0.2], "cutoff": [0]})
    assert [row["param_alpha"] for row in rows] == [0.1, 0.2]
    assert rows[0]["dev_token_accuracy"] == rows[1]["dev_token_accuracy"] == 0.75
    assert rows[0]["dev_macro_f1"] > rows[1]["dev_macro_f1"]


def test_macro_f1_tolerance_treats_roundoff_as_a_tie():
    def row(alpha, cutoff, macro):
        return dict(param_alpha=alpha, param_cutoff=cutoff, dev_score=0.75, dev_macro_f1=macro)

    rows = [row(1.0, 1, 0.5), row(0.1, 0, 0.5 - 0.5e-12), row(1.0, 0, 0.5 - 2e-12)]
    assert _rank_rows(rows) == [rows[1], rows[0], rows[2]]
    assert _rank_rows(rows[::-1]) == _rank_rows(rows)


def test_grid_search_supports_macro_f1_scoring():
    train = [(["a"], ["NOUN"])]
    dev = [(["a"], ["NOUN"])]

    results = grid_search_pos(
        DummyTagger,
        train,
        dev,
        {"alpha": [0.1], "cutoff": [0]},
        scoring="macro_f1",
    )

    assert results[0]["dev_score"] == results[0]["dev_macro_f1"]


def test_grid_search_rejects_empty_grid():
    with pytest.raises(ValueError):
        grid_search_pos(DummyTagger, [], [], {})


def test_grid_search_rejects_empty_parameter_values():
    with pytest.raises(ValueError):
        grid_search_pos(
            DummyTagger,
            [],
            [],
            {"alpha": [], "cutoff": [0]},
        )


def test_grid_search_rejects_unsupported_metric():
    with pytest.raises(ValueError):
        grid_search_pos(
            DummyTagger,
            [],
            [],
            {"alpha": [0.1], "cutoff": [0]},
            scoring="unknown_metric",
        )


def test_grid_search_does_not_accept_test_data():
    import inspect

    parameters = inspect.signature(grid_search_pos).parameters
    assert "test_data" not in parameters
