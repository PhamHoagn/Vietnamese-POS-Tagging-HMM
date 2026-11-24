import numpy as np

from scripts.diagnose_dev_unk import hapax_emissions
from src.hmm_bigram import BigramHMMTagger


def test_only_global_singletons_contribute_unk_counts_and_rows_stay_normalized():
    train = [(["a", "b", "c", "a"], ["NOUN", "NOUN", "VERB", "VERB"])]
    model = BigramHMMTagger(alpha=1.0, cutoff=0).fit(train)
    # Each tag has two observed tokens and one hapax, then four smoothing counts.
    expected = np.array([[2, 2, 1, 2], [2, 1, 2, 2]]) / 7
    np.testing.assert_allclose(np.exp(hapax_emissions(model)), expected)
    np.testing.assert_array_equal(model.emission_counts_[:, model.unk_index_], 0)
