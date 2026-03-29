"""Unit tests for the spam filter.

Run with:
    pytest tests/
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import extract_features
from data_import import load_data, train_manual_nb, evaluate_manual_nb
from configuration import NUM_WORD_FEATURES, LABEL_COL, FEATURE_NAMES


# ── Feature extraction ────────────────────────────────────────────────────────

class TestFeatureExtraction:
    def test_output_shape(self):
        assert extract_features("Hello world!").shape == (57,)

    def test_empty_text_no_nan(self):
        features = extract_features("")
        assert features.shape == (57,)
        assert not np.any(np.isnan(features))

    def test_word_freq_free(self):
        idx = FEATURE_NAMES.index("word_freq_free")
        assert extract_features("free free money")[ idx] > 0
        assert extract_features("no such word here")[idx] == 0

    def test_char_freq_exclamation(self):
        idx = FEATURE_NAMES.index("char_freq_!")
        assert extract_features("Win now!!!")[ idx] > 0
        assert extract_features("no exclamation")[idx] == 0

    def test_capital_run_features_present(self):
        feats = extract_features("CLICK HERE NOW")
        assert feats[54] > 0   # average capital run length
        assert feats[55] > 0   # longest capital run
        assert feats[56] > 0   # total capital letters

    def test_no_capitals_zero_run_features(self):
        feats = extract_features("all lowercase text")
        assert feats[54] == 0
        assert feats[55] == 0
        assert feats[56] == 0

    def test_word_freq_proportional(self):
        # "free" appears 2/4 times = 50%
        idx = FEATURE_NAMES.index("word_freq_free")
        feats = extract_features("free money free spam")
        assert abs(feats[idx] - 50.0) < 1e-6


# ── Manual Naive Bayes ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_nb():
    df = load_data()
    model = train_manual_nb(df)
    results = evaluate_manual_nb(model)
    return df, model, results


class TestManualNaiveBayes:
    def test_priors_sum_to_one(self, trained_nb):
        _, model, _ = trained_nb
        assert abs(model["spam_prior"] + model["ham_prior"] - 1.0) < 1e-9

    def test_priors_in_valid_range(self, trained_nb):
        _, model, _ = trained_nb
        assert 0 < model["spam_prior"] < 1
        assert 0 < model["ham_prior"] < 1

    def test_word_probs_positive(self, trained_nb):
        _, model, _ = trained_nb
        assert (model["spam_word_probs"] > 0).all(), "Laplace smoothing should prevent zeros"
        assert (model["ham_word_probs"] > 0).all()

    def test_word_probs_correct_length(self, trained_nb):
        _, model, _ = trained_nb
        assert len(model["spam_word_probs"]) == NUM_WORD_FEATURES
        assert len(model["ham_word_probs"]) == NUM_WORD_FEATURES

    def test_metrics_in_valid_range(self, trained_nb):
        _, _, results = trained_nb
        for key in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= results[key] <= 1.0, f"{key} out of range"

    def test_confusion_matrix_totals_match_test_sizes(self, trained_nb):
        _, model, results = trained_nb
        tp, tn, fp, fn = (
            results["true_positive"], results["true_negative"],
            results["false_positive"], results["false_negative"],
        )
        assert tp + fn == len(model["spam_test"]), "TP+FN must equal spam test size"
        assert tn + fp == len(model["ham_test"]), "TN+FP must equal ham test size"

    def test_spam_test_contains_only_spam(self, trained_nb):
        _, model, _ = trained_nb
        assert (model["spam_test"][LABEL_COL] == 1).all(), "Spam test set contains non-spam rows"

    def test_ham_test_contains_only_ham(self, trained_nb):
        _, model, _ = trained_nb
        assert (model["ham_test"][LABEL_COL] == 0).all(), "Ham test set contains non-ham rows"

    def test_reasonable_accuracy(self, trained_nb):
        _, _, results = trained_nb
        assert results["accuracy"] > 0.80, "Expected >80% accuracy on fixed model"
