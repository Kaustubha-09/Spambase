"""
CS5002 Fall 2023 SV
Final Project
Kaustubha Eluri

- Spam/ham split filters by label column (57), not by row position.
- Probabilities are reset for each email (no cross-email contamination).
- Log-space arithmetic prevents floating-point underflow.
- Laplace smoothing prevents zero-probability words from zeroing the posterior.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from configuration import DATA_PATH, LABEL_COL, NUM_WORD_FEATURES


def load_data():
    """Load Spambase and binarize word/char frequency columns (0-53)."""
    df = pd.read_csv(DATA_PATH, header=None)
    df.iloc[:, 0:NUM_WORD_FEATURES] = np.where(
        df.iloc[:, 0:NUM_WORD_FEATURES] > 0, 1, 0
    )
    return df


def train_manual_nb(df):
    """Train a Bernoulli Naive Bayes classifier on the Spambase dataset.

    Returns a model dict containing priors, per-class word probabilities,
    and the held-out test splits.
    """
    # Correct split: filter by label, not by row position
    spam_data = df[df[LABEL_COL] == 1]
    ham_data = df[df[LABEL_COL] == 0]

    spam_train, spam_test = train_test_split(spam_data, test_size=0.1, random_state=42)
    ham_train, ham_test = train_test_split(ham_data, test_size=0.1, random_state=42)

    total_train = len(spam_train) + len(ham_train)
    spam_prior = len(spam_train) / total_train
    ham_prior = len(ham_train) / total_train

    # Laplace smoothing: small epsilon prevents log(0)
    epsilon = 1e-6
    spam_word_probs = spam_train.iloc[:, 0:NUM_WORD_FEATURES].mean() + epsilon
    ham_word_probs = ham_train.iloc[:, 0:NUM_WORD_FEATURES].mean() + epsilon

    return {
        "spam_prior": spam_prior,
        "ham_prior": ham_prior,
        "spam_word_probs": spam_word_probs,
        "ham_word_probs": ham_word_probs,
        "spam_test": spam_test,
        "ham_test": ham_test,
    }


def evaluate_manual_nb(model):
    """Evaluate the manual Naive Bayes on held-out spam and ham test sets."""
    log_spam_prior = np.log(model["spam_prior"])
    log_ham_prior = np.log(model["ham_prior"])
    log_spam_probs = np.log(model["spam_word_probs"].values)
    log_ham_probs = np.log(model["ham_word_probs"].values)

    def classify(row):
        """Classify one email; log probs are summed (not multiplied) per email."""
        log_spam = log_spam_prior
        log_ham = log_ham_prior
        for i in range(NUM_WORD_FEATURES):
            if row[i] == 1:
                log_spam += log_spam_probs[i]
                log_ham += log_ham_probs[i]
        return "spam" if log_spam >= log_ham else "ham"

    tp = tn = fp = fn = 0

    for _, row in model["spam_test"].iterrows():
        if classify(row) == "spam":
            tp += 1
        else:
            fn += 1

    for _, row in model["ham_test"].iterrows():
        if classify(row) == "spam":
            fp += 1
        else:
            tn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    df = load_data()
    model = train_manual_nb(df)
    results = evaluate_manual_nb(model)

    print("=== Manual Naive Bayes (Fixed) ===")
    print(f"True Positive:  {results['true_positive']}")
    print(f"True Negative:  {results['true_negative']}")
    print(f"False Positive: {results['false_positive']}")
    print(f"False Negative: {results['false_negative']}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
