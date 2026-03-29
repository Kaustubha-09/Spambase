"""
Spam Filter — Main Entry Point

Usage:
    python main.py                          # Train all models, print comparison, save plots
    python main.py --email "Win a prize!"   # Classify a single email string
    python main.py --no-plots               # Skip generating plots
"""
import argparse
import re
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from configuration import DATA_PATH, LABEL_COL, NUM_FEATURES, FEATURE_NAMES
from data_import import load_data, train_manual_nb, evaluate_manual_nb
from visualizations import (
    plot_confusion_matrices, plot_roc_curves, plot_top_spam_words, plot_model_comparison,
)

# ── Feature names for the 48 word features ────────────────────────────────────
_WORD_FEATURES = [f.replace("word_freq_", "") for f in FEATURE_NAMES[:48]]
_CHAR_FEATURES = [";", "(", "[", "!", "$", "#"]   # maps to FEATURE_NAMES[48:54]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_spambase():
    """Return feature matrix X, label vector y, and the full DataFrame."""
    df = pd.read_csv(DATA_PATH, header=None)
    X = df.iloc[:, :NUM_FEATURES].values
    y = df.iloc[:, LABEL_COL].values
    return X, y, df


# ── Feature extraction from raw text (for CLI) ────────────────────────────────

def extract_features(text: str) -> np.ndarray:
    """Extract Spambase-style feature vector from a raw email string."""
    words = re.findall(r"[a-zA-Z0-9]+", text)
    total_words = max(len(words), 1)
    total_chars = max(len(text), 1)
    words_lower = [w.lower() for w in words]

    word_feats = [
        100.0 * words_lower.count(w) / total_words
        for w in _WORD_FEATURES
    ]
    char_feats = [
        100.0 * text.count(c) / total_chars
        for c in _CHAR_FEATURES
    ]

    capital_runs = re.findall(r"[A-Z]+", text)
    if capital_runs:
        lengths = [len(r) for r in capital_runs]
        avg_cap = sum(lengths) / len(lengths)
        longest_cap = float(max(lengths))
        total_cap = float(sum(lengths))
    else:
        avg_cap = longest_cap = total_cap = 0.0

    return np.array(word_feats + char_feats + [avg_cap, longest_cap, total_cap])


# ── Model training & evaluation ───────────────────────────────────────────────

def train_and_evaluate(name, clf, X_train, X_test, y_train, y_test):
    """Fit clf, predict on test set, return metrics dict."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
    else:
        y_proba = clf.decision_function(X_test)

    return {
        "clf": clf,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def run_sklearn_models(X, y):
    """Train Bernoulli NB, Logistic Regression, and SVM. Return results and CV scores."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_defs = {
        "Bernoulli NB": BernoulliNB(),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000))),
        ]),
    }

    results = {}
    cv_scores = {}

    for name, clf in model_defs.items():
        results[name] = train_and_evaluate(name, clf, X_train, X_test, y_train, y_test)
        scores = cross_val_score(clone(clf), X, y, cv=cv, scoring="accuracy")
        cv_scores[name] = scores

    return results, cv_scores


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results_table(results: dict, cv_scores: dict):
    header = (
        f"{'Model':<22} {'Accuracy':>9} {'Precision':>10}"
        f" {'Recall':>8} {'F1':>7} {'AUC-ROC':>9} {'CV Acc (5-fold)':>16}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for name, res in results.items():
        cv = cv_scores.get(name)
        cv_str = f"{cv.mean():.4f} ± {cv.std():.4f}" if cv is not None else "     N/A"
        print(
            f"{name:<22} {res['accuracy']:>9.4f} {res['precision']:>10.4f}"
            f" {res['recall']:>8.4f} {res['f1']:>7.4f} {res['auc']:>9.4f} {cv_str:>16}"
        )
    print(f"{sep}\n")


# ── CLI prediction ─────────────────────────────────────────────────────────────

def classify_email(text: str, clf) -> None:
    features = extract_features(text).reshape(1, -1)
    prediction = clf.predict(features)[0]
    label = "SPAM" if prediction == 1 else "HAM"

    if hasattr(clf, "predict_proba"):
        spam_prob = clf.predict_proba(features)[0][1]
        print(f"\n  Result:           {label}")
        print(f"  Spam probability: {spam_prob:.1%}\n")
    else:
        print(f"\n  Result: {label}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spam email classifier")
    parser.add_argument("--email", type=str, help="Raw email text to classify")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots")
    args = parser.parse_args()

    print("Loading Spambase dataset...")
    X, y, df = load_spambase()
    print(f"  {len(X)} emails — {y.sum()} spam ({y.mean():.1%}) | {(y == 0).sum()} ham\n")

    # ── Manual Naive Bayes (Kaustubha Eluri — fixed) ────────────────────────────────
    print("=== Manual Naive Bayes (Kaustubha Eluri — Fixed) ===")
    df_binary = load_data()
    manual_model = train_manual_nb(df_binary)
    manual_res = evaluate_manual_nb(manual_model)
    print(f"  Accuracy:  {manual_res['accuracy']:.4f}")
    print(f"  Precision: {manual_res['precision']:.4f}")
    print(f"  Recall:    {manual_res['recall']:.4f}")
    print(f"  F1 Score:  {manual_res['f1']:.4f}\n")

    # ── sklearn model comparison ───────────────────────────────────────────────
    print("=== sklearn Model Comparison ===")
    sklearn_results, cv_scores = run_sklearn_models(X, y)
    print_results_table(sklearn_results, cv_scores)

    # ── CLI email classification ───────────────────────────────────────────────
    best_name = max(sklearn_results, key=lambda k: sklearn_results[k]["f1"])
    best_clf = sklearn_results[best_name]["clf"]

    if args.email:
        print(f"Classifying with best model ({best_name}):")
        classify_email(args.email, best_clf)
        return

    # ── Visualizations ─────────────────────────────────────────────────────────
    if not args.no_plots:
        print("Generating plots...")
        plot_confusion_matrices(sklearn_results)
        plot_roc_curves(sklearn_results)
        plot_model_comparison(sklearn_results)

        spam_df = df[df[LABEL_COL] == 1]
        ham_df = df[df[LABEL_COL] == 0]
        spam_means = spam_df.iloc[:, : len(FEATURE_NAMES)].mean().values
        ham_means = ham_df.iloc[:, : len(FEATURE_NAMES)].mean().values
        plot_top_spam_words(FEATURE_NAMES, spam_means, ham_means)
        print()


if __name__ == "__main__":
    main()
