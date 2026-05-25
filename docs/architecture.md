# Architecture

A four-model spam-classification study built on two datasets — the canonical 4,601-row Spambase feature dataset and a 1,035-email raw-text corpus. Implements Multinomial / Bernoulli / Logistic Regression / SVM classifiers plus a hand-rolled Naive Bayes in log space.

## Pipeline

```
                  ┌────────────────────────────────────────┐
                  │           Data sources                 │
                  │  spambase/spambase.data (4,601 rows)   │
                  │  test database/ham/ (749 .txt emails)  │
                  │  test database/spam/ (286 .txt emails) │
                  └────────────────┬───────────────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────────────┐
                  │      data_import.py + helper.py        │
                  │  - get_files_path: recursive walker    │
                  │  - read with latin-1 / iso-8859-1 /    │
                  │    utf-8 fallback                      │
                  └────────────────┬───────────────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────────────┐
                  │           configuration.py             │
                  │  Centralized constants & imports       │
                  │  TEST_SIZE = 0.25, RANDOM_STATE = 42   │
                  │  CV_FOLDS = 5                          │
                  └────────────────┬───────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌──────────────────┐    ┌──────────────────────┐   ┌──────────────────────┐
│  beyes.py        │    │   main.py            │   │ data_import.py       │
│  Manual Naive    │    │   Multi-model        │   │ Manual NB on the     │
│  Bayes —         │    │   benchmarking on    │   │ raw-text corpus      │
│  log-space arith │    │   Spambase features  │   │ (alternate path)     │
│  + Laplace       │    │   BernoulliNB · LR · │   │                      │
│  smoothing       │    │   SVM                │   │                      │
└──────────────────┘    └──────────┬───────────┘   └──────────────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────────────┐
                  │       visualizations.py                │
                  │  Saved to plots/:                      │
                  │   confusion_matrices.png               │
                  │   roc_curves.png                       │
                  │   model_comparison.png                 │
                  │   top_spam_ham_words.png               │
                  └────────────────┬───────────────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────────────┐
                  │            CLI surface                 │
                  │  python main.py                        │
                  │  python main.py --email "…"            │
                  │  python main.py --no-plots             │
                  └────────────────────────────────────────┘
```

## Modules

| File | Responsibility |
|---|---|
| `main.py` | Entry point. Multi-model training, evaluation, comparison table, plot generation, CLI `--email` classifier |
| `beyes.py` | Naive Bayes training/classification wrappers around scikit-learn (Multinomial NB on `CountVectorizer` output) |
| `data_import.py` | Manual Naive Bayes — hand-rolled Bayesian classifier with log-space arithmetic and Laplace smoothing |
| `configuration.py` | Centralized constants, shared imports |
| `helper.py` | `get_files_path()` walker + text-loading utilities with encoding fallback |
| `visualizations.py` | Plot generation — confusion matrices, ROC curves, top spam/ham word charts, model-comparison bar |
| `tests/test_spam_filter.py` | Unit tests — 16 cases covering vectorization, manual NB log-prob arithmetic, file loading |

## The two model paths

| Path | Input | Models | Where |
|---|---|---|---|
| **Spambase (feature)** | 57 engineered features × 4,601 rows | Manual NB · Bernoulli NB · Logistic Regression · SVM | `main.py` |
| **Raw text** | 1,035 .txt emails (286 spam + 749 ham) | Multinomial NB on `CountVectorizer` | `beyes.py` (programmatic) |

The feature dataset path is the comparison study (4 models side-by-side). The raw-text path is the classic NB pipeline that produces the 97% precision / 95% recall result on the original test database.

## Manual Naive Bayes (`data_import.py`)

```python
def calc_log_prob(word_count, class_word_count, vocab_size, alpha=1.0):
    """Laplace-smoothed log probability of a word given a class."""
    return log((word_count + alpha) / (class_word_count + alpha * vocab_size))
```

- Log-space arithmetic — prevents underflow when multiplying ~10,000 token probabilities.
- Laplace (add-one) smoothing — handles tokens unseen in a class.
- Hand-rolled prior + likelihood computation; uses scikit-learn only for `train_test_split`.

The hand-rolled version is pedagogical — the scikit-learn `MultinomialNB` does the same thing faster. We keep both for the comparison and so the architecture reads "here is how the algorithm works" without hiding it behind a library call.

## Evaluation protocol

| Property | Value |
|---|---|
| Train/test split | 75% / 25%, `random_state=42` |
| Cross-validation | 5-fold (where applicable) |
| Metrics | Accuracy, Precision, Recall, F1, AUC-ROC, CV accuracy ± std |
| Visualizations | Confusion matrix per model · ROC curves overlay · top-20 spam/ham words bar · accuracy comparison bar |

## CLI

```bash
python main.py                                    # full pipeline + plots
python main.py --email "Win FREE iPhone now!"      # classify one string
python main.py --no-plots                          # skip plot generation
```

`--email` flag re-uses the trained Multinomial NB to classify a raw string in under a second.

## What runs where

| Concern | Lives in |
|---|---|
| Spambase feature dataset loading | `main.py` (`pd.read_csv` on `spambase.data`) |
| Raw email corpus loading | `helper.py::get_files_path` + per-encoding open |
| Vectorization (BoW) | `CountVectorizer` in `beyes.py` |
| Manual NB | `data_import.py` |
| Library models | scikit-learn `MultinomialNB`, `BernoulliNB`, `LogisticRegression`, `SVC` |
| Evaluation | `main.py` + `sklearn.metrics` |
| Plots | `visualizations.py` |
| Tests | `tests/test_spam_filter.py` (pytest, 16 cases) |
