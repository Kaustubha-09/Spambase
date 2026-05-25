# Spambase — Portfolio Case Study

A four-model spam-classification study built for CS5002 (Northeastern). Skim time: 2 minutes.

## The brief

Build a spam email classifier and show the classical ML pipeline end-to-end: data ingestion → vectorization → classifier comparison → evaluation → visualization. Demonstrate both **how the algorithm works** (hand-rolled NB) and **how it's done in practice** (scikit-learn).

## The result

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | CV Accuracy |
|---|---|---|---|---|---|---|
| Manual Naive Bayes | 0.8243 | 0.6950 | 0.9890 | 0.8163 | — | — |
| Bernoulli NB | 0.8762 | 0.8716 | 0.8044 | 0.8367 | 0.9496 | 0.8868 ± 0.0081 |
| Logistic Regression | **0.9294** | 0.9209 | 0.8981 | **0.9093** | **0.9702** | **0.9246 ± 0.0068** |
| SVM | 0.9251 | 0.9224 | 0.8843 | 0.9030 | 0.9688 | 0.9213 ± 0.0062 |

On the raw email corpus (1,035 emails), Multinomial NB achieves **97% precision, 95% recall**.

## The engineering I'd defend

### 1. Two-path study: feature dataset + raw text

Spambase (4,601 rows × 57 engineered features) gives a stable benchmark for multi-model comparison. The raw .txt email corpus gives the full pipeline (text → vectorizer → classifier). Both paths exist on purpose; each demonstrates a different thing. See [decisions.md, ADR-001](decisions.md#adr-001--two-datasets-two-model-paths).

### 2. Hand-rolled Naive Bayes alongside the library

`data_import.py` implements NB by hand — log-space arithmetic, Laplace smoothing — even though `sklearn.naive_bayes.MultinomialNB` does the same. The algorithm is the lesson. Hiding it behind `clf.fit(X, y)` skips the pedagogical core. See [ADR-002](decisions.md#adr-002--hand-rolled-naive-bayes-alongside-scikit-learn).

### 3. Log-space for numerical stability

Multiplying ~10,000 small probabilities underflows to zero. Summing logs doesn't. This is one of the most important practical lessons in implementing NB. The hand-rolled implementation makes it visible. See [ADR-003](decisions.md#adr-003--log-space-probability-arithmetic).

### 4. Laplace smoothing for unseen events

A word that never appeared in ham would force `P(word | ham) = 0` and the entire posterior to zero. `α = 1.0` Laplace smoothing assigns a small non-zero probability to unseen events. See [ADR-004](decisions.md#adr-004--laplace-add-one-smoothing).

### 5. Comparison, not a single "winner"

We report four models because the lesson is the **tradeoff** — Manual NB has the highest recall (catches more spam) at the cost of precision; LR has the highest AUC-ROC; SVM is competitive but slower. The right model depends on what kind of error is worse. See [ADR-006](decisions.md#adr-006--multi-model-comparison-rather-than-one-winning-model).

### 6. CLI for fast experiments

```bash
python main.py --email "Win FREE iPhone!"
```

Anyone evaluating the project can ask *"how does it do on a string I made up?"* in one command. See [ADR-007](decisions.md#adr-007--cli-as-a-usability-surface).

## The honest part

- **Spambase is a 1999 dataset.** Modern spam looks very different. We're demonstrating the pipeline, not building a production filter.
- **Bag-of-words only.** Real spam classifiers use header features, sender reputation, n-grams, neural encoders, sometimes image OCR.
- **No hyperparameter tuning.** Default sklearn hyperparameters across the board.
- **No model calibration.** Reported probabilities aren't calibrated — `0.8` doesn't mean "80% spam".
- **Single train/test split.** 5-fold CV included, but not nested CV or held-out test set.

All of this is in [limitations.md](limitations.md). The point is to **demonstrate the classical ML pipeline well**, not to ship a competitor to SpamAssassin.

## What I'd do next

Roadmap Phase 1: TF-IDF vectorization (almost always beats raw counts on unbalanced spam corpora), bigrams for adversarial robustness, `class_weight='balanced'` to penalize false positives.

## What this signals to a recruiter

- I can run the full classical ML pipeline — vectorization, training, evaluation, visualization — and report results in a comparison table that's useful, not just impressive.
- I implement algorithms by hand when the algorithm is the point (Manual NB, log-space arithmetic, Laplace smoothing) and use the library when speed matters (the comparison).
- I write tests for the deterministic plumbing (`tests/` — 16 cases) and don't bake stochastic accuracy numbers into assertions.
- I know what spam classification looks like in 2024 versus what this project does, and I document the gap explicitly.
- I write ADRs even on a course project.
