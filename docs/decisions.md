# Architecture Decision Records

Dated decisions, append-only.

## ADR-001 · Two datasets, two model paths

**Date:** 2024-12
**Status:** Accepted

The project uses **both** the engineered 57-feature Spambase dataset (4,601 rows) and a raw .txt email corpus (1,035 emails). The Spambase path runs Bernoulli NB / Logistic Regression / SVM; the raw-text path runs Multinomial NB on `CountVectorizer` output.

**Why:**
- The engineered Spambase dataset lets us compare classifiers on a stable, well-studied benchmark.
- The raw .txt corpus lets us demonstrate the full feature-extraction pipeline (text → BoW → vectorizer → classifier) end-to-end.
- The pedagogical value is in showing both — the model-comparison story on Spambase and the NB pipeline on raw text.

**Cost:** two evaluation tables, two reporting paths. Acceptable.

---

## ADR-002 · Hand-rolled Naive Bayes alongside scikit-learn

**Date:** 2024-12
**Status:** Accepted

`data_import.py` implements Naive Bayes by hand — log-space probability arithmetic, Laplace smoothing — even though `sklearn.naive_bayes.MultinomialNB` does the same thing.

**Why:** the algorithm is the lesson. Hiding it behind `clf.fit(X, y)` skips the pedagogical core (Bayes' theorem, why log-space, why smoothing). The manual implementation reads top-to-bottom.

**Cost:** ~50 extra lines. Acceptable — and the library models still run for the comparison.

---

## ADR-003 · Log-space probability arithmetic

**Date:** 2024-12
**Status:** Accepted

Manual NB sums log-probabilities instead of multiplying probabilities.

**Why:** spam classification multiplies ~10,000 token probabilities per email. Most token probabilities are near zero. Multiplying 10,000 numbers near zero underflows to `0.0` in any reasonable float type. Adding their logs is numerically stable.

This is one of the most important practical lessons in implementing NB; the comment in `data_import.py` documents it.

---

## ADR-004 · Laplace (add-one) smoothing

**Date:** 2024-12
**Status:** Accepted

`P(word | class) = (count + α) / (total_count + α × vocab_size)` with `α = 1.0`.

**Why:** without smoothing, a word that never appears in (say) ham gets `P(word | ham) = 0`, which forces the entire posterior to zero — even if 50 other words in the email screamed ham. Laplace smoothing assigns a small non-zero probability to unseen events.

**Cost:** slight regularization toward uniform. Acceptable, and `α = 1.0` is the standard choice.

---

## ADR-005 · 75/25 train/test split with `random_state=42`

**Date:** 2024-12
**Status:** Accepted

`train_test_split(X, y, test_size=0.25, random_state=42)`.

**Why:** standard scikit-learn defaults. The fixed seed means results are reproducible across runs.

---

## ADR-006 · Multi-model comparison rather than one "winning" model

**Date:** 2024-12
**Status:** Accepted

We report Manual NB / Bernoulli NB / Logistic Regression / SVM side by side rather than picking one.

**Why:** the lesson is the comparison. NB is fast and interpretable; LR is the strongest performer on this dataset by AUC-ROC; SVM is competitive but slower; Manual NB has the highest recall (catches more spam) at the cost of precision. Each makes a different precision/recall tradeoff — the right model depends on what kind of error is worse.

---

## ADR-007 · CLI as a usability surface

**Date:** 2024-12
**Status:** Accepted

`main.py --email "Win FREE iPhone!"` returns a classification + spam probability in under a second.

**Why:** anyone evaluating the project should be able to ask *"how does it do on a string I made up?"* without reading the code. The CLI is the cheapest way to make that question one command.

---

## ADR-008 · Save plots to `plots/`, don't `plt.show()`

**Date:** 2024-12
**Status:** Accepted

`visualizations.py` writes `plots/{name}.png` and never calls `plt.show()`.

**Why:** the pipeline runs in CI / on a headless server. `plt.show()` blocks the process. Saving PNGs means the output is reviewable without an interactive display, and the README can reference them.

---

## ADR-009 · Tests cover the deterministic parts

**Date:** 2024-12
**Status:** Accepted

`tests/test_spam_filter.py` covers:
- `CountVectorizer` output shape on known fixed inputs.
- Manual NB log-probability arithmetic on hand-computed examples.
- `helper.get_files_path` directory walking.
- File loader handling of multiple encodings.

The tests **don't** assert specific accuracy numbers from training. Training is stochastic enough that `random_state` pinning doesn't fully eliminate run-to-run variance on some scikit-learn versions; baking specific accuracies into tests would make them flaky on minor library upgrades.
