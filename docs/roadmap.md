# Roadmap

Phased plan for elevating this from a course-project comparison study to something with broader research utility.

## Phase 1 — Modeling upgrades (1–2 weeks)

- **TF-IDF vectorizer** alongside `CountVectorizer` — usually better for unbalanced spam corpora.
- **Bigrams / character n-grams** for adversarial robustness.
- **Hyperparameter tuning** via `GridSearchCV` (logistic regression C, SVM gamma, NB alpha).
- **Class weighting** (`class_weight='balanced'`) — penalize false positives more heavily.

## Phase 2 — Deep learning baseline (2 weeks)

- Fine-tune `distilbert-base-uncased` on the raw email corpus.
- Compare against the classical models on the same train/test split.
- Report training time, inference latency, parameter count.

## Phase 3 — Adversarial robustness (1 week)

- Generate adversarial spam via whitespace tricks, character substitution, Unicode look-alikes.
- Measure model accuracy on adversarial vs. clean test set.
- Document failure modes.

## Phase 4 — Header + metadata features (1 week)

- Parse email headers via `email.parser`.
- Engineer features: sender domain reputation, SPF/DKIM/DMARC alignment, header anomalies.
- Concatenate with text features for a hybrid model.

## Phase 5 — Online learning + retrain schedule (1 week)

- Wrap `SGDClassifier.partial_fit` for incremental updates.
- Simulate a stream: new spam wave at week N, measure how fast the model adapts.

## Phase 6 — Calibration + cost-sensitive evaluation (1 week)

- Platt scaling / isotonic regression to calibrate probabilities.
- Asymmetric cost matrix: false positives cost 10× false negatives.
- Report expected cost per email, not just accuracy.

## Phase 7 — Deployment harness

- FastAPI endpoint: `POST /classify {"email_text": "..."}` → `{"label": "spam|ham", "probability": float}`.
- Containerize with Docker.
- Latency benchmark: target < 50ms p99.

## Out of scope

- **Replacing Gmail / SpamAssassin.** Modern email-provider spam filters use header-, content-, reputation-, ML-, and rule-based signals across billions of emails per day. This is a pedagogical project.
- **Production-grade adversarial defense.** Real systems use red-team workflows and continuous adversarial training.
