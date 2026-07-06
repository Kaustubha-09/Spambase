# Spambase — Interview Talking Points

**Why hand-roll Naive Bayes when sklearn has `MultinomialNB`?** Because the algorithm is the lesson. Hiding NB behind `clf.fit(X, y)` skips Bayes' theorem, log-space arithmetic, and Laplace smoothing — the three things you need to *implement* NB, not just use it. The library version is faster and gets used in `main.py` for the comparison. The hand-rolled version exists in `data_import.py` to demonstrate the math.

**Log-space arithmetic and why it matters.** Spam classification multiplies ~10,000 token probabilities per email. Most are near zero. Multiplying 10,000 near-zero numbers underflows to `0.0` in any float type. Summing their logs is numerically stable. This is one of the most important practical lessons in implementing NB by hand — every production NB implementation does this.

**Laplace smoothing as the unseen-event defense.** Without smoothing, a word that never appeared in (say) the ham training set gets `P(word | ham) = 0`. Multiply that into the posterior and the entire ham probability collapses to zero — even if 50 other words in the email screamed ham. `α = 1.0` Laplace smoothing assigns a small non-zero probability to unseen events. This is the difference between a working classifier and a brittle one.

**Why compare four models instead of picking a winner.** The lesson is the tradeoff. Manual NB has the highest recall (catches 99% of spam) at 70% precision (lots of false positives). LR has the highest AUC-ROC and balanced precision/recall. SVM is competitive but slower. The right model depends on whether sending a legit email to spam (false positive) is worse than letting spam through (false negative). A real spam filter would weight the loss function asymmetrically — a roadmap item.

**The honest scope.** Spambase is a 1999 dataset. Modern spam looks very different (image spam, polymorphic content, header-aware evasion). This project demonstrates the classical ML pipeline — vectorization, training, evaluation, visualization — at a depth that's pedagogically useful. It is not a production spam filter and the README documents the gap explicitly.


