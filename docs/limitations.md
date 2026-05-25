# Limitations

This is a classical-ML course project. Honest about what it is and isn't.

## Dataset

- **Spambase dataset is from 1999.** Spam techniques have evolved dramatically since then; modern spam includes images, polymorphic content, and adversarial perturbations that this model cannot see.
- **Raw email corpus is small.** 1,035 emails (286 spam, 749 ham). Not enough to reflect production traffic distributions.
- **English only.** No multilingual evaluation.
- **No adversarial test set.** No measurement of how the model fares against intentionally crafted spam designed to evade detection (whitespace obfuscation, character substitution, Unicode tricks).

## Modeling

- **Bag-of-words only.** No n-grams, no character-level features, no embeddings. Modern spam classifiers use neural encoders (CNNs over character sequences, transformer-based text encoders, sometimes image OCR for image-spam).
- **No header / metadata features.** Real spam filters use sender reputation, SPF/DKIM/DMARC alignment, IP reputation, sending-volume patterns. We use email body text only.
- **No model ensembling.** We compare models; we don't ensemble them. Stacking + calibration would improve over any single model.
- **No hyperparameter tuning.** Default sklearn hyperparameters across all models. No grid search, no Bayesian optimization.
- **No model calibration.** Reported probabilities (e.g., from `predict_proba`) aren't calibrated — `0.8` doesn't mean "80% chance of spam".

## Evaluation

- **Single train/test split + 5-fold CV.** Not stratified by spam ratio (though our 286/749 ratio is preserved by `train_test_split` defaults).
- **Accuracy / precision / recall numbers are for one specific split.** Run-to-run variance from minor sklearn version differences is real.
- **No latency measurement.** Inference time isn't profiled — relevant in real spam-filter deployment.
- **No memory profiling.** Vectorizer + models load full corpus into memory; not validated at million-email scale.

## Operational

- **Not deployed.** No production endpoint. No retraining schedule. No drift monitoring.
- **No incremental learning.** A new spam wave requires a full retrain. Real systems use online learning (`SGDClassifier.partial_fit` or active learning loops).
- **No false-positive cost weighting.** Spam classifiers should bias against false positives (don't send a real email to spam folder). We don't currently weight the loss function asymmetrically.

## Honest scope

This is a **CS5002 team project**, intended to demonstrate the classical ML pipeline:
1. text → vectorize → classify
2. compare multiple algorithms
3. report metrics + visualizations

It is **not** a production spam filter. Modern production spam classifiers are neural, header-aware, sender-reputation-driven, online-learned, and adversarially trained.
