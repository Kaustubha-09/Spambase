from configuration import *


def spam_filter_train(X, y):
    """Train and evaluate a Multinomial Naive Bayes classifier on raw email text.

    Args:
        X: list of raw email strings
        y: list/array of labels (1=spam, 0=ham)

    Returns:
        clf: trained MultinomialNB classifier
        vectorizer: fitted CountVectorizer
        metrics_dict: accuracy, precision, recall, F1, AUC-ROC, CV scores
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    y_proba = clf.predict_proba(X_test_vec)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    # 5-fold stratified cross-validation
    pipeline = Pipeline([("vec", CountVectorizer()), ("clf", MultinomialNB())])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return clf, vectorizer, {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
