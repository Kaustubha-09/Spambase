"""Visualization functions for the spam filter project."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

PLOTS_DIR = "plots"


def _save(filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices(results: dict):
    """Confusion matrix heatmap for each model, side by side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["y_test"], res["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"],
        )
        ax.set_title(f"{name}\nAcc: {res['accuracy']:.3f}  F1: {res['f1']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    _save("confusion_matrices.png")


def plot_roc_curves(results: dict):
    """ROC curves for all models overlaid on one chart."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, res in results.items():
        if "y_proba" not in res:
            continue
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
        auc = roc_auc_score(res["y_test"], res["y_proba"])
        ax.plot(fpr, tpr, linewidth=2, label=f"{name}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("roc_curves.png")


def plot_top_spam_words(feature_names: list, spam_means: np.ndarray, ham_means: np.ndarray, top_n: int = 20):
    """Side-by-side bar charts: top spam indicators vs top ham indicators."""
    diff = np.array(spam_means) - np.array(ham_means)

    def short(name):
        return name.replace("word_freq_", "").replace("char_freq_", "")

    # Top spam words (highest positive diff)
    spam_idx = np.argsort(diff)[-top_n:][::-1]
    spam_labels = [short(feature_names[i]) for i in spam_idx]
    spam_vals = diff[spam_idx]

    # Top ham words (most negative diff)
    ham_idx = np.argsort(diff)[:top_n]
    ham_labels = [short(feature_names[i]) for i in ham_idx]
    ham_vals = -diff[ham_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(spam_labels[::-1], spam_vals[::-1], color="#e74c3c")
    axes[0].set_title(f"Top {top_n} Spam Indicators\n(spam freq − ham freq)")
    axes[0].set_xlabel("Frequency Difference (%)")

    axes[1].barh(ham_labels[::-1], ham_vals[::-1], color="#3498db")
    axes[1].set_title(f"Top {top_n} Ham Indicators\n(ham freq − spam freq)")
    axes[1].set_xlabel("Frequency Difference (%)")

    plt.tight_layout()
    _save("top_spam_ham_words.png")


def plot_model_comparison(results: dict):
    """Grouped bar chart comparing accuracy, precision, recall, and F1 across models."""
    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [results[m].get(metric, 0) for m in model_names]
        bars = ax.bar(x + i * width, vals, width, label=label)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=10)
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save("model_comparison.png")
