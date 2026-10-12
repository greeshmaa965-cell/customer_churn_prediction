"""
evaluate.py
-----------
Generates a full evaluation report for the trained model:
  - Confusion matrix
  - Classification report
  - ROC curve
  - Precision / Recall / F1 vs threshold plot

Results are saved to reports/figures/.

Usage:
    python -m src.evaluate
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.train import BEST_THRESHOLD, load_model
from src.preprocessor import PROCESSED_DIR

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def load_test_data():
    """Load saved test split from disk."""
    import pickle

    with open(PROCESSED_DIR / "x_test_processed.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open(PROCESSED_DIR / "y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
    return X_test, y_test


def plot_confusion_matrix(y_test, y_pred, save_path: Path):
    """Plot and save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — Logistic Regression (threshold=0.3)", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {save_path}")


def plot_roc_curve(y_test, y_prob, save_path: Path):
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"Logistic Regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random Guess")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Logistic Regression", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {save_path}")


def plot_threshold_analysis(y_test, y_prob, save_path: Path):
    """Plot Precision / Recall / F1 across thresholds."""
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, marker="o", label="Precision")
    ax.plot(thresholds, recalls, marker="o", label="Recall")
    ax.plot(thresholds, f1s, marker="o", label="F1-score")
    ax.axvline(x=BEST_THRESHOLD, linestyle="--", color="red", alpha=0.6, label=f"Selected threshold ({BEST_THRESHOLD})")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 at Different Thresholds", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {save_path}")


def run_full_evaluation(threshold: float = BEST_THRESHOLD) -> dict:
    """Run full evaluation and save figures.

    Returns:
        Dict of evaluation metrics.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading model and test data …")
    model = load_model()
    X_test, y_test = load_test_data()

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("\n[2/4] Computing metrics …")
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "threshold": threshold,
    }

    print(f"\n  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1-score  : {metrics['f1']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")

    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    print("\n[3/4] Saving plots …")
    plot_confusion_matrix(y_test, y_pred, FIGURES_DIR / "confusion_matrix.png")
    plot_roc_curve(y_test, y_prob, FIGURES_DIR / "roc_curve.png")
    plot_threshold_analysis(y_test, y_prob, FIGURES_DIR / "threshold_analysis.png")

    print("\n[4/4] Done!")
    return metrics


if __name__ == "__main__":
    run_full_evaluation()
