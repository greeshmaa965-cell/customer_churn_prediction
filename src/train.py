"""
train.py
--------
Trains a Logistic Regression model on the processed Telco churn data
(as per notebook 03) and saves the best model to disk.

Usage:
    python -m src.train
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from src.data_loader import load_raw_data
from src.preprocessor import run_preprocessing_pipeline

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "logistic_regression_model.pkl"

# The threshold tuned in notebook 03 to maximise recall of churners
BEST_THRESHOLD = 0.3


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Fit Logistic Regression on the pre-processed training data.

    Args:
        X_train: Processed feature matrix (numeric/encoded).
        y_train: Binary target series (1=Churn, 0=No Churn).

    Returns:
        Fitted LogisticRegression estimator.
    """
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = BEST_THRESHOLD) -> dict:
    """Evaluate a fitted model and return a metrics dict.

    Args:
        model: Fitted classifier.
        X_test: Processed test features.
        y_test: True binary labels.
        threshold: Decision threshold (default 0.3 chosen in notebook).

    Returns:
        Dict with accuracy, roc_auc, threshold.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    return {
        "accuracy": round(accuracy, 4),
        "roc_auc": round(roc_auc, 4),
        "threshold": threshold,
    }


def save_model(model: LogisticRegression, path: Path = MODEL_PATH) -> None:
    """Pickle the trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {path}")


def load_model(path: Path = MODEL_PATH) -> LogisticRegression:
    """Load a saved model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    print("=" * 50)
    print("Customer Churn Prediction - Training Pipeline")
    print("=" * 50)

    # 1. Load & clean raw data
    print("\n[1/3] Loading raw data …")
    df = load_raw_data()
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

    # 2. Preprocess (split, encode, scale, save artifacts)
    print("\n[2/3] Running preprocessing pipeline …")
    artifacts = run_preprocessing_pipeline(df, save_artifacts=True)
    X_train = artifacts["X_train"]
    X_test = artifacts["X_test"]
    y_train = artifacts["y_train"]
    y_test = artifacts["y_test"]
    print(f"  Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # 3. Train & evaluate
    print("\n[3/3] Training Logistic Regression …")
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    print(f"\n  [OK] Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  [OK] ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  [OK] Threshold  : {metrics['threshold']}")

    save_model(model)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
