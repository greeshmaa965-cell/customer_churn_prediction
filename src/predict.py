"""
predict.py
----------
Provides single-sample and batch prediction utilities using
the saved model and preprocessor — used by the Streamlit app.

Usage:
    from src.predict import predict_single
"""

import pickle
from pathlib import Path

import pandas as pd

from src.train import BEST_THRESHOLD, load_model
from src.preprocessor import load_preprocessor

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def predict_single(input_dict: dict, threshold: float = BEST_THRESHOLD) -> dict:
    """Predict churn probability for a single customer.

    Args:
        input_dict: Dict mapping raw feature names → values.
                    Example:
                        {
                          "gender": "Female",
                          "SeniorCitizen": 0,
                          "Partner": "Yes",
                          ...
                        }
        threshold: Decision threshold (default 0.3).

    Returns:
        Dict with keys:
            "churn_probability": float  (0–1)
            "churn_prediction":  int    (0 or 1)
            "risk_label":        str    ("Low" | "Medium" | "High")
    """
    model = load_model()
    preprocessor = load_preprocessor()

    X_raw = pd.DataFrame([input_dict])

    # Fix TotalCharges type
    if "TotalCharges" in X_raw.columns:
        X_raw["TotalCharges"] = pd.to_numeric(X_raw["TotalCharges"], errors="coerce").fillna(0)

    X_processed = preprocessor.transform(X_raw)
    feature_names = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    prob = model.predict_proba(X_processed_df)[0, 1]
    prediction = int(prob >= threshold)

    if prob < 0.35:
        risk = "Low"
    elif prob < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "churn_probability": round(float(prob), 4),
        "churn_prediction": prediction,
        "risk_label": risk,
    }


def predict_batch(df_raw: pd.DataFrame, threshold: float = BEST_THRESHOLD) -> pd.DataFrame:
    """Predict churn for a batch DataFrame of raw (unprocessed) customers.

    Args:
        df_raw: Raw feature DataFrame (same schema as training data, without customerID/Churn).
        threshold: Decision threshold.

    Returns:
        Input DataFrame with additional columns:
            churn_probability, churn_prediction, risk_label.
    """
    model = load_model()
    preprocessor = load_preprocessor()

    if "TotalCharges" in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce").fillna(0)

    X_processed = preprocessor.transform(df_raw)
    feature_names = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=df_raw.index)

    probs = model.predict_proba(X_processed_df)[:, 1]
    preds = (probs >= threshold).astype(int)

    df_out = df_raw.copy()
    df_out["churn_probability"] = probs.round(4)
    df_out["churn_prediction"] = preds

    def risk_label(p):
        if p < 0.35:
            return "Low"
        elif p < 0.60:
            return "Medium"
        return "High"

    df_out["risk_label"] = df_out["churn_probability"].apply(risk_label)
    return df_out
