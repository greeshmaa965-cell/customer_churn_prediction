"""
data_loader.py
--------------
Loads the raw Telco Customer Churn CSV and performs
initial cleaning (type coercion, dropping unused columns).
"""

import pandas as pd
from pathlib import Path

# Default path relative to project root
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw_data(filepath: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw data from CSV.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        A cleaned pandas DataFrame ready for feature engineering.
    """
    df = pd.read_csv(filepath)

    # Fix TotalCharges — some rows have empty strings (new customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # customerID is not a predictive feature
    df.drop("customerID", axis=1, inplace=True)

    return df


def get_features_and_target(df: pd.DataFrame):
    """Split dataframe into features X and target y.

    Args:
        df: Cleaned dataframe with a 'Churn' column.

    Returns:
        Tuple (X, y) where y is binary (1=Churn, 0=No Churn).
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})
    return X, y


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Data shape: {df.shape}")
    print(df.head(3))
