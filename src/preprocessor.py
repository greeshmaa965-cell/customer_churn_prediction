"""
preprocessor.py
---------------
Builds and fits the sklearn ColumnTransformer pipeline used
across notebook 02 (feature engineering).  The fitted preprocessor
can be saved and reused at inference time.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PREPROCESSOR_PATH = Path(__file__).resolve().parents[1] / "models" / "preprocessor.pkl"


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer for the Telco churn features.

    Categorical columns → OneHotEncoder(handle_unknown='ignore')
    Numerical columns   → StandardScaler()

    Args:
        X_train: Training feature DataFrame (used to detect col types).

    Returns:
        An *unfitted* ColumnTransformer.
    """
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    return preprocessor


def fit_and_transform(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """Fit the preprocessor on training data and transform both splits.

    Args:
        X_train: Raw training features.
        X_test:  Raw test features.

    Returns:
        Tuple (preprocessor, X_train_processed, X_test_processed)
        where processed frames are dense DataFrames with named columns.
    """
    preprocessor = build_preprocessor(X_train)
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test.index)

    return preprocessor, X_train_processed, X_test_processed


def save_preprocessor(preprocessor: ColumnTransformer, path: Path = PREPROCESSOR_PATH) -> None:
    """Persist fitted preprocessor to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved -> {path}")


def load_preprocessor(path: Path = PREPROCESSOR_PATH) -> ColumnTransformer:
    """Load a saved preprocessor from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
):
    """End-to-end preprocessing: split → encode → scale → optionally save.

    Args:
        df: Cleaned dataframe (customerID already dropped, TotalCharges numeric).
        target_col: Name of the target column.
        test_size: Fraction of data for the test split.
        random_state: Reproducibility seed.
        save_artifacts: If True, save processed splits and preprocessor to disk.

    Returns:
        Dict with keys: preprocessor, X_train, X_test, y_train, y_test.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col].map({"Yes": 1, "No": 0})

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor, X_train_proc, X_test_proc = fit_and_transform(X_train_raw, X_test_raw)

    if save_artifacts:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name, obj in [
            ("x_train_processed.pkl", X_train_proc),
            ("x_test_processed.pkl", X_test_proc),
            ("y_train.pkl", y_train),
            ("y_test.pkl", y_test),
        ]:
            with open(PROCESSED_DIR / name, "wb") as f:
                pickle.dump(obj, f)
        save_preprocessor(preprocessor)
        print("All processed artifacts saved.")

    return {
        "preprocessor": preprocessor,
        "X_train": X_train_proc,
        "X_test": X_test_proc,
        "y_train": y_train,
        "y_test": y_test,
    }
