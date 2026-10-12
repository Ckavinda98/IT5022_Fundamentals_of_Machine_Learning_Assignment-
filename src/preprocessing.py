"""
Leakage-safe preprocessing and stratified dataset splitting.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "tau1", "tau2", "tau3", "tau4",
    "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4"
]
TARGET_COLUMN = "stabf"


def load_and_split_data(
    filepath: str | Path,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads dataset, isolates the 12 operating features, strictly discards 'stab'
    to prevent target leakage, and splits with stratification.
    """
    df = pd.read_csv(filepath)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "electrical_grid_stability.csv"
    X_tr, X_te, y_tr, y_te = load_and_split_data(data_path)
    print("Train Shape:", X_tr.shape, y_tr.shape)
    print("Test Shape:", X_te.shape, y_te.shape)
    print("Train Target Distribution:\n", y_tr.value_counts(normalize=True))
    print("Test Target Distribution:\n", y_te.value_counts(normalize=True))
