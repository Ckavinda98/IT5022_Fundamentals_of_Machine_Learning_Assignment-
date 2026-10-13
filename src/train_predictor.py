"""Train and save the Random Forest model used by the interactive app."""

from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
from src.train_models import get_random_forest_model


def train_model():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "electrical_grid_stability.csv"

    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = get_random_forest_model()
    model.fit(X, y)
    return model, len(df)


def train_and_save_model() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "outputs" / "models" / "random_forest.joblib"
    model, row_count = train_model()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Trained Random Forest on {row_count:,} labeled rows.")
    print(f"Saved model to: {model_path}")
    return model_path


if __name__ == "__main__":
    train_and_save_model()
