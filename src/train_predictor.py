"""Train and save the Random Forest model used by the interactive app."""

from pathlib import Path

import joblib

from src.preprocessing import load_and_split_data
from src.train_models import get_random_forest_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "electrical_grid_stability.csv"


def train_model():
    """
    Fits the Random Forest on the same 80% training split used in the report, so the
    20% test split stays unseen and the app can show honest held-out predictions.
    """
    X_train, _, y_train, _ = load_and_split_data(DATA_PATH)

    model = get_random_forest_model()
    model.fit(X_train, y_train)
    return model, len(X_train)


def train_and_save_model() -> Path:
    model_path = PROJECT_ROOT / "outputs" / "models" / "random_forest.joblib"
    model, row_count = train_model()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Trained Random Forest on {row_count:,} training rows (test split held out).")
    print(f"Saved model to: {model_path}")
    return model_path


if __name__ == "__main__":
    train_and_save_model()
