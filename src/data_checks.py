"""
Data audit and sanity checks for Electrical Grid Stability dataset.
"""

from pathlib import Path
import pandas as pd


def audit_dataset(filepath: str | Path) -> dict:
    df = pd.read_csv(filepath)
    report = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": df["stabf"].value_counts().to_dict(),
        "class_percentages": (df["stabf"].value_counts(normalize=True) * 100).round(2).to_dict(),
    }
    return report


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "electrical_grid_stability.csv"
    res = audit_dataset(data_path)
    print("Dataset Shape:", res["shape"])
    print("Class Counts:", res["class_counts"])
    print("Class Percentages (%):", res["class_percentages"])
    print("Missing Values Total:", sum(res["missing_values"].values()))
    print("Duplicates:", res["duplicate_rows"])
