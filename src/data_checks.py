"""
Data audit and sanity checks for Electrical Grid Stability dataset.
"""

from pathlib import Path
import pandas as pd

# Documented generation ranges of the simulation (Arzamasov et al., 2018).
EXPECTED_RANGES = {
    "tau": (0.5, 10.0),   # reaction times tau1-tau4 (s)
    "p_consumer": (-2.0, -0.5),  # consumed power p2-p4
    "g": (0.05, 1.0),     # price elasticity g1-g4
}


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


def validity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks every feature against its documented simulation range, the power-balance
    constraint (p1 + p2 + p3 + p4 = 0) and the stab -> stabf labelling rule.
    Returns one row per check with the number of violating records.
    """
    tau_cols = ["tau1", "tau2", "tau3", "tau4"]
    p_cons_cols = ["p2", "p3", "p4"]
    g_cols = ["g1", "g2", "g3", "g4"]

    lo, hi = EXPECTED_RANGES["tau"]
    tau_bad = int(((df[tau_cols] < lo) | (df[tau_cols] > hi)).any(axis=1).sum())
    lo, hi = EXPECTED_RANGES["p_consumer"]
    p_bad = int(((df[p_cons_cols] < lo) | (df[p_cons_cols] > hi)).any(axis=1).sum())
    lo, hi = EXPECTED_RANGES["g"]
    g_bad = int(((df[g_cols] < lo) | (df[g_cols] > hi)).any(axis=1).sum())

    power_sum = df[["p1", "p2", "p3", "p4"]].sum(axis=1)
    balance_bad = int((power_sum.abs() > 1e-9).sum())

    # stabf must be "stable" exactly when stab <= 0 (target-leakage evidence)
    rule_bad = int(((df["stab"] <= 0) != (df["stabf"] == "stable")).sum())
    label_bad = int((~df["stabf"].isin(["stable", "unstable"])).sum())

    return pd.DataFrame({
        "Check": [
            "tau1-tau4 within [0.5, 10]",
            "p2-p4 within [-2, -0.5]",
            "g1-g4 within [0.05, 1]",
            "p1 + p2 + p3 + p4 = 0 (|sum| <= 1e-9)",
            "stabf == 'stable' exactly when stab <= 0",
            "stabf only contains 'stable'/'unstable'",
        ],
        "Violating rows": [tau_bad, p_bad, g_bad, balance_bad, rule_bad, label_bad],
        "Max |p1+p2+p3+p4|": [None, None, None, float(power_sum.abs().max()), None, None],
    })


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "electrical_grid_stability.csv"
    res = audit_dataset(data_path)
    print("Dataset Shape:", res["shape"])
    print("Class Counts:", res["class_counts"])
    print("Class Percentages (%):", res["class_percentages"])
    print("Missing Values Total:", sum(res["missing_values"].values()))
    print("Duplicates:", res["duplicate_rows"])
    print("\nValidity checks:")
    print(validity_checks(pd.read_csv(data_path)).to_string(index=False))
