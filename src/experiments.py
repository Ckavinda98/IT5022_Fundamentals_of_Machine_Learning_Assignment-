"""
Validation and accuracy-improvement experiments.

Every model-selection decision here is made with stratified 5-fold cross-validation
on the TRAINING set only. The held-out test set is used once, at the end, to report
the final scores of the baseline and improved models.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.preprocessing import FEATURE_COLUMNS, load_and_split_data
from src.train_models import get_logistic_regression_pipeline, get_random_forest_model

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"accuracy": "accuracy", "macro_f1": "f1_macro", "roc_auc": "roc_auc"}

FEATURE_GROUPS = {
    "All 12 features": FEATURE_COLUMNS,
    "Without p1 (11 features)": [c for c in FEATURE_COLUMNS if c != "p1"],
    "Without p1-p4 (tau + g)": [c for c in FEATURE_COLUMNS if not c.startswith("p")],
    "tau only": ["tau1", "tau2", "tau3", "tau4"],
    "g only": ["g1", "g2", "g3", "g4"],
    "p only": ["p1", "p2", "p3", "p4"],
}


def get_polynomial_logistic_pipeline(degree: int = 2, C: float = 1.0) -> Pipeline:
    """
    Logistic Regression on degree-2 polynomial features (squares and pairwise
    interactions such as tau1*g1), giving the linear model a non-linear boundary.
    """
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=C, max_iter=5000, random_state=RANDOM_STATE)),
    ])


def cv_summary(model, X, y) -> dict:
    """Mean and standard deviation of 5-fold CV scores plus mean train-fold accuracy."""
    res = cross_validate(model, X, y, cv=CV, scoring=SCORING, return_train_score=True, n_jobs=-1)
    out = {}
    for name in SCORING:
        out[f"cv_{name}_mean"] = res[f"test_{name}"].mean()
        out[f"cv_{name}_std"] = res[f"test_{name}"].std()
    out["cv_train_accuracy_mean"] = res["train_accuracy"].mean()
    return out


def test_scores(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Test-set metrics. `threshold`: predict "unstable" when P(unstable) > threshold (0.5 matches model.predict)."""
    p_unstable = model.predict_proba(X_test)[:, list(model.classes_).index("unstable")]
    y_pred = np.where(p_unstable > threshold, "unstable", "stable")
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_test == "unstable", p_unstable),
        "unstable_precision": precision_score(y_test, y_pred, pos_label="unstable"),
        "unstable_recall": recall_score(y_test, y_pred, pos_label="unstable"),
        "stable_recall": recall_score(y_test, y_pred, pos_label="stable"),
        "false_stable": int(((y_test == "unstable") & (y_pred == "stable")).sum()),
        "false_unstable": int(((y_test == "stable") & (y_pred == "unstable")).sum()),
    }


def run_experiments():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "electrical_grid_stability.csv"
    figures_dir = project_root / "outputs" / "figures"
    metrics_dir = project_root / "outputs" / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    baseline_models = {
        "Logistic Regression": get_logistic_regression_pipeline(),
        "Random Forest": get_random_forest_model(),
    }

    # ---------------------------------------------------------------
    # 1. Cross-validation and overfitting check for the baseline models
    # ---------------------------------------------------------------
    print("1. Cross-validating baseline models...")
    rows = []
    for name, model in baseline_models.items():
        row = {"Model": name, **cv_summary(model, X_train, y_train)}
        fitted = clone(model).fit(X_train, y_train)
        row["full_train_accuracy"] = accuracy_score(y_train, fitted.predict(X_train))
        row["test_accuracy"] = accuracy_score(y_test, fitted.predict(X_test))
        rows.append(row)
    cv_df = pd.DataFrame(rows).round(4)
    cv_df.to_csv(metrics_dir / "cv_and_overfitting.csv", index=False)
    print(cv_df.to_string(index=False))

    # ---------------------------------------------------------------
    # 2. Feature-group ablation (redundant p1, feature families)
    # ---------------------------------------------------------------
    print("\n2. Feature-group ablation (CV on training set)...")
    rows = []
    for group, cols in FEATURE_GROUPS.items():
        for name, model in baseline_models.items():
            s = cv_summary(model, X_train[cols], y_train)
            rows.append({
                "Feature set": group, "Model": name, "n_features": len(cols),
                "cv_accuracy": s["cv_accuracy_mean"], "cv_macro_f1": s["cv_macro_f1_mean"],
                "cv_macro_f1_std": s["cv_macro_f1_std"],
            })
    ablation_df = pd.DataFrame(rows).round(4)
    ablation_df.to_csv(metrics_dir / "feature_ablation.csv", index=False)
    print(ablation_df.to_string(index=False))

    # ---------------------------------------------------------------
    # 3. Dimensionality reduction: PCA on standardised training features
    # ---------------------------------------------------------------
    print("\n3. PCA explained variance...")
    pca = PCA().fit(StandardScaler().fit_transform(X_train))
    pca_df = pd.DataFrame({
        "component": np.arange(1, len(FEATURE_COLUMNS) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
    }).round(6)
    pca_df.to_csv(metrics_dir / "pca_explained_variance.csv", index=False)
    print(pca_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(pca_df["component"], pca_df["explained_variance_ratio"], color="#4c72b0", label="Per component")
    ax.plot(pca_df["component"], pca_df["cumulative"], "o-", color="#dd8452", label="Cumulative")
    ax.set_xticks(pca_df["component"])
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA on 12 Standardised Operating Features (training set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "pca_explained_variance.png", dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 4. Hyperparameter tuning (GridSearchCV on training set only)
    # ---------------------------------------------------------------
    print("\n4. Tuning Logistic Regression (with/without polynomial features)...")
    lr_grid = GridSearchCV(
        Pipeline([
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        param_grid={"poly__degree": [1, 2], "classifier__C": [0.01, 0.1, 1, 10, 100]},
        scoring="f1_macro", cv=CV, n_jobs=-1,
    ).fit(X_train, y_train)

    print("   Tuning Random Forest...")
    rf_grid = GridSearchCV(
        get_random_forest_model(n_estimators=300),
        param_grid={
            "max_depth": [None, 12, 20],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", 0.5],
        },
        scoring="f1_macro", cv=CV, n_jobs=-1,
    ).fit(X_train, y_train)

    tuning_rows = []
    for name, grid in [("Logistic Regression", lr_grid), ("Random Forest", rf_grid)]:
        res = pd.DataFrame(grid.cv_results_)
        for _, r in res.iterrows():
            tuning_rows.append({
                "Model": name,
                "params": str(r["params"]),
                "cv_macro_f1_mean": r["mean_test_score"],
                "cv_macro_f1_std": r["std_test_score"],
                "rank": int(r["rank_test_score"]),
            })
        print(f"   Best {name}: {grid.best_params_}  CV macro-F1 = {grid.best_score_:.4f}")
    tuning_df = pd.DataFrame(tuning_rows).sort_values(["Model", "rank"]).round(4)
    tuning_df.to_csv(metrics_dir / "hyperparameter_tuning.csv", index=False)

    tuned_models = {
        "Logistic Regression (tuned, polynomial)": lr_grid.best_estimator_,
        "Random Forest (tuned)": rf_grid.best_estimator_,
    }

    # ---------------------------------------------------------------
    # 5. Decision-threshold analysis for missed instabilities
    #    Threshold chosen on out-of-fold training probabilities, never on test.
    # ---------------------------------------------------------------
    print("\n5. Decision-threshold analysis (out-of-fold training probabilities)...")
    thresholds = np.round(np.arange(0.20, 0.81, 0.05), 2)
    rows = []
    chosen_thresholds = {}
    for name, model in tuned_models.items():
        oof = cross_val_predict(clone(model), X_train, y_train, cv=CV, method="predict_proba", n_jobs=-1)
        p_unstable = oof[:, list(model.classes_).index("unstable")]
        for t in thresholds:
            y_pred = np.where(p_unstable > t, "unstable", "stable")
            rows.append({
                "Model": name, "threshold": t,
                "unstable_recall": recall_score(y_train, y_pred, pos_label="unstable"),
                "unstable_precision": precision_score(y_train, y_pred, pos_label="unstable"),
                "macro_f1": f1_score(y_train, y_pred, average="macro"),
                "accuracy": accuracy_score(y_train, y_pred),
            })
        # Highest threshold that still reaches >= 98% unstable recall on the training folds
        model_rows = [r for r in rows if r["Model"] == name and r["unstable_recall"] >= 0.98]
        chosen_thresholds[name] = max(r["threshold"] for r in model_rows) if model_rows else 0.5
        print(f"   {name}: threshold for >=98% unstable recall = {chosen_thresholds[name]}")
    threshold_df = pd.DataFrame(rows).round(4)
    threshold_df.to_csv(metrics_dir / "threshold_analysis.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, name in zip(axes, tuned_models):
        d = threshold_df[threshold_df["Model"] == name]
        ax.plot(d["threshold"], d["unstable_recall"], "o-", label="Unstable recall")
        ax.plot(d["threshold"], d["unstable_precision"], "s-", label="Unstable precision")
        ax.plot(d["threshold"], d["macro_f1"], "^-", label="Macro F1")
        ax.axvline(0.5, color="grey", ls="--", alpha=0.6)
        ax.axvline(chosen_thresholds[name], color="red", ls=":", label="Chosen threshold")
        ax.set_title(name)
        ax.set_xlabel("Predict unstable when P(unstable) > threshold")
    axes[0].set_ylabel("Score (out-of-fold, training set)")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(figures_dir / "threshold_tradeoff.png", dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 6. Final held-out test evaluation (used once)
    # ---------------------------------------------------------------
    print("\n6. Final held-out test evaluation...")
    final_models = {name: clone(m).fit(X_train, y_train) for name, m in baseline_models.items()}
    final_models.update(tuned_models)  # GridSearchCV already refit these on the full training set

    rows = [{
        "Model": "Majority baseline (always unstable)", "threshold": "-",
        "accuracy": (y_test == "unstable").mean(), "macro_f1": f1_score(
            y_test, np.full(len(y_test), "unstable"), average="macro"),
        "roc_auc": 0.5, "unstable_precision": (y_test == "unstable").mean(),
        "unstable_recall": 1.0, "stable_recall": 0.0,
        "false_stable": 0, "false_unstable": int((y_test == "stable").sum()),
    }]
    for name, model in final_models.items():
        rows.append({"Model": name, "threshold": 0.5, **test_scores(model, X_test, y_test)})
    for name, model in tuned_models.items():
        t = chosen_thresholds[name]
        rows.append({"Model": f"{name}, recall-oriented threshold", "threshold": t,
                     **test_scores(model, X_test, y_test, threshold=t)})
    final_df = pd.DataFrame(rows)
    num_cols = final_df.select_dtypes("number").columns
    final_df[num_cols] = final_df[num_cols].round(4)
    final_df.to_csv(metrics_dir / "final_model_comparison.csv", index=False)
    print(final_df.to_string(index=False))

    # ROC curves on the test set
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, model in final_models.items():
        p = model.predict_proba(X_test)[:, list(model.classes_).index("unstable")]
        fpr, tpr, _ = roc_curve(y_test == "unstable", p)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test == 'unstable', p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax.set_xlabel("False positive rate (stable predicted unstable)")
    ax.set_ylabel("True positive rate (unstable recall)")
    ax.set_title("ROC Curves on Held-Out Test Set (positive = unstable)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png", dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 7. Permutation importance (less biased than impurity-based MDI)
    # ---------------------------------------------------------------
    print("\n7. Permutation importance on test set...")
    rows = []
    for name in ["Logistic Regression", "Random Forest"]:
        pi = permutation_importance(final_models[name], X_test, y_test, scoring="f1_macro",
                                    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        for f, m, s in zip(FEATURE_COLUMNS, pi.importances_mean, pi.importances_std):
            rows.append({"Model": name, "feature": f, "importance_mean": m, "importance_std": s})
    perm_df = pd.DataFrame(rows).round(4)
    perm_df.to_csv(metrics_dir / "permutation_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    order = (perm_df[perm_df["Model"] == "Random Forest"]
             .sort_values("importance_mean", ascending=False)["feature"])
    sns.barplot(data=perm_df, x="importance_mean", y="feature", hue="Model", order=order, ax=ax)
    ax.set_xlabel("Mean drop in macro-F1 when feature is shuffled")
    ax.set_ylabel("Grid parameter")
    ax.set_title("Permutation Feature Importance (held-out test set)")
    fig.tight_layout()
    fig.savefig(figures_dir / "permutation_importance.png", dpi=300)
    plt.close(fig)

    print("\nExperiments complete. Results saved to outputs/metrics and outputs/figures.")
    return {
        "cv": cv_df, "ablation": ablation_df, "pca": pca_df, "tuning": tuning_df,
        "thresholds": threshold_df, "final": final_df, "permutation": perm_df,
        "best_params": {"Logistic Regression": lr_grid.best_params_, "Random Forest": rf_grid.best_params_},
        "chosen_thresholds": chosen_thresholds,
    }


if __name__ == "__main__":
    run_experiments()
