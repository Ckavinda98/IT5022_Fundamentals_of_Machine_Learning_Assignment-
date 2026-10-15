"""
Automated pipeline execution to train both models, evaluate on held-out test data,
and generate all output figures and metrics tables.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src.preprocessing import load_and_split_data
from src.train_models import get_logistic_regression_pipeline, get_random_forest_model
from src.evaluation import evaluate_predictions


def run_full_pipeline():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "electrical_grid_stability.csv"
    figures_dir = project_root / "outputs" / "figures"
    metrics_dir = project_root / "outputs" / "metrics"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    print("1. Loading and splitting dataset...")
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    
    # Raw dataframe for EDA plots
    df_raw = pd.read_csv(data_path)
    
    print("2. Generating EDA figures...")
    # Plot 1: Target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_raw, x="stabf", hue="stabf", order=["unstable", "stable"],
                  palette={"unstable": "#e74c3c", "stable": "#2ecc71"}, legend=False)
    plt.title("Distribution of Stability Class (stabf)")
    plt.xlabel("Grid Stability Label")
    plt.ylabel("Sample Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png", dpi=300)
    plt.close()
    
    # Plot 2: Boxplot of stab by stabf (Leakage illustration)
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_raw, x="stabf", y="stab", hue="stabf", order=["unstable", "stable"],
                palette={"unstable": "#e74c3c", "stable": "#2ecc71"}, legend=False)
    plt.axhline(0, color="black", linestyle="--", alpha=0.7, label="Threshold (stab=0)")
    plt.title("Continuous Stability Score (stab) vs Target Class (stabf)")
    plt.xlabel("Grid Stability Label")
    plt.ylabel("Eigenvalue Stability Indicator (stab)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "stab_vs_stabf_leakage.png", dpi=300)
    plt.close()
    
    # Plot 3: Feature Correlation Heatmap (12 operating features)
    feature_cols = [c for c in df_raw.columns if c not in ["stab", "stabf"]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_raw[feature_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap of 12 Operating Parameters")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_correlation_heatmap.png", dpi=300)
    plt.close()

    # Plot 4: Distribution of each operating feature by stability class
    long_df = df_raw.melt(id_vars="stabf", value_vars=feature_cols, var_name="feature")
    grid = sns.FacetGrid(long_df, col="feature", col_wrap=4, hue="stabf", sharex=False, sharey=False,
                         palette={"unstable": "#e74c3c", "stable": "#2ecc71"}, height=2.2, aspect=1.4)
    grid.map(sns.kdeplot, "value", fill=True, alpha=0.35)
    grid.add_legend(title="stabf")
    grid.figure.suptitle("Operating Feature Distributions by Stability Class", y=1.02)
    grid.savefig(figures_dir / "feature_distributions_by_class.png", dpi=300, bbox_inches="tight")
    plt.close(grid.figure)

    print("3. Training Logistic Regression pipeline...")
    lr_pipe = get_logistic_regression_pipeline()
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    
    print("4. Training Random Forest classifier...")
    rf_model = get_random_forest_model()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    print("5. Evaluating predictions...")
    res_lr = evaluate_predictions(y_test, y_pred_lr)
    res_rf = evaluate_predictions(y_test, y_pred_rf)
    
    # Zero-rule baseline: always predict the majority class seen in training
    majority_class = y_train.mode()[0]
    baseline_acc = (y_test == majority_class).mean()

    # Comparison Table
    comparison_df = pd.DataFrame({
        "Metric": [
            "Baseline Zero-Rule",
            "Accuracy",
            "Macro F1-Score",
            "Weighted F1-Score",
            "Stable Precision",
            "Stable Recall",
            "Stable F1-Score",
            "Unstable Precision",
            "Unstable Recall",
            "Unstable F1-Score"
        ],
        "Logistic Regression": [
            round(baseline_acc, 4),
            round(res_lr["accuracy"], 4),
            round(res_lr["macro_f1"], 4),
            round(res_lr["weighted_f1"], 4),
            round(res_lr["stable_precision"], 4),
            round(res_lr["stable_recall"], 4),
            round(res_lr["stable_f1"], 4),
            round(res_lr["unstable_precision"], 4),
            round(res_lr["unstable_recall"], 4),
            round(res_lr["unstable_f1"], 4),
        ],
        "Random Forest": [
            round(baseline_acc, 4),
            round(res_rf["accuracy"], 4),
            round(res_rf["macro_f1"], 4),
            round(res_rf["weighted_f1"], 4),
            round(res_rf["stable_precision"], 4),
            round(res_rf["stable_recall"], 4),
            round(res_rf["stable_f1"], 4),
            round(res_rf["unstable_precision"], 4),
            round(res_rf["unstable_recall"], 4),
            round(res_rf["unstable_f1"], 4),
        ]
    })
    comparison_df.to_csv(metrics_dir / "model_comparison.csv", index=False)
    print("\n=== Model Comparison Table ===")
    print(comparison_df.to_string(index=False))

    # Confusion Matrices Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(res_lr["confusion_matrix"], display_labels=["stable", "unstable"]).plot(
        ax=axes[0], cmap="Blues", values_format="d"
    )
    axes[0].set_title(f"Logistic Regression\nAccuracy: {res_lr['accuracy']:.4f}")
    
    ConfusionMatrixDisplay(res_rf["confusion_matrix"], display_labels=["stable", "unstable"]).plot(
        ax=axes[1], cmap="Greens", values_format="d"
    )
    axes[1].set_title(f"Random Forest\nAccuracy: {res_rf['accuracy']:.4f}")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrices.png", dpi=300)
    plt.close()

    # Feature Importance Figure (Random Forest)
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="viridis", legend=False)
    plt.title("Random Forest Feature Importances (MDI)")
    plt.xlabel("Mean Decrease in Impurity")
    plt.ylabel("Grid Parameter")
    plt.tight_layout()
    plt.savefig(figures_dir / "rf_feature_importance.png", dpi=300)
    plt.close()

    print("\nPipeline execution complete! Figures and metrics saved successfully.")


if __name__ == "__main__":
    run_full_pipeline()
