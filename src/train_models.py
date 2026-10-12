"""
Model definition and training pipelines for Logistic Regression and Random Forest.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_logistic_regression_pipeline(max_iter: int = 2000, random_state: int = 42) -> Pipeline:
    """
    StandardScaler inside a Pipeline guarantees feature normalization is fitted
    strictly on training folds without test-set leakage.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=max_iter, random_state=random_state))
    ])


def get_random_forest_model(n_estimators: int = 200, random_state: int = 42, n_jobs: int = -1) -> RandomForestClassifier:
    """
    Ensemble decision tree model capable of learning non-linear feature interactions.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs
    )
