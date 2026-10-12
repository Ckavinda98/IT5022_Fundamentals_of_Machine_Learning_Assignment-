"""
Evaluation and metrics computation module.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_predictions(y_true, y_pred, labels=("stable", "unstable")) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    
    prec_stable = precision_score(y_true, y_pred, pos_label="stable")
    rec_stable = recall_score(y_true, y_pred, pos_label="stable")
    f1_stable = f1_score(y_true, y_pred, pos_label="stable")
    
    prec_unstable = precision_score(y_true, y_pred, pos_label="unstable")
    rec_unstable = recall_score(y_true, y_pred, pos_label="unstable")
    f1_unstable = f1_score(y_true, y_pred, pos_label="unstable")
    
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "stable_precision": prec_stable,
        "stable_recall": rec_stable,
        "stable_f1": f1_stable,
        "unstable_precision": prec_unstable,
        "unstable_recall": rec_unstable,
        "unstable_f1": f1_unstable,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_true, y_pred, digits=4)
    }
