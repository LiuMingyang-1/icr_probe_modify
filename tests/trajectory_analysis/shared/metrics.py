"""Shared evaluation metrics for trajectory analysis."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 42


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute binary classification metrics.

    Returns dict with AUROC, AUPRC, F1, Accuracy.
    """
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    # Find optimal threshold for F1
    thresholds = np.linspace(0, 1, 200)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = t

    preds = (y_prob >= best_thresh).astype(int)
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "F1": best_f1,
        "Accuracy": accuracy_score(y_true, preds),
        "Threshold": best_thresh,
    }


def cross_validate(X: np.ndarray, y: np.ndarray, model_class, n_splits: int = 5, **model_kwargs) -> dict:
    """Stratified K-fold cross-validation.

    model_class must support fit(X, y) and predict_proba(X) or decision_function(X).

    Returns dict with mean and std for each metric.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    all_metrics = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            y_prob = model.decision_function(X_val)

        metrics = evaluate(y_val, y_prob)
        all_metrics.append(metrics)

    result = {}
    for key in all_metrics[0]:
        if key == "Threshold":
            continue
        vals = [m[key] for m in all_metrics]
        result[f"{key}_mean"] = np.mean(vals)
        result[f"{key}_std"] = np.std(vals)

    return result


def print_results(results: dict, method_name: str):
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f"  {method_name}")
    print(f"{'='*50}")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key:20s}: {val:.4f}")
        else:
            print(f"  {key:20s}: {val}")
    print()
