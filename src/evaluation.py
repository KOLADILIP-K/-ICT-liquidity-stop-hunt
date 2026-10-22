"""
Model Evaluation and Metrics Module
Calculates classification metrics, PR-AUC, ROC-AUC, Brier score, calibration curve,
and cross-instrument performance matrices.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of binary classification metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # Basic stats
    n_total = len(y_true)
    n_pos = int(np.sum(y_true))
    n_neg = n_total - n_pos

    if n_total == 0:
        return {"error": "Empty dataset"}

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))

    # ROC-AUC and PR-AUC require at least one positive and one negative sample
    roc_auc = float(roc_auc_score(y_true, y_prob)) if n_pos > 0 and n_neg > 0 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if n_pos > 0 else 0.0

    cm = confusion_matrix(y_true, y_pred).tolist() if n_total > 0 else [[0, 0], [0, 0]]

    # Calibration curve (5 bins)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform")

    return {
        "n_samples": n_total,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "positive_rate": round(n_pos / max(n_total, 1), 4),
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "calibration": {
            "prob_true": [round(float(p), 4) for p in prob_true],
            "prob_pred": [round(float(p), 4) for p in prob_pred]
        }
    }


def compare_instrument_metrics(
    eval_df: pd.DataFrame,
    threshold: float = 0.50
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates model performance across instruments (Combined, Gold only, Silver only).
    """
    results = {}
    
    # Combined
    results["Combined"] = evaluate_binary_predictions(
        eval_df["y_true"].values,
        eval_df["y_prob"].values,
        threshold=threshold
    )

    # Gold (XAUUSD)
    gold_mask = eval_df["instrument"].str.contains("XAU", case=False, na=False)
    if gold_mask.sum() > 0:
        results["XAUUSD"] = evaluate_binary_predictions(
            eval_df.loc[gold_mask, "y_true"].values,
            eval_df.loc[gold_mask, "y_prob"].values,
            threshold=threshold
        )

    # Silver (XAGUSD)
    silver_mask = eval_df["instrument"].str.contains("XAG", case=False, na=False)
    if silver_mask.sum() > 0:
        results["XAGUSD"] = evaluate_binary_predictions(
            eval_df.loc[silver_mask, "y_true"].values,
            eval_df.loc[silver_mask, "y_prob"].values,
            threshold=threshold
        )

    return results
