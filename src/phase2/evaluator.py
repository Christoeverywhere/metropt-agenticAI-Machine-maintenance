"""
Phase 2 Evaluator: Probability Threshold Calibration,
Classification Metrics (PR-AUC, ROC-AUC, F1), Confusion Matrices, and Predictive Warning Lead Times.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

try:
    from src.config import KNOWN_FAILURES, DEVICE
except ImportError:
    from config import KNOWN_FAILURES, DEVICE


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    windows: np.ndarray,
    batch_size: int = 512,
    device: torch.device = DEVICE,
) -> np.ndarray:
    """Runs forward pass and computes sigmoid failure probabilities (n_samples,)."""
    model.eval()
    tensor = torch.from_numpy(windows).float().to(device)
    n_windows = len(windows)
    probs_list = []

    for i in range(0, n_windows, batch_size):
        batch = tensor[i : i + batch_size]
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        probs_list.append(probs)

    return np.concatenate(probs_list) if probs_list else np.empty(0, dtype=np.float32)


def calibrate_probability_threshold(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Sweeps threshold grid on validation data to select optimal decision boundary."""
    if grid is None:
        grid = np.linspace(0.10, 0.90, 81)

    best_thresh = 0.50
    best_f1 = -1.0
    results_grid = []

    for thresh in grid:
        y_pred = (val_probs >= thresh).astype(np.int32)
        prec = float(precision_score(y_val, y_pred, zero_division=0))
        rec = float(recall_score(y_val, y_pred, zero_division=0))
        f1 = float(f1_score(y_val, y_pred, zero_division=0))

        results_grid.append({
            "threshold": float(thresh),
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    return {
        "calibrated_threshold": best_thresh,
        "best_val_f1": best_f1,
        "grid_results": results_grid,
    }


def evaluate_classifier_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Computes all classification and ranking metrics for impending failure prediction."""
    y_pred = (probs >= threshold).astype(np.int32)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, probs))
        pr_auc = float(average_precision_score(y_true, probs))
    else:
        roc_auc, pr_auc = 0.0, 0.0

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
    }


def evaluate_failure_lead_times(
    timestamps: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    failures: List[Dict] = KNOWN_FAILURES,
    max_lookback_days: float = 4.0,
) -> List[Dict[str, Any]]:
    """Calculates predictive warning lead time for each documented failure event.

    Predictive Warning Lead Time = Failure Start Time - First Qualifying Prediction Time.
    """
    ts_series = pd.Series(pd.to_datetime(timestamps))
    prob_series = pd.Series(probs)

    results = []
    for f in failures:
        fail_start = pd.Timestamp(f["start"])
        search_start = fail_start - pd.Timedelta(days=max_lookback_days)

        mask = (ts_series >= search_start) & (ts_series <= fail_start)
        window_ts = ts_series.loc[mask]
        window_prob = prob_series.loc[mask]

        crossings = window_ts[window_prob >= threshold]

        if len(crossings) > 0:
            first_flag = crossings.iloc[0]
            lead_delta = fail_start - first_flag
            lead_minutes = lead_delta.total_seconds() / 60.0
            results.append({
                "failure_id": f["id"],
                "failure_type": f["type"],
                "failure_start": str(fail_start),
                "detected": True,
                "first_flag": str(first_flag),
                "lead_time_minutes": round(lead_minutes, 1),
                "lead_time_hours": round(lead_minutes / 60.0, 2),
                "peak_probability": round(float(window_prob.max()), 4),
            })
        else:
            results.append({
                "failure_id": f["id"],
                "failure_type": f["type"],
                "failure_start": str(fail_start),
                "detected": False,
                "first_flag": None,
                "lead_time_minutes": None,
                "lead_time_hours": None,
                "peak_probability": round(float(window_prob.max()), 4) if len(window_prob) > 0 else 0.0,
            })

    return results
