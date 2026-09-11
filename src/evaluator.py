"""
Comprehensive Evaluation Engine: Metrics, Confusion Matrix, Failure Lead-Time,
and Sensor-Level Explainability.
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
    from src.config import ANALOGUE_SENSORS, FEATURES, KNOWN_FAILURES, DEVICE, SEQUENCE_LENGTH
    from src.dataset import build_sliding_windows
    from src.preprocessing import transform_data, generate_ground_truth_labels
except ImportError:
    from config import ANALOGUE_SENSORS, FEATURES, KNOWN_FAILURES, DEVICE, SEQUENCE_LENGTH
    from dataset import build_sliding_windows
    from preprocessing import transform_data, generate_ground_truth_labels


@torch.no_grad()
def compute_reconstruction_errors(
    model: nn.Module,
    windows: np.ndarray,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes sequence-level MSE and sensor-level MSE.

    Args:
        model: Trained autoencoder model
        windows: (n_windows, seq_len, n_features)

    Returns:
        seq_errors: (n_windows,)
        sensor_errors: (n_windows, n_features)
    """
    model.eval()
    tensor = torch.from_numpy(windows).float().to(DEVICE)
    n_windows = len(windows)

    seq_err_list = []
    sensor_err_list = []

    for i in range(0, n_windows, batch_size):
        batch = tensor[i : i + batch_size]
        recon = model(batch)

        # Squared difference: (B, T, D)
        sq_diff = (batch - recon) ** 2

        # Sequence error: mean over time and sensors -> (B,)
        seq_err = sq_diff.mean(dim=(1, 2)).cpu().numpy()
        # Sensor error: mean over time -> (B, D)
        sensor_err = sq_diff.mean(dim=1).cpu().numpy()

        seq_err_list.append(seq_err)
        sensor_err_list.append(sensor_err)

    seq_errors = np.concatenate(seq_err_list)
    sensor_errors = np.concatenate(sensor_err_list, axis=0)
    return seq_errors, sensor_errors


def evaluate_predictions(
    y_true: np.ndarray,
    seq_errors: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Computes all classification and ranking metrics for anomaly detection."""
    y_pred = (seq_errors > threshold).astype(np.int32)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC-AUC and PR-AUC require at least 1 positive and 1 negative sample
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, seq_errors))
        pr_auc = float(average_precision_score(y_true, seq_errors))
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


def extract_anomaly_events(
    timestamps: np.ndarray,
    seq_errors: np.ndarray,
    sensor_errors: Optional[np.ndarray] = None,
    threshold: float = 1.0,
    sensor_names: List[str] = ANALOGUE_SENSORS,
    max_gap_minutes: float = 30.0,
    failures: List[Dict] = KNOWN_FAILURES,
) -> List[Dict[str, Any]]:
    """Groups consecutive or closely spaced anomalous windows into discrete anomaly episodes.

    For each detected event reports:
      - Event ID
      - Start time
      - End time
      - Peak anomaly score
      - Mean anomaly score
      - Duration (minutes and hours)
      - Dominant sensor
      - Normalized sensor contributions
      - Associated failure lead time (if preceding a documented failure)
    """
    ts_series = pd.to_datetime(timestamps)
    is_anom = seq_errors > threshold
    anom_indices = np.where(is_anom)[0]

    if len(anom_indices) == 0:
        return []

    # Group adjacent anomaly indices separated by less than max_gap_minutes
    events = []
    current_cluster = [anom_indices[0]]

    for i in range(1, len(anom_indices)):
        prev_idx = anom_indices[i - 1]
        curr_idx = anom_indices[i]
        delta_min = (ts_series[curr_idx] - ts_series[prev_idx]).total_seconds() / 60.0

        if delta_min <= max_gap_minutes:
            current_cluster.append(curr_idx)
        else:
            events.append(current_cluster)
            current_cluster = [curr_idx]
    if current_cluster:
        events.append(current_cluster)

    structured_events = []
    for evt_id, cluster in enumerate(events, 1):
        c_ts = ts_series[cluster]
        c_seq_err = seq_errors[cluster]

        start_time = c_ts.min()
        end_time = c_ts.max()
        duration_sec = (end_time - start_time).total_seconds()
        duration_min = max(duration_sec / 60.0, 1.0)
        peak_score = float(np.max(c_seq_err))
        mean_score = float(np.mean(c_seq_err))

        # Sensor attribution for this event
        if sensor_errors is not None:
            c_sens_err = sensor_errors[cluster].mean(axis=0)  # (n_sensors,)
            tot = float(np.sum(c_sens_err))
            sens_contrib = {}
            for s_i, s_name in enumerate(sensor_names):
                pct = float((c_sens_err[s_i] / tot) * 100.0) if tot > 0 else 0.0
                sens_contrib[s_name] = round(pct, 2)
            dom_sensor = max(sens_contrib.items(), key=lambda x: x[1])[0]
        else:
            sens_contrib = {}
            dom_sensor = "Unknown"

        # Check if event falls within pre-failure window of any documented failure
        matched_failure = None
        lead_time_min = None
        for f in failures:
            f_start = pd.Timestamp(f["start"])
            f_end = pd.Timestamp(f["end"])
            # If event start is before or within the failure window and within 3 days prior
            if (start_time <= f_end) and (f_start - start_time).total_seconds() <= (3 * 86400) and (start_time <= f_start):
                matched_failure = f["id"]
                lead_time_min = (f_start - start_time).total_seconds() / 60.0
                break

        structured_events.append({
            "event_id": f"EVT-{evt_id:03d}",
            "start_time": str(start_time),
            "end_time": str(end_time),
            "duration_minutes": round(duration_min, 1),
            "duration_hours": round(duration_min / 60.0, 2),
            "window_count": len(cluster),
            "peak_anomaly_score": round(peak_score, 6),
            "mean_anomaly_score": round(mean_score, 6),
            "dominant_sensor": dom_sensor,
            "sensor_contributions": sens_contrib,
            "associated_failure_id": matched_failure,
            "lead_time_minutes": round(lead_time_min, 1) if lead_time_min is not None else None,
            "lead_time_hours": round(lead_time_min / 60.0, 2) if lead_time_min is not None else None,
        })

    return structured_events


def evaluate_lead_times(
    timestamps: np.ndarray,
    seq_errors: np.ndarray,
    threshold: float,
    failures: List[Dict] = KNOWN_FAILURES,
    lookback_days: float = 2.0,
) -> List[Dict[str, Any]]:
    """Evaluates detection lead time for each documented failure event."""
    ts_series = pd.Series(pd.to_datetime(timestamps))
    err_series = pd.Series(seq_errors)

    results = []
    for f in failures:
        fail_start = pd.Timestamp(f["start"])
        search_start = fail_start - pd.Timedelta(days=lookback_days)

        mask = (ts_series >= search_start) & (ts_series <= fail_start)
        window_ts = ts_series.loc[mask]
        window_err = err_series.loc[mask]

        crossings = window_ts[window_err > threshold]

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
            })

    return results


def compute_sensor_contributions(
    sensor_errors: np.ndarray,
    y_true: np.ndarray,
    sensor_names: List[str] = ANALOGUE_SENSORS,
) -> Dict[str, Any]:
    """Computes sensor-level error contribution and importance ranking."""
    normal_mask = (y_true == 0)
    anomaly_mask = (y_true == 1)

    mean_normal = sensor_errors[normal_mask].mean(axis=0) if np.any(normal_mask) else np.zeros(len(sensor_names))
    mean_anomaly = sensor_errors[anomaly_mask].mean(axis=0) if np.any(anomaly_mask) else np.zeros(len(sensor_names))

    total_anomaly_error = float(np.sum(mean_anomaly))
    contributions = {}
    for i, s in enumerate(sensor_names):
        pct = float((mean_anomaly[i] / total_anomaly_error) * 100.0) if total_anomaly_error > 0 else 0.0
        contributions[s] = {
            "mean_normal_error": float(mean_normal[i]),
            "mean_anomaly_error": float(mean_anomaly[i]),
            "error_ratio": float(mean_anomaly[i] / max(mean_normal[i], 1e-8)),
            "contribution_pct": round(pct, 2),
        }

    # Rank sensors by contribution percentage
    ranked = sorted(contributions.items(), key=lambda x: x[1]["contribution_pct"], reverse=True)
    return {
        "sensor_breakdown": contributions,
        "ranking": [(s, data["contribution_pct"]) for s, data in ranked],
    }
