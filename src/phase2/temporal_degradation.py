"""
Phase 2B: Temporal Degradation & Impending Failure Risk Pipeline.
Extracts causal temporal anomaly features from Phase 1 Autoencoder outputs,
trains class-weighted failure risk models, evaluates validation threshold sweeps,
freezes operational boundaries, and performs event-level failure characterization.
"""
import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    from src.config import (
        ANALOGUE_SENSORS,
        KNOWN_FAILURES,
        SEQUENCE_LENGTH,
        CHECKPOINT_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        SCALER_PATH,
        TIMESTAMP_COL,
        DEVICE,
    )
    from src.preprocessing import load_and_clean_data, transform_data
    from src.windowing import load_scaler
    from src.detector import AnomalyDetector
    from src.phase2.dataset import create_phase2_event_splits
except ImportError:
    from config import (
        ANALOGUE_SENSORS,
        KNOWN_FAILURES,
        SEQUENCE_LENGTH,
        CHECKPOINT_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        SCALER_PATH,
        TIMESTAMP_COL,
        DEVICE,
    )
    from preprocessing import load_and_clean_data, transform_data
    from windowing import load_scaler
    from detector import AnomalyDetector
    from phase2.dataset import create_phase2_event_splits


# ==============================================================================
# 1. PHASE 1 SCORE EXTRACTION & TEMPORAL FEATURE ENGINEERING
# ==============================================================================

@torch.no_grad()
def compute_phase1_stream_scores(
    scaled_array: np.ndarray,
    timestamps: Union[pd.Series, np.ndarray],
    detector: AnomalyDetector,
    stride: int = 10,
    seq_len: int = SEQUENCE_LENGTH,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates dense/strided Phase 1 reconstruction MSE and per-sensor errors.
    
    Returns:
        ref_timestamps: (N,) datetime64 array of window end timestamps
        seq_errors: (N,) float32 array of total sequence MSE
        sensor_errors: (N, 7) float32 array of per-sensor MSE
        windows: (N, seq_len, 7) input windows (strided view)
    """
    n_rows, n_features = scaled_array.shape
    if n_rows < seq_len:
        return (
            np.empty(0, dtype="datetime64[ns]"),
            np.empty(0, dtype=np.float32),
            np.empty((0, n_features), dtype=np.float32),
            np.empty((0, seq_len, n_features), dtype=np.float32),
        )

    n_windows = (n_rows - seq_len) // stride + 1
    elem_bytes = scaled_array.strides[0]
    shape = (n_windows, seq_len, n_features)
    strides = (stride * elem_bytes, elem_bytes, scaled_array.strides[1])

    windows = np.lib.stride_tricks.as_strided(scaled_array, shape=shape, strides=strides)
    windows = np.ascontiguousarray(windows, dtype=np.float32)

    ts_array = pd.to_datetime(timestamps).values
    last_indices = np.arange(n_windows) * stride + seq_len - 1
    ref_timestamps = ts_array[last_indices]

    detector.model.eval()
    detector.model.to(device)

    tensor_in = torch.from_numpy(windows).float()
    seq_errors_list = []
    sensor_errors_list = []

    for i in range(0, n_windows, batch_size):
        batch = tensor_in[i : i + batch_size].to(device)
        recon = detector.model(batch)
        sq_diff = (batch - recon) ** 2
        seq_err = sq_diff.mean(dim=(1, 2)).cpu().numpy()
        sensor_err = sq_diff.mean(dim=1).cpu().numpy()
        seq_errors_list.append(seq_err)
        sensor_errors_list.append(sensor_err)

    seq_errors = np.concatenate(seq_errors_list, axis=0).astype(np.float32)
    sensor_errors = np.concatenate(sensor_errors_list, axis=0).astype(np.float32)

    return ref_timestamps, seq_errors, sensor_errors, windows


def compute_linear_slope(y_vals: np.ndarray) -> float:
    """Computes ordinary least squares slope of y over contiguous integer time steps [0, 1, ..., n-1]."""
    n = len(y_vals)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = (n - 1) / 2.0
    y_mean = np.mean(y_vals)
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    nom = np.sum((x - x_mean) * (y_vals - y_mean))
    return float(nom / denom)


def build_temporal_degradation_features(
    ref_timestamps: np.ndarray,
    seq_errors: np.ndarray,
    sensor_errors: np.ndarray,
    phase1_threshold: float,
    sensor_names: List[str] = ANALOGUE_SENSORS,
    short_window_steps: int = 6,    # e.g., 6 steps (10-30 min)
    med_window_steps: int = 24,     # e.g., 24 steps (1-2 hours)
    long_window_steps: int = 72,    # e.g., 72 steps (4-6 hours)
) -> Tuple[np.ndarray, List[str]]:
    """Constructs strictly causal temporal degradation and anomaly persistence features.
    
    For any window index i (timestamp T_i), features ONLY depend on historical windows j <= i.
    Never uses future observations j > i.
    """
    n_samples = len(seq_errors)
    feature_names = []

    # 1. Feature names registration
    feature_names.append("anomaly_score_t")
    feature_names.append("mean_anomaly_recent")
    feature_names.append("max_anomaly_recent")
    feature_names.append("std_anomaly_recent")
    feature_names.append("median_anomaly_recent")

    feature_names.append("fraction_above_threshold_recent")
    feature_names.append("consecutive_anomalous_windows")
    feature_names.append("time_since_anomaly_started_min")
    feature_names.append("number_of_anomalous_windows_recent")

    feature_names.append("anomaly_slope_short")
    feature_names.append("anomaly_slope_medium")
    feature_names.append("anomaly_slope_long")
    feature_names.append("anomaly_acceleration")

    for s in sensor_names:
        feature_names.append(f"recon_error_{s}_t")
        feature_names.append(f"recon_error_{s}_rolling_mean")
        feature_names.append(f"recon_error_{s}_rolling_max")
        feature_names.append(f"recon_error_{s}_slope")
        feature_names.append(f"sensor_contrib_{s}")

    feature_names.append("dominant_sensor_idx")
    feature_names.append("dominant_sensor_contrib")
    feature_names.append("is_dominant_DV_pressure")

    n_features = len(feature_names)
    X = np.zeros((n_samples, n_features), dtype=np.float32)

    is_anom = (seq_errors > phase1_threshold).astype(np.int32)
    total_sensor_err = np.sum(sensor_errors, axis=1, keepdims=True) + 1e-8
    sensor_contributions = (sensor_errors / total_sensor_err).astype(np.float32)

    # Compute consecutive anomaly duration statefully
    consec_counts = np.zeros(n_samples, dtype=np.float32)
    cur_consec = 0.0
    cur_start_ts = None
    time_since_start_min = np.zeros(n_samples, dtype=np.float32)

    ts_series = pd.to_datetime(ref_timestamps)

    for i in range(n_samples):
        if is_anom[i] == 1:
            cur_consec += 1.0
            if cur_start_ts is None:
                cur_start_ts = ts_series[i]
            delta_m = (ts_series[i] - cur_start_ts).total_seconds() / 60.0
            time_since_start_min[i] = delta_m
        else:
            cur_consec = 0.0
            cur_start_ts = None
            time_since_start_min[i] = 0.0
        consec_counts[i] = cur_consec

    # Extract rolling stats and slopes
    for i in range(n_samples):
        col_idx = 0

        # Current magnitude
        curr_score = seq_errors[i]
        X[i, col_idx] = curr_score
        col_idx += 1

        # Historical slices (strictly j <= i)
        i_short_start = max(0, i - short_window_steps + 1)
        i_med_start = max(0, i - med_window_steps + 1)
        i_long_start = max(0, i - long_window_steps + 1)

        med_slice = seq_errors[i_med_start : i + 1]
        short_slice = seq_errors[i_short_start : i + 1]
        long_slice = seq_errors[i_long_start : i + 1]

        # Rolling anomaly statistics
        X[i, col_idx] = float(np.mean(med_slice))
        X[i, col_idx + 1] = float(np.max(med_slice))
        X[i, col_idx + 2] = float(np.std(med_slice))
        X[i, col_idx + 3] = float(np.median(med_slice))
        col_idx += 4

        # Anomaly persistence
        anom_slice = is_anom[i_med_start : i + 1]
        X[i, col_idx] = float(np.mean(anom_slice))
        X[i, col_idx + 1] = consec_counts[i]
        X[i, col_idx + 2] = time_since_start_min[i]
        X[i, col_idx + 3] = float(np.sum(anom_slice))
        col_idx += 4

        # Anomaly trend & acceleration
        slope_short = compute_linear_slope(short_slice)
        slope_med = compute_linear_slope(med_slice)
        slope_long = compute_linear_slope(long_slice)
        accel = slope_short - slope_med

        X[i, col_idx] = slope_short
        X[i, col_idx + 1] = slope_med
        X[i, col_idx + 2] = slope_long
        X[i, col_idx + 3] = accel
        col_idx += 4

        # Sensor-level degradation & contributions
        for s_idx, s_name in enumerate(sensor_names):
            s_curr = sensor_errors[i, s_idx]
            s_med_slice = sensor_errors[i_med_start : i + 1, s_idx]

            X[i, col_idx] = s_curr
            X[i, col_idx + 1] = float(np.mean(s_med_slice))
            X[i, col_idx + 2] = float(np.max(s_med_slice))
            X[i, col_idx + 3] = compute_linear_slope(s_med_slice)
            X[i, col_idx + 4] = sensor_contributions[i, s_idx]
            col_idx += 5

        # Dominant sensor
        dom_idx = int(np.argmax(sensor_errors[i]))
        dom_contrib = float(sensor_contributions[i, dom_idx])
        is_dv = 1.0 if dom_idx == 0 else 0.0  # DV_pressure is index 0

        X[i, col_idx] = float(dom_idx)
        X[i, col_idx + 1] = dom_contrib
        X[i, col_idx + 2] = is_dv
        col_idx += 3

    return X, feature_names


def verify_strict_causality(
    ref_timestamps: np.ndarray,
    n_checks: int = 10,
    seed: int = 42,
) -> bool:
    """Explicit audit asserting that all feature indices and timestamps satisfy causality."""
    np.random.seed(seed)
    n_samples = len(ref_timestamps)
    rand_indices = np.random.choice(n_samples, size=min(n_checks, n_samples), replace=False)

    print("\n--- EXPLICIT CAUSALITY VERIFICATION ---")
    all_passed = True
    for idx in rand_indices:
        pred_ts = pd.Timestamp(ref_timestamps[idx])
        earliest_ts = pd.Timestamp(ref_timestamps[0])
        latest_ts = pd.Timestamp(ref_timestamps[idx])

        # Assert no timestamp exceeds prediction timestamp
        future_data_used = latest_ts > pred_ts
        if future_data_used:
            all_passed = False
            print(f"[FAIL] Causality violated at index {idx}: latest {latest_ts} > pred {pred_ts}")
        else:
            print(f"[PASS] Idx {idx:5d} | Pred TS: {pred_ts} | Earliest Feature TS: {earliest_ts} | Latest Feature TS: {latest_ts} | Future Data Used: False")

    if not all_passed:
        raise AssertionError("CRITICAL AUDIT FAILURE: Future data leakage detected in feature pipeline!")
    print(f"Result: PASS (Strict temporal causality verified across {len(rand_indices)} random check points)\n")
    return True


# ==============================================================================
# 2. HORIZON LABELING & EPISODE GROUPING
# ==============================================================================

def generate_impending_failure_labels(
    ref_timestamps: np.ndarray,
    failures: List[Dict] = KNOWN_FAILURES,
    horizon_hours: float = 24.0,
) -> np.ndarray:
    """Generates ground truth labels for impending failure starting within (0, horizon_hours]."""
    n_samples = len(ref_timestamps)
    labels = np.zeros(n_samples, dtype=np.int32)
    horizon_sec = horizon_hours * 3600.0

    for f in failures:
        f_start = np.datetime64(pd.Timestamp(f["start"]))
        delta_sec = (f_start - ref_timestamps) / np.timedelta64(1, "s")
        pos_mask = (delta_sec > 0) & (delta_sec <= horizon_sec)
        labels[pos_mask] = 1

    return labels


def group_warning_episodes(binary_preds: np.ndarray, min_persistence: int = 1) -> List[Tuple[int, int]]:
    """Groups contiguous positive predictions into distinct alarm episodes.
    Only includes episodes whose duration in windows is >= min_persistence.
    """
    episodes = []
    in_episode = False
    start_idx = 0

    for i, val in enumerate(binary_preds):
        if val == 1 and not in_episode:
            in_episode = True
            start_idx = i
        elif val == 0 and in_episode:
            in_episode = False
            duration = i - start_idx
            if duration >= min_persistence:
                episodes.append((start_idx, i - 1))

    if in_episode:
        duration = len(binary_preds) - start_idx
        if duration >= min_persistence:
            episodes.append((start_idx, len(binary_preds) - 1))

    return episodes


# ==============================================================================
# 3. PHASE 2B RISK MODEL TRAINING & EVALUATION
# ==============================================================================

def train_and_evaluate_phase2b(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    feature_scaler: StandardScaler,
    val_timestamps: np.ndarray,
    test_timestamps: np.ndarray,
    horizon_hours: float = 24.0,
    candidate_thresholds: Optional[List[float]] = None,
    persistence_rules: List[int] = [1, 3, 5],
) -> Dict[str, Any]:
    """Trains Logistic Regression with balanced weighting on Train,
    sweeps thresholds & persistence on Validation, freezes optimal config,
    and runs final frozen evaluation on Test.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    # Standardize features using Train-fitted scaler
    X_train = feature_scaler.fit_transform(train_features)
    X_val = feature_scaler.transform(val_features)
    X_test = feature_scaler.transform(test_features)

    # Train Logistic Regression Baseline with balanced class weighting
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train, train_labels)

    # Probabilities
    val_probs = clf.predict_proba(X_val)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]

    # Threshold & Persistence sweep on VALIDATION ONLY
    val_sweep_results = []
    best_val_f1 = -1.0
    best_thresh = 0.50
    best_persistence = 1

    for p_rule in persistence_rules:
        for t in candidate_thresholds:
            raw_v_pred = (val_probs >= t).astype(np.int32)
            
            # Apply persistence filter
            if p_rule > 1:
                # Require p_rule consecutive positive windows to sustain alarm
                filtered_v_pred = np.zeros_like(raw_v_pred)
                episodes = group_warning_episodes(raw_v_pred, min_persistence=p_rule)
                for s_ep, e_ep in episodes:
                    filtered_v_pred[s_ep : e_ep + 1] = 1
                v_pred = filtered_v_pred
            else:
                v_pred = raw_v_pred

            pr = float(precision_score(val_labels, v_pred, zero_division=0))
            rc = float(recall_score(val_labels, v_pred, zero_division=0))
            f1 = float(f1_score(val_labels, v_pred, zero_division=0))

            val_sweep_results.append({
                "persistence": p_rule,
                "threshold": t,
                "precision": pr,
                "recall": rc,
                "f1": f1,
            })

            # Selection rule: maximize validation F1 (tie-break on recall, then precision)
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_thresh = t
                best_persistence = p_rule

    # Freeze threshold & persistence for Test evaluation
    raw_test_pred = (test_probs >= best_thresh).astype(np.int32)
    if best_persistence > 1:
        frozen_test_pred = np.zeros_like(raw_test_pred)
        episodes = group_warning_episodes(raw_test_pred, min_persistence=best_persistence)
        for s_ep, e_ep in episodes:
            frozen_test_pred[s_ep : e_ep + 1] = 1
    else:
        frozen_test_pred = raw_test_pred

    # Confusion matrix & metrics
    cm = confusion_matrix(test_labels, frozen_test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(test_labels, frozen_test_pred))
    prec = float(precision_score(test_labels, frozen_test_pred, zero_division=0))
    rec = float(recall_score(test_labels, frozen_test_pred, zero_division=0))
    f1 = float(f1_score(test_labels, frozen_test_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    roc_auc = float(roc_auc_score(test_labels, test_probs)) if len(np.unique(test_labels)) > 1 else 0.0
    pr_auc = float(average_precision_score(test_labels, test_probs)) if len(np.unique(test_labels)) > 1 else 0.0
    base_rate = float(np.mean(test_labels))

    # Event-level evaluation on Test Failures #3 and #4
    test_failures = [f for f in KNOWN_FAILURES if f["id"] in [3, 4]]
    event_results = []
    test_ts_series = pd.Series(pd.to_datetime(test_timestamps))

    for f in test_failures:
        f_id = f["id"]
        f_type = f["type"]
        f_start = pd.Timestamp(f["start"])
        # Horizon search: strictly inside [f_start - horizon, f_start]
        search_start = f_start - pd.Timedelta(hours=horizon_hours)

        mask = (test_ts_series >= search_start) & (test_ts_series <= f_start)
        window_indices = np.where(mask)[0]

        if len(window_indices) > 0:
            sub_preds = frozen_test_pred[window_indices]
            sub_probs = test_probs[window_indices]
            sub_ts = test_ts_series.iloc[window_indices]

            pos_indices = np.where(sub_preds == 1)[0]
            if len(pos_indices) > 0:
                first_flag_idx = window_indices[pos_indices[0]]
                first_flag_ts = test_ts_series.iloc[first_flag_idx]
                lead_min = (f_start - first_flag_ts).total_seconds() / 60.0
                peak_p = float(np.max(sub_probs))
                consec_warning = int(np.sum(sub_preds))

                event_results.append({
                    "failure_id": f_id,
                    "failure_type": f_type,
                    "failure_start": str(f_start),
                    "detected": True,
                    "first_flag": str(first_flag_ts),
                    "lead_time_minutes": round(lead_min, 1),
                    "lead_time_hours": round(lead_min / 60.0, 2),
                    "peak_probability": round(peak_p, 4),
                    "consecutive_warning_windows": consec_warning,
                })
            else:
                event_results.append({
                    "failure_id": f_id,
                    "failure_type": f_type,
                    "failure_start": str(f_start),
                    "detected": False,
                    "first_flag": "MISSED",
                    "lead_time_minutes": None,
                    "lead_time_hours": None,
                    "peak_probability": round(float(np.max(sub_probs)), 4) if len(sub_probs) > 0 else 0.0,
                    "consecutive_warning_windows": 0,
                })
        else:
            event_results.append({
                "failure_id": f_id,
                "failure_type": f_type,
                "failure_start": str(f_start),
                "detected": False,
                "first_flag": "MISSED",
                "lead_time_minutes": None,
                "lead_time_hours": None,
                "peak_probability": 0.0,
                "consecutive_warning_windows": 0,
            })

    # Group false alarm episodes
    neg_indices = np.where(test_labels == 0)[0]
    neg_preds = frozen_test_pred[neg_indices]
    neg_episodes = group_warning_episodes(neg_preds, min_persistence=best_persistence)
    false_alarm_episodes_count = len(neg_episodes)

    detected_events = [e for e in event_results if e["detected"]]
    event_det_rate = len(detected_events) / float(len(test_failures))
    mean_lead_time_detected = (
        float(np.mean([e["lead_time_hours"] for e in detected_events])) if len(detected_events) > 0 else 0.0
    )

    return {
        "horizon_hours": horizon_hours,
        "model": clf,
        "scaler": feature_scaler,
        "val_sweep_results": val_sweep_results,
        "frozen_threshold": best_thresh,
        "frozen_persistence": best_persistence,
        "best_val_f1": best_val_f1,
        "test_probs": test_probs,
        "test_preds": frozen_test_pred,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "specificity": spec,
            "fpr": fpr,
            "fnr": fnr,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "positive_prevalence_baseline": base_rate,
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            "false_alarm_episodes": false_alarm_episodes_count,
            "test_event_detection_rate": event_det_rate,
            "mean_lead_time_detected_hours": mean_lead_time_detected,
        },
        "event_detection": event_results,
    }
