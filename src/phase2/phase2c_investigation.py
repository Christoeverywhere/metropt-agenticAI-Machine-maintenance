"""
Phase 2C Comprehensive Investigation: Forensic Failure #4 Analysis, False-Alarm Characterization,
Validation-Driven Operational Filtering (Persistence & Alert Cooldown), and Distribution Shift Audit.
"""
import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    ANALOGUE_SENSORS,
    KNOWN_FAILURES,
    SEQUENCE_LENGTH,
    CHECKPOINT_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    SCALER_PATH,
    TIMESTAMP_COL,
)
from src.preprocessing import load_and_clean_data, transform_data
from src.windowing import load_scaler
from src.detector import AnomalyDetector
from src.phase2.dataset import create_phase2_event_splits
from src.phase2.temporal_degradation import (
    compute_phase1_stream_scores,
    build_temporal_degradation_features,
    verify_strict_causality,
    generate_impending_failure_labels,
    group_warning_episodes,
)

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200,
})

PHASE2C_REPORTS_DIR = os.path.join(REPORTS_DIR, "phase2c")
PHASE2C_FIGURES_DIR = os.path.join(PHASE2C_REPORTS_DIR, "figures")
PHASE2C_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "phase2c")

os.makedirs(PHASE2C_REPORTS_DIR, exist_ok=True)
os.makedirs(PHASE2C_FIGURES_DIR, exist_ok=True)
os.makedirs(PHASE2C_CHECKPOINT_DIR, exist_ok=True)


# ==============================================================================
# 1. OPTIONAL CAUSAL REFINED FEATURES (<= 10 Additional Features)
# ==============================================================================

def build_refined_phase2c_features(
    ref_timestamps: np.ndarray,
    seq_errors: np.ndarray,
    sensor_errors: np.ndarray,
    phase1_threshold: float,
    sensor_names: List[str] = ANALOGUE_SENSORS,
    short_window_steps: int = 6,    # ~30m
    med_window_steps: int = 24,     # ~2h
    long_window_steps: int = 72,    # ~6h
) -> Tuple[np.ndarray, List[str]]:
    """Builds the 51 baseline features + 7 scientifically justified refined features (Total: 58).
    
    Refined Causal Features Added:
      1. anomaly_area_above_threshold: integral of excess error sum(max(0, S_j - threshold)) over W_med
      2. rate_of_change_of_DV_contribution: linear slope of DV_pressure contribution over W_med
      3. dominant_sensor_persistence: fraction of windows in W_med where dominant sensor remained identical to current
      4. recent_to_baseline_anomaly_ratio: mean(S_j in W_med) / max(mean(S_j in W_long), 1e-4)
      5. max_sensor_error_spread: max(sensor_error) - min(sensor_error) at current timestamp
      6. anomaly_crest_factor: S_t / max(mean(S_j in W_med), 1e-4)
      7. cumulative_time_above_threshold_hours: total hours above threshold in W_med
    """
    # 1. Base 51 features
    X_base, base_names = build_temporal_degradation_features(
        ref_timestamps, seq_errors, sensor_errors, phase1_threshold,
        sensor_names, short_window_steps, med_window_steps, long_window_steps
    )

    n_samples = len(seq_errors)
    refined_names = [
        "anomaly_area_above_threshold",
        "rate_of_change_of_DV_contribution",
        "dominant_sensor_persistence",
        "recent_to_baseline_anomaly_ratio",
        "max_sensor_error_spread",
        "anomaly_crest_factor",
        "cumulative_time_above_threshold_hours",
    ]

    X_refined = np.zeros((n_samples, len(refined_names)), dtype=np.float32)
    total_sensor_err = np.sum(sensor_errors, axis=1, keepdims=True) + 1e-8
    sensor_contributions = (sensor_errors / total_sensor_err).astype(np.float32)
    dom_indices = np.argmax(sensor_errors, axis=1)

    for i in range(n_samples):
        i_med_start = max(0, i - med_window_steps + 1)
        i_long_start = max(0, i - long_window_steps + 1)

        med_slice = seq_errors[i_med_start : i + 1]
        long_slice = seq_errors[i_long_start : i + 1]
        dv_contrib_slice = sensor_contributions[i_med_start : i + 1, 0]  # DV_pressure
        dom_slice = dom_indices[i_med_start : i + 1]

        # 1. Area above threshold
        excess = np.maximum(0.0, med_slice - phase1_threshold)
        area_above = float(np.sum(excess))

        # 2. Rate of change of DV contribution slope
        # compute simple slope
        n_dv = len(dv_contrib_slice)
        if n_dv >= 2:
            x_ax = np.arange(n_dv)
            x_m = (n_dv - 1) / 2.0
            y_m = np.mean(dv_contrib_slice)
            denom = np.sum((x_ax - x_m) ** 2)
            dv_slope = float(np.sum((x_ax - x_m) * (dv_contrib_slice - y_m)) / denom) if denom > 0 else 0.0
        else:
            dv_slope = 0.0

        # 3. Dominant sensor persistence
        cur_dom = dom_indices[i]
        dom_persist = float(np.mean(dom_slice == cur_dom))

        # 4. Recent to baseline ratio
        mean_med = float(np.mean(med_slice))
        mean_long = float(np.mean(long_slice))
        ratio = mean_med / max(mean_long, 1e-4)

        # 5. Sensor error spread
        spread = float(np.max(sensor_errors[i]) - np.min(sensor_errors[i]))

        # 6. Crest factor
        crest = float(seq_errors[i] / max(mean_med, 1e-4))

        # 7. Cumulative hours above threshold in med window (~2h window)
        # stride 10 = ~100s per window step
        step_hours = (10 * 10.0) / 3600.0  # ~0.0278h per step
        cum_hours = float(np.sum(med_slice > phase1_threshold) * step_hours)

        X_refined[i, 0] = area_above
        X_refined[i, 1] = dv_slope
        X_refined[i, 2] = dom_persist
        X_refined[i, 3] = ratio
        X_refined[i, 4] = spread
        X_refined[i, 5] = crest
        X_refined[i, 6] = cum_hours

    full_X = np.hstack([X_base, X_refined])
    full_names = base_names + refined_names
    return full_X, full_names


# ==============================================================================
# 2. OPERATIONAL FILTERING (PERSISTENCE, COOLDOWN, HYSTERESIS)
# ==============================================================================

def apply_operational_alert_policy(
    probabilities: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    persistence: int = 1,
    cooldown_minutes: float = 0.0,
    hysteresis_exit_ratio: float = 1.0,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Applies operational decision logic on probability stream:
      1. Hysteresis: enter when P >= threshold, exit when P < threshold * hysteresis_exit_ratio.
      2. Persistence: require 'persistence' consecutive active windows to trigger alarm.
      3. Alert Cooldown: once an alert fires, suppress new alert events for 'cooldown_minutes'.
    
    Returns:
        binary_alerts: (N,) binary array of active alert states
        alert_episodes: list of structured alert event dictionaries
    """
    n = len(probabilities)
    raw_active = np.zeros(n, dtype=np.int32)
    exit_thresh = threshold * hysteresis_exit_ratio

    # 1. Hysteresis
    is_in_high_state = False
    for i in range(n):
        p = probabilities[i]
        if not is_in_high_state:
            if p >= threshold:
                is_in_high_state = True
                raw_active[i] = 1
        else:
            if p < exit_thresh:
                is_in_high_state = False
                raw_active[i] = 0
            else:
                raw_active[i] = 1

    # 2. Persistence grouping
    candidate_episodes = group_warning_episodes(raw_active, min_persistence=persistence)

    # 3. Alert Cooldown Management
    ts_series = pd.to_datetime(timestamps)
    final_alerts = np.zeros(n, dtype=np.int32)
    filtered_episodes = []
    last_alert_time = None

    for start_idx, end_idx in candidate_episodes:
        ep_start_ts = ts_series[start_idx]
        ep_end_ts = ts_series[end_idx]

        if last_alert_time is not None and cooldown_minutes > 0.0:
            elapsed_m = (ep_start_ts - last_alert_time).total_seconds() / 60.0
            if elapsed_m < cooldown_minutes:
                # Suppressed by active cooldown
                continue

        final_alerts[start_idx : end_idx + 1] = 1
        last_alert_time = ep_start_ts

        filtered_episodes.append({
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_timestamp": str(ep_start_ts),
            "end_timestamp": str(ep_end_ts),
            "duration_windows": int(end_idx - start_idx + 1),
            "duration_minutes": round((ep_end_ts - ep_start_ts).total_seconds() / 60.0, 1),
            "peak_probability": round(float(np.max(probabilities[start_idx : end_idx + 1])), 4),
            "mean_probability": round(float(np.mean(probabilities[start_idx : end_idx + 1])), 4),
        })

    return final_alerts, filtered_episodes


# ==============================================================================
# 3. FORENSIC COMPARISON: FAILURE #3 vs FAILURE #4
# ==============================================================================

def perform_failure_forensic_analysis(
    timestamps: np.ndarray,
    seq_errors: np.ndarray,
    sensor_errors: np.ndarray,
    risk_probs: np.ndarray,
    features_df: pd.DataFrame,
    phase1_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Analyzes Failure #3 vs Failure #4 at lookback checkpoints:
    [T-48h, T-36h, T-24h, T-18h, T-12h, T-6h, T-3h, T-1h, T]
    """
    ts = pd.to_datetime(timestamps)
    intervals_hours = [48.0, 36.0, 24.0, 18.0, 12.0, 6.0, 3.0, 1.0, 0.0]

    rows = []
    test_failures = [f for f in KNOWN_FAILURES if f["id"] in [3, 4]]

    for f in test_failures:
        f_id = f["id"]
        f_start = pd.Timestamp(f["start"])

        for h in intervals_hours:
            target_ts = f_start - pd.Timedelta(hours=h)
            # Find nearest index <= target_ts
            valid_mask = ts <= target_ts
            if not np.any(valid_mask):
                continue
            idx = np.where(valid_mask)[0][-1]
            actual_ts = ts[idx]

            # Sensor errors
            s_err = sensor_errors[idx]
            total_err = np.sum(s_err) + 1e-8
            dom_idx = int(np.argmax(s_err))
            dom_name = ANALOGUE_SENSORS[dom_idx]
            dom_contrib = float((s_err[dom_idx] / total_err) * 100.0)

            row_dict = {
                "failure_id": f_id,
                "failure_start": str(f_start),
                "checkpoint": f"T-{h:.0f}h" if h > 0 else "T (Failure Start)",
                "checkpoint_target_ts": str(target_ts),
                "actual_timestamp": str(actual_ts),
                "time_to_failure_hours": round((f_start - actual_ts).total_seconds() / 3600.0, 2),
                "mean_anomaly_score": round(float(features_df["mean_anomaly_recent"].iloc[idx]), 4),
                "max_anomaly_score": round(float(features_df["max_anomaly_recent"].iloc[idx]), 4),
                "current_anomaly_score": round(float(seq_errors[idx]), 4),
                "anomaly_slope": round(float(features_df["anomaly_slope_medium"].iloc[idx]), 6),
                "anomaly_acceleration": round(float(features_df["anomaly_acceleration"].iloc[idx]), 6),
                "anomaly_persistence": round(float(features_df["fraction_above_threshold_recent"].iloc[idx]), 4),
                "dominant_sensor": dom_name,
                "dominant_sensor_contribution_pct": round(dom_contrib, 2),
                "phase2_risk_probability": round(float(risk_probs[idx]), 4),
            }

            for s_i, s_name in enumerate(ANALOGUE_SENSORS):
                row_dict[f"{s_name}_error"] = round(float(s_err[s_i]), 4)

            rows.append(row_dict)

    df_forensics = pd.DataFrame(rows)
    return df_forensics


# ==============================================================================
# 4. FALSE-ALARM EPISODE CHARACTERIZATION
# ==============================================================================

def characterize_false_alarm_episodes(
    timestamps: np.ndarray,
    ground_truth_labels: np.ndarray,
    risk_probs: np.ndarray,
    seq_errors: np.ndarray,
    sensor_errors: np.ndarray,
    features_df: pd.DataFrame,
    threshold: float,
    persistence: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Forensic characterization of all false-alarm episodes across Test split."""
    raw_preds = (risk_probs >= threshold).astype(np.int32)
    # Find false alarm windows: where ground_truth == 0 and pred == 1
    neg_mask = (ground_truth_labels == 0)
    fa_window_preds = np.zeros_like(raw_preds)
    fa_window_preds[neg_mask] = raw_preds[neg_mask]

    episodes = group_warning_episodes(fa_window_preds, min_persistence=persistence)
    ts = pd.Series(pd.to_datetime(timestamps))

    fa_rows = []
    for s_idx, e_idx in episodes:
        ep_ts_slice = ts.iloc[s_idx : e_idx + 1]
        ep_prob_slice = risk_probs[s_idx : e_idx + 1]
        ep_err_slice = seq_errors[s_idx : e_idx + 1]
        ep_sens_slice = sensor_errors[s_idx : e_idx + 1]

        duration_w = len(ep_ts_slice)
        duration_m = (ep_ts_slice.iloc[-1] - ep_ts_slice.iloc[0]).total_seconds() / 60.0
        peak_p = float(np.max(ep_prob_slice))
        mean_p = float(np.mean(ep_prob_slice))
        peak_mse = float(np.max(ep_err_slice))
        mean_mse = float(np.mean(ep_err_slice))

        # Dominant sensor across episode
        sum_sens = np.sum(ep_sens_slice, axis=0)
        dom_idx = int(np.argmax(sum_sens))
        dom_name = ANALOGUE_SENSORS[dom_idx]
        dom_contrib = float((sum_sens[dom_idx] / (np.sum(sum_sens) + 1e-8)) * 100.0)

        # Persistence & Slope
        max_slope = float(np.max(features_df["anomaly_slope_medium"].iloc[s_idx : e_idx + 1]))
        mean_persist = float(np.mean(features_df["fraction_above_threshold_recent"].iloc[s_idx : e_idx + 1]))

        # Category
        if duration_w <= 3:
            category = "short_isolated_spike"
        elif mean_persist > 0.5:
            category = "persistent_abnormality"
        elif max_slope > 0.01:
            category = "rapid_drift"
        elif dom_contrib > 80.0:
            category = f"sensor_specific_{dom_name}"
        else:
            category = "operational_regime_noise"

        start_t = ep_ts_slice.iloc[0]
        fa_rows.append({
            "episode_id": len(fa_rows) + 1,
            "start_timestamp": str(start_t),
            "end_timestamp": str(ep_ts_slice.iloc[-1]),
            "duration_windows": duration_w,
            "duration_minutes": round(duration_m, 1),
            "peak_risk_prob": round(peak_p, 4),
            "mean_risk_prob": round(mean_p, 4),
            "peak_anomaly_mse": round(peak_mse, 4),
            "mean_anomaly_mse": round(mean_mse, 4),
            "dominant_sensor": dom_name,
            "dominant_sensor_contribution_pct": round(dom_contrib, 2),
            "max_anomaly_slope": round(max_slope, 6),
            "mean_persistence": round(mean_persist, 4),
            "category": category,
            "hour_of_day": start_t.hour,
            "day_of_week": start_t.day_name(),
            "month": start_t.month_name(),
        })

    df_fa = pd.DataFrame(fa_rows)

    # Breakdown summary table
    breakdown_summary = []
    if len(df_fa) > 0:
        for cat, grp in df_fa.groupby("category"):
            breakdown_summary.append({
                "category": cat,
                "episode_count": len(grp),
                "pct_of_total": round(len(grp) / len(df_fa) * 100.0, 2),
                "mean_duration_minutes": round(grp["duration_minutes"].mean(), 2),
                "mean_peak_prob": round(grp["peak_risk_prob"].mean(), 4),
                "mean_peak_mse": round(grp["peak_anomaly_mse"].mean(), 4),
            })
    df_fa_summary = pd.DataFrame(breakdown_summary)
    return df_fa, df_fa_summary


# ==============================================================================
# 5. VALIDATION SWEEPS: PERSISTENCE, COOLDOWN & HYSTERESIS
# ==============================================================================

def sweep_validation_operational_policies(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    val_timestamps: np.ndarray,
    candidate_thresholds: List[float] = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
    persistence_list: List[int] = [1, 3, 5, 10, 20],
    cooldown_minutes_list: List[float] = [0.0, 30.0, 60.0, 180.0, 360.0],
    hysteresis_ratios: List[float] = [1.0, 0.75, 0.50],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Sweeps operational filtering parameters ON VALIDATION ONLY to select optimal policy."""
    val_failures = [f for f in KNOWN_FAILURES if f["id"] == 2]  # Failure #2 in Val
    val_ts_series = pd.Series(pd.to_datetime(val_timestamps))
    f2_start = pd.Timestamp(val_failures[0]["start"])

    records_persist = []
    records_cooldown = []

    best_val_f1 = -1.0
    best_config = {
        "threshold": 0.01,
        "persistence": 1,
        "cooldown_minutes": 0.0,
        "hysteresis_ratio": 1.0,
    }

    # 1. Sweep Persistence across Thresholds (at cooldown=0)
    for p in persistence_list:
        for t in candidate_thresholds:
            for hyst in hysteresis_ratios:
                alerts, episodes = apply_operational_alert_policy(
                    val_probs, val_timestamps, threshold=t, persistence=p,
                    cooldown_minutes=0.0, hysteresis_exit_ratio=hyst
                )

                pr = float(precision_score(val_labels, alerts, zero_division=0))
                rc = float(recall_score(val_labels, alerts, zero_division=0))
                f1 = float(f1_score(val_labels, alerts, zero_division=0))
                
                # Val False Alarm Episodes
                neg_mask = (val_labels == 0)
                fa_alerts = np.zeros_like(alerts)
                fa_alerts[neg_mask] = alerts[neg_mask]
                fa_eps = len(group_warning_episodes(fa_alerts, min_persistence=p))

                # Event detection for Failure #2
                f2_mask = (val_ts_series >= f2_start - pd.Timedelta(hours=24)) & (val_ts_series <= f2_start)
                f2_detected = bool(np.any(alerts[f2_mask] == 1))

                records_persist.append({
                    "persistence": p,
                    "threshold": t,
                    "hysteresis_ratio": hyst,
                    "val_precision": round(pr, 4),
                    "val_recall": round(rc, 4),
                    "val_f1": round(f1, 4),
                    "val_fa_episodes": fa_eps,
                    "val_failure2_detected": f2_detected,
                })

                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_config = {
                        "threshold": t,
                        "persistence": p,
                        "cooldown_minutes": 0.0,
                        "hysteresis_ratio": hyst,
                    }

    # 2. Sweep Cooldown on top of best persistence & threshold
    frozen_t = best_config["threshold"]
    frozen_p = best_config["persistence"]
    frozen_hyst = best_config["hysteresis_ratio"]

    for cd in cooldown_minutes_list:
        alerts, episodes = apply_operational_alert_policy(
            val_probs, val_timestamps, threshold=frozen_t, persistence=frozen_p,
            cooldown_minutes=cd, hysteresis_exit_ratio=frozen_hyst
        )

        pr = float(precision_score(val_labels, alerts, zero_division=0))
        rc = float(recall_score(val_labels, alerts, zero_division=0))
        f1 = float(f1_score(val_labels, alerts, zero_division=0))

        # Count false alarm episodes
        fa_count = 0
        for ep in episodes:
            ep_start = pd.Timestamp(ep["start_timestamp"])
            if not ((ep_start >= f2_start - pd.Timedelta(hours=24)) and (ep_start <= f2_start + pd.Timedelta(hours=6))):
                fa_count += 1

        f2_mask = (val_ts_series >= f2_start - pd.Timedelta(hours=24)) & (val_ts_series <= f2_start)
        f2_detected = bool(np.any(alerts[f2_mask] == 1))

        records_cooldown.append({
            "cooldown_minutes": cd,
            "threshold": frozen_t,
            "persistence": frozen_p,
            "hysteresis_ratio": frozen_hyst,
            "val_precision": round(pr, 4),
            "val_recall": round(rc, 4),
            "val_f1": round(f1, 4),
            "val_total_episodes": len(episodes),
            "val_fa_episodes": fa_count,
            "val_failure2_detected": f2_detected,
        })

    # Select operational cooldown that maximizes F1 / minimizes FA while keeping Failure #2 detected
    valid_cds = [r for r in records_cooldown if r["val_failure2_detected"]]
    if valid_cds:
        best_cd_rec = max(valid_cds, key=lambda x: (x["val_f1"], -x["val_fa_episodes"]))
        best_config["cooldown_minutes"] = best_cd_rec["cooldown_minutes"]

    df_persist = pd.DataFrame(records_persist)
    df_cooldown = pd.DataFrame(records_cooldown)
    return df_persist, df_cooldown, best_config


# ==============================================================================
# 6. DISTRIBUTION SHIFT AUDIT
# ==============================================================================

def audit_feature_distributions(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Audits feature distributions across Train, Val, and Test splits."""
    stats = []
    shift_summary = []

    for idx, fname in enumerate(feature_names):
        tr_col = X_train[:, idx]
        va_col = X_val[:, idx]
        te_col = X_test[:, idx]

        tr_mean, tr_std = float(np.mean(tr_col)), float(np.std(tr_col))
        va_mean, va_std = float(np.mean(va_col)), float(np.std(va_col))
        te_mean, te_std = float(np.mean(te_col)), float(np.std(te_col))

        stats.append({
            "feature": fname,
            "train_mean": round(tr_mean, 4),
            "train_std": round(tr_std, 4),
            "train_min": round(float(np.min(tr_col)), 4),
            "train_max": round(float(np.max(tr_col)), 4),
            "val_mean": round(va_mean, 4),
            "val_std": round(va_std, 4),
            "val_min": round(float(np.min(va_col)), 4),
            "val_max": round(float(np.max(va_col)), 4),
            "test_mean": round(te_mean, 4),
            "test_std": round(te_std, 4),
            "test_min": round(float(np.min(te_col)), 4),
            "test_max": round(float(np.max(te_col)), 4),
        })

        # Z-score shift between Test and Train
        z_shift = abs(te_mean - tr_mean) / max(tr_std, 1e-4)
        shift_summary.append({
            "feature": fname,
            "train_mean": round(tr_mean, 4),
            "test_mean": round(te_mean, 4),
            "z_score_shift": round(z_shift, 4),
            "is_significant_shift": bool(z_shift > 1.5),
        })

    df_stats = pd.DataFrame(stats)
    df_shift = pd.DataFrame(shift_summary).sort_values("z_score_shift", ascending=False)
    return df_stats, df_shift


# ==============================================================================
# 7. GENERATE ALL 9 REQUIRED VISUALIZATIONS
# ==============================================================================

def generate_phase2c_figures(
    test_ts: np.ndarray,
    test_mse: np.ndarray,
    test_sens_err: np.ndarray,
    test_probs: np.ndarray,
    df_fa: pd.DataFrame,
    df_persist: pd.DataFrame,
    df_cooldown: pd.DataFrame,
    df_shift: pd.DataFrame,
    frozen_config: Dict[str, Any],
    phase1_threshold: float,
):
    """Generates all 9 publication-quality diagnostic plots into reports/phase2c/figures/."""
    ts = pd.to_datetime(test_ts)
    f3 = [f for f in KNOWN_FAILURES if f["id"] == 3][0]
    f4 = [f for f in KNOWN_FAILURES if f["id"] == 4][0]

    f3_start = pd.Timestamp(f3["start"])
    f4_start = pd.Timestamp(f4["start"])

    # --------------------------------------------------------------------------
    # Figure 1: Failure #3 vs Failure #4 Phase-1 Anomaly Trajectories
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
    for ax, f, f_start in zip(axes, [f3, f4], [f3_start, f4_start]):
        mask = (ts >= f_start - pd.Timedelta(hours=72)) & (ts <= f_start + pd.Timedelta(hours=6))
        ax.plot(ts[mask], test_mse[mask], color="#1f77b4", lw=1.5, label="Phase 1 Anomaly Score (MSE)")
        ax.axhline(phase1_threshold, color="#d9534f", linestyle="--", lw=1.5, label=f"P99 Threshold ({phase1_threshold:.2f})")
        ax.axvspan(f_start - pd.Timedelta(hours=24), f_start, color="#ffa726", alpha=0.25, label="24h Horizon")
        ax.axvspan(f_start, f_start + pd.Timedelta(hours=6), color="#ff4d4d", alpha=0.35, label=f"Failure #{f['id']} Incident")
        ax.set_title(f"Failure #{f['id']} (Start: {f_start.strftime('%m-%d %H:%M')})", fontweight="bold")
        ax.set_xlabel("Date & Time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("Reconstruction MSE")
    plt.suptitle("Figure 1: 72-Hour Pre-Failure Anomaly Progression — Failure #3 vs Failure #4", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PHASE2C_FIGURES_DIR, "fig1_failure3_vs_failure4_anomaly_trajectories.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 2: Failure #3 vs Failure #4 Sensor Reconstruction Errors
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for ax, f, f_start in zip(axes, [f3, f4], [f3_start, f4_start]):
        mask = (ts >= f_start - pd.Timedelta(hours=72)) & (ts <= f_start + pd.Timedelta(hours=6))
        for s_i, s_name in enumerate(ANALOGUE_SENSORS):
            ax.plot(ts[mask], test_sens_err[mask, s_i], lw=1.3, color=colors[s_i % len(colors)], label=s_name)
        ax.axvspan(f_start - pd.Timedelta(hours=24), f_start, color="#ffa726", alpha=0.2, label="24h Horizon")
        ax.axvspan(f_start, f_start + pd.Timedelta(hours=6), color="#ff4d4d", alpha=0.3, label=f"Failure #{f['id']}")
        ax.set_title(f"Figure 2: Sensor Decomposition 72h Prior to Failure #{f['id']}", fontweight="bold")
        ax.set_ylabel("Sensor MSE")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.legend(loc="upper left", ncol=4, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PHASE2C_FIGURES_DIR, "fig2_sensor_reconstruction_errors_f3_f4.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 3 & 4: Risk Probability around Failure #3 & #4
    # --------------------------------------------------------------------------
    for f, f_start, f_num, fname in [(f3, f3_start, 3, "fig3_risk_probability_failure3.png"), (f4, f4_start, 4, "fig4_risk_probability_failure4.png")]:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        mask = (ts >= f_start - pd.Timedelta(hours=48)) & (ts <= f_start + pd.Timedelta(hours=6))
        ax.plot(ts[mask], test_probs[mask], color="#800080", lw=2.0, label="Phase 2 Risk Probability $P(y=1)$")
        ax.axhline(frozen_config["threshold"], color="#d9534f", linestyle="--", lw=1.8, label=f"Operating Threshold ({frozen_config['threshold']:.2f})")
        ax.axvspan(f_start - pd.Timedelta(hours=24), f_start, color="#ffa726", alpha=0.25, label="24h Impending Horizon")
        ax.axvspan(f_start, f_start + pd.Timedelta(hours=6), color="#ff4d4d", alpha=0.35, label=f"Failure #{f_num} Incident")
        ax.set_title(f"Figure {3 if f_num==3 else 4}: Phase-2 Failure Risk Probability Timeline — Failure #{f_num}", fontweight="bold")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Predicted Risk $P(y=1)$")
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.legend(loc="upper left")
        plt.tight_layout()
        fig.savefig(os.path.join(PHASE2C_FIGURES_DIR, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 5: False-Alarm Episode Duration Distribution
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if len(df_fa) > 0:
        durations = df_fa["duration_minutes"]
        ax.hist(durations, bins=40, color="#2b5c8f", edgecolor="black", alpha=0.75)
        ax.axvline(durations.median(), color="#d9534f", linestyle="--", lw=1.8, label=f"Median Duration ({durations.median():.1f} min)")
        ax.set_title("Figure 5: False-Alarm Episode Duration Distribution (Test Split)", fontweight="bold")
        ax.set_xlabel("Duration (Minutes)")
        ax.set_ylabel("Episode Count")
        ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(PHASE2C_FIGURES_DIR, "fig5_false_alarm_duration_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 6: False Alarms by Dominant Sensor
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if len(df_fa) > 0:
        dom_counts = df_fa["dominant_sensor"].value_counts()
        dom_counts.plot(kind="bar", color="#ff7f0e", edgecolor="black", ax=ax)
        ax.set_title("Figure 6: Distribution of False-Alarm Episodes by Dominant Anomaly Indicator", fontweight="bold")
        ax.set_xlabel("Dominant Sensor")
        ax.set_ylabel("False-Alarm Episodes Count")
        plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(PHASE2C_FIGURES_DIR, "fig6_false_alarms_by_dominant_sensor.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 7: Validation Persistence Analysis
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for p in [1, 3, 5, 10, 20]:
        sub = df_persist[(df_persist["persistence"] == p) & (df_persist["hysteresis_ratio"] == 1.0)]
        ax.plot(sub["threshold"], sub["val_f1"], marker="o", lw=1.5, label=f"Persistence P={p}")
    ax.set_title("Figure 7: Validation F1-Score vs. Threshold across Temporal Persistence Rules", fontweight="bold")
    ax.set_xlabel("Risk Decision Threshold")
    ax.set_ylabel("Validation F1-Score")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(PHASE2C_FIGURES_DIR, "fig7_validation_persistence_analysis.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 8: Validation Cooldown Analysis
    # --------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(df_cooldown["cooldown_minutes"], df_cooldown["val_fa_episodes"], color="#d62728", marker="s", lw=2.0, label="False-Alarm Episodes (Val)")
    ax2.plot(df_cooldown["cooldown_minutes"], df_cooldown["val_f1"], color="#1f77b4", marker="o", lw=2.0, label="Validation F1-Score")
    ax1.set_xlabel("Alert Cooldown Window (Minutes)")
    ax1.set_ylabel("False-Alarm Episodes", color="#d62728")
    ax2.set_ylabel("Validation F1-Score", color="#1f77b4")
    ax1.set_title("Figure 8: Operational Alert Cooldown Tradeoff on Validation Set", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PHASE2C_FIGURES_DIR, "fig8_validation_cooldown_analysis.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure 9: Train vs Validation vs Test Feature Distribution Shift
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    top_shift = df_shift.head(10)
    y_pos = np.arange(len(top_shift))
    ax.barh(y_pos, top_shift["z_score_shift"], color="#2ca02c", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_shift["feature"])
    ax.invert_yaxis()
    ax.axvline(1.5, color="#d9534f", linestyle="--", lw=1.5, label="Significant Shift Threshold (|Z| > 1.5)")
    ax.set_title("Figure 9: Top 10 Feature Distribution Shifts (|Z-Score|) Between Train and Test", fontweight="bold")
    ax.set_xlabel("Absolute Z-Score Shift (|Test Mean - Train Mean| / Train Std)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(PHASE2C_FIGURES_DIR, "fig9_distribution_shift_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualizer] Successfully generated all 9 figures in: {PHASE2C_FIGURES_DIR}")


# ==============================================================================
# 8. MASTER PHASE 2C EXECUTION ENGINE
# ==============================================================================

def run_phase2c_master():
    print("=" * 110)
    print("       METROPT-3 PREDICTIVE MAINTENANCE — PHASE 2C: FALSE ALARM REDUCTION & FORENSIC AUDIT")
    print("=" * 110)

    # 1. Load Data & Create Chronological Event Splits
    raw_df = load_and_clean_data()
    train_df, val_df, test_df = create_phase2_event_splits(raw_df)

    # Load Scaler (fit on TRAIN only)
    scaler = load_scaler(SCALER_PATH)
    scaled_train = transform_data(train_df, scaler)
    scaled_val = transform_data(val_df, scaler)
    scaled_test = transform_data(test_df, scaler)

    # 2. Phase 1 Denoising Autoencoder
    phase1_ckpt_path = os.path.join(CHECKPOINT_DIR, "denoising_dense_ae.pt")
    phase1_threshold = 2.913448
    detector = AnomalyDetector(
        model_path=phase1_ckpt_path,
        scaler=scaler,
        model_type="dense",
        threshold=phase1_threshold,
    )

    # 3. Extract Phase 1 Scores & Per-Sensor Decomposition
    print("\n[features] Extracting Phase 1 reconstruction MSE and per-sensor decomposition...")
    tr_ts, tr_mse, tr_sens_err, _ = compute_phase1_stream_scores(scaled_train, train_df["timestamp"], detector, stride=10)
    va_ts, va_mse, va_sens_err, _ = compute_phase1_stream_scores(scaled_val, val_df["timestamp"], detector, stride=10)
    te_ts, te_mse, te_sens_err, _ = compute_phase1_stream_scores(scaled_test, test_df["timestamp"], detector, stride=10)

    # 4. Build Refined Causal Features
    print("\n[features] Constructing causal temporal features with refined degradation indicators (58 features)...")
    X_tr, feat_names = build_refined_phase2c_features(tr_ts, tr_mse, tr_sens_err, phase1_threshold=phase1_threshold)
    X_va, _ = build_refined_phase2c_features(va_ts, va_mse, va_sens_err, phase1_threshold=phase1_threshold)
    X_te, _ = build_refined_phase2c_features(te_ts, te_mse, te_sens_err, phase1_threshold=phase1_threshold)

    # Audit causality
    verify_strict_causality(te_ts, n_checks=10)

    # 5. Distribution Shift & Feature Scaling Audit
    print("\n[audit] Auditing feature distributions and standardizer compliance...")
    feat_scaler = StandardScaler()
    # Fit scaler on TRAIN ONLY
    X_tr_scaled = feat_scaler.fit_transform(X_tr)
    X_va_scaled = feat_scaler.transform(X_va)
    X_te_scaled = feat_scaler.transform(X_te)

    df_stats, df_shift = audit_feature_distributions(X_tr, X_va, X_te, feat_names)
    df_stats.to_csv(os.path.join(PHASE2C_REPORTS_DIR, "feature_statistics.csv"), index=False)
    df_shift.to_csv(os.path.join(PHASE2C_REPORTS_DIR, "distribution_shift.csv"), index=False)

    # 6. Fit Logistic Regression on Train (Class-weighted balanced)
    y_tr_24h = generate_impending_failure_labels(tr_ts, horizon_hours=24.0)
    y_va_24h = generate_impending_failure_labels(va_ts, horizon_hours=24.0)
    y_te_24h = generate_impending_failure_labels(te_ts, horizon_hours=24.0)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_tr_scaled, y_tr_24h)

    # Feature Coefficients Table
    coefs = clf.coef_[0]
    df_coef = pd.DataFrame({
        "feature": feat_names,
        "coefficient": np.round(coefs, 5),
        "abs_coefficient": np.round(np.abs(coefs), 5),
        "direction": ["Positive Risk" if c > 0 else "Negative / Inhibitory" for c in coefs],
    }).sort_values("abs_coefficient", ascending=False)
    df_coef.to_csv(os.path.join(PHASE2C_REPORTS_DIR, "feature_coefficients.csv"), index=False)

    # Model Probabilities
    va_probs = clf.predict_proba(X_va_scaled)[:, 1]
    te_probs = clf.predict_proba(X_te_scaled)[:, 1]

    # 7. Sweep Validation Operational Policies (Persistence, Cooldown, Hysteresis)
    print("\n[validation] Sweeping persistence, hysteresis, and alert cooldown policies on VALIDATION DATA ONLY...")
    df_persist, df_cooldown, frozen_config = sweep_validation_operational_policies(va_probs, y_va_24h, va_ts)
    df_persist.to_csv(os.path.join(PHASE2C_REPORTS_DIR, "persistence_validation.csv"), index=False)
    df_cooldown.to_csv(os.path.join(PHASE2C_REPORTS_DIR, "cooldown_validation.csv"), index=False)

    print(f"[validation] Selected Frozen Configuration -> Threshold: {frozen_config['threshold']:.2f}, "
          f"Persistence: {frozen_config['persistence']} windows, Cooldown: {frozen_config['cooldown_minutes']:.1f} min, "
          f"Hysteresis Ratio: {frozen_config['hysteresis_ratio']:.2f}")

    # 8. Forensic Failure #3 vs Failure #4 Analysis
    print("\n[forensics] Executing deep forensic lookback analysis of Failure #3 vs Failure #4...")
    te_feat_df = pd.DataFrame(X_te, columns=feat_names)
    df_forensics = perform_failure_forensic_analysis(te_ts, te_mse, te_sens_err, te_probs, te_feat_df, phase1_threshold)
    df_forensics.to_csv(os.path.join(PHASE2C_REPORTS_DIR, "failure3_vs_failure4_analysis.csv"), index=False)

    # 9. False-Alarm Episode Characterization
    print("\n[false_alarms] Characterizing all false alarm episodes across Test Split...")
    df_fa, df_fa_summary = characterize_false_alarm_episodes(
        te_ts, y_te_24h, te_probs, te_mse, te_sens_err, te_feat_df,
        threshold=frozen_config["threshold"], persistence=frozen_config["persistence"]
    )
    df_fa.to_csv(os.path.join(PHASE2C_REPORTS_DIR, "false_alarm_analysis.csv"), index=False)

    # 10. Multi-Horizon Test Evaluation under Frozen Operational Policy
    print("\n" + "=" * 110)
    print("                     PHASE 2C FROZEN TEST EVALUATION ACROSS HORIZONS")
    print("=" * 110)
    
    phase2c_horizons = {}
    test_days = (pd.Timestamp(te_ts[-1]) - pd.Timestamp(te_ts[0])).total_seconds() / 86400.0

    primary_alerts = None

    for h in [6.0, 12.0, 24.0, 48.0]:
        y_te_h = generate_impending_failure_labels(te_ts, horizon_hours=h)
        y_va_h = generate_impending_failure_labels(va_ts, horizon_hours=h)
        y_tr_h = generate_impending_failure_labels(tr_ts, horizon_hours=h)

        clf_h = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        clf_h.fit(X_tr_scaled, y_tr_h)
        h_te_probs = clf_h.predict_proba(X_te_scaled)[:, 1]

        alerts_h, eps_h = apply_operational_alert_policy(
            h_te_probs, te_ts,
            threshold=frozen_config["threshold"],
            persistence=frozen_config["persistence"],
            cooldown_minutes=frozen_config["cooldown_minutes"],
            hysteresis_exit_ratio=frozen_config["hysteresis_ratio"],
        )

        if h == 24.0:
            primary_alerts = alerts_h

        # Metrics
        cm = confusion_matrix(y_te_h, alerts_h, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = float(accuracy_score(y_te_h, alerts_h))
        prec = float(precision_score(y_te_h, alerts_h, zero_division=0))
        rec = float(recall_score(y_te_h, alerts_h, zero_division=0))
        f1 = float(f1_score(y_te_h, alerts_h, zero_division=0))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        roc_auc = float(roc_auc_score(y_te_h, h_te_probs))
        pr_auc = float(average_precision_score(y_te_h, h_te_probs))
        base_rate = float(np.mean(y_te_h))

        # Count genuine false alarm episodes
        fa_eps_count = 0
        test_fails = [f for f in KNOWN_FAILURES if f["id"] in [3, 4]]
        for ep in eps_h:
            ep_start = pd.Timestamp(ep["start_timestamp"])
            is_true_alarm = False
            for f in test_fails:
                f_st = pd.Timestamp(f["start"])
                if (ep_start >= f_st - pd.Timedelta(hours=h)) and (ep_start <= f_st):
                    is_true_alarm = True
                    break
            if not is_true_alarm:
                fa_eps_count += 1

        fa_per_day = fa_eps_count / max(test_days, 1.0)

        # Event Detection
        event_dets = []
        te_ts_series = pd.Series(pd.to_datetime(te_ts))
        for f in test_fails:
            f_st = pd.Timestamp(f["start"])
            mask = (te_ts_series >= f_st - pd.Timedelta(hours=h)) & (te_ts_series <= f_st)
            sub_alerts = alerts_h[mask]
            sub_probs = h_te_probs[mask]
            if np.any(sub_alerts == 1):
                first_flag_idx = np.where(mask)[0][np.where(sub_alerts == 1)[0][0]]
                first_flag_ts = te_ts_series.iloc[first_flag_idx]
                lead_min = (f_st - first_flag_ts).total_seconds() / 60.0
                event_dets.append({
                    "failure_id": f["id"],
                    "failure_type": f["type"],
                    "failure_start": str(f_st),
                    "detected": True,
                    "first_flag": str(first_flag_ts),
                    "lead_time_minutes": round(lead_min, 1),
                    "lead_time_hours": round(lead_min / 60.0, 2),
                    "peak_probability": round(float(np.max(sub_probs)), 4),
                })
            else:
                event_dets.append({
                    "failure_id": f["id"],
                    "failure_type": f["type"],
                    "failure_start": str(f_st),
                    "detected": False,
                    "first_flag": "MISSED",
                    "lead_time_minutes": None,
                    "lead_time_hours": None,
                    "peak_probability": round(float(np.max(sub_probs)), 4) if len(sub_probs) > 0 else 0.0,
                })

        det_events = [e for e in event_dets if e["detected"]]
        det_rate = len(det_events) / float(len(test_fails))
        mean_lead = float(np.mean([e["lead_time_hours"] for e in det_events])) if det_events else 0.0

        phase2c_horizons[f"{int(h)}h"] = {
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
                "false_alarm_episodes": fa_eps_count,
                "false_alarm_episodes_per_day": round(fa_per_day, 2),
                "test_event_detection_rate": det_rate,
                "mean_lead_time_detected_hours": mean_lead,
            },
            "event_detection": event_dets,
        }

        print(
            f"Horizon: {int(h):2d}h | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} (Base: {base_rate:.4f}) | "
            f"F1: {f1:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f} | Spec: {spec:.4f} | "
            f"FA Episodes: {fa_eps_count:4d} ({fa_per_day:.1f}/day) | Events: {len(det_events)}/2 ({det_rate*100:.0f}%)"
        )

    # 11. Save Phase 2C Outputs
    print("\n[storage] Saving Phase 2C prediction stream, metrics JSON, and event detection...")
    preds_df = pd.DataFrame({
        "timestamp": te_ts,
        "ground_truth_label_24h": y_te_24h,
        "predicted_risk_probability": te_probs,
        "operational_alert_class": primary_alerts,
        "anomaly_score_mse": te_mse,
    })
    preds_df.to_csv(os.path.join(PHASE2C_REPORTS_DIR, "phase2c_predictions.csv"), index=False)

    metrics_payload = {
        "pipeline": "Phase 2C — Operational False Alarm Reduction & Forensic Audit",
        "frozen_operational_policy": frozen_config,
        "test_period_days": round(test_days, 2),
        "horizons": phase2c_horizons,
    }
    with open(os.path.join(PHASE2C_REPORTS_DIR, "phase2c_metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    with open(os.path.join(PHASE2C_REPORTS_DIR, "phase2c_event_detection.json"), "w") as f:
        json.dump(phase2c_horizons["24h"]["event_detection"], f, indent=2)

    # 12. Generate Visualizations
    print("\n[visualizer] Rendering all 9 required publication figures...")
    generate_phase2c_figures(
        te_ts, te_mse, te_sens_err, te_probs, df_fa, df_persist, df_cooldown, df_shift,
        frozen_config, phase1_threshold
    )

    print("\n=== PHASE 2C INVESTIGATION COMPLETED ===")
    return metrics_payload, df_forensics, df_fa_summary, df_coef, df_shift, frozen_config


if __name__ == "__main__":
    run_phase2c_master()
