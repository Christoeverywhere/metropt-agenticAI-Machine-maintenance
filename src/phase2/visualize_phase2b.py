"""
Phase 2B Visualizer: 6 Publication-Quality Diagnostic & Risk Figures.
"""
import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from src.config import ANALOGUE_SENSORS, KNOWN_FAILURES, FIGURES_DIR
except ImportError:
    from config import ANALOGUE_SENSORS, KNOWN_FAILURES, FIGURES_DIR

# Set publication style
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


def plot_1_anomaly_score_timeline(
    all_timestamps: np.ndarray,
    all_scores: np.ndarray,
    phase1_threshold: float,
    failures: List[Dict] = KNOWN_FAILURES,
    save_path: Optional[str] = None,
):
    """Plot 1: Full-stream Phase 1 anomaly score over time with documented failure periods marked."""
    fig, ax = plt.subplots(figsize=(15, 5))
    ts = pd.to_datetime(all_timestamps)

    ax.plot(ts, all_scores, color="#2b5c8f", alpha=0.7, lw=1.0, label="Phase 1 Anomaly Score (MSE)")
    ax.axhline(phase1_threshold, color="#d9534f", linestyle="--", lw=1.8, label=f"P99 Threshold ({phase1_threshold:.2f})")

    # Mark all 4 failure episodes
    for f in failures:
        f_start = pd.Timestamp(f["start"])
        f_end = pd.Timestamp(f["end"])
        ax.axvspan(f_start, f_end, color="#ff4d4d", alpha=0.35, label=f"Failure #{f['id']} Event" if f['id'] == 1 else "")
        ax.axvline(f_start, color="#b30000", linestyle=":", lw=1.5)
        ax.text(f_start, ax.get_ylim()[1] * 0.85, f" Fail #{f['id']}", rotation=90, color="#800000", fontweight="bold", fontsize=9)

    ax.set_title("Full Stream Phase-1 Anomaly Score with Documented Failure Episodes", fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sequence MSE")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[visualizer] Saved Plot 1 to: {save_path}")
    plt.close(fig)


def plot_2_and_3_anomaly_around_failure(
    timestamps: np.ndarray,
    scores: np.ndarray,
    failure_id: int,
    phase1_threshold: float,
    failures: List[Dict] = KNOWN_FAILURES,
    lookback_days: float = 3.0,
    lookahead_days: float = 1.0,
    save_path: Optional[str] = None,
):
    """Plot 2 & 3: Phase-1 anomaly score zoomed around Failure #3 or Failure #4."""
    target_f = [f for f in failures if f["id"] == failure_id][0]
    f_start = pd.Timestamp(target_f["start"])
    f_end = pd.Timestamp(target_f["end"])

    t_min = f_start - pd.Timedelta(days=lookback_days)
    t_max = f_end + pd.Timedelta(days=lookahead_days)

    ts = pd.to_datetime(timestamps)
    mask = (ts >= t_min) & (ts <= t_max)
    sub_ts = ts[mask]
    sub_scores = scores[mask]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(sub_ts, sub_scores, color="#1f77b4", lw=1.5, label="Phase 1 Anomaly Score")
    ax.axhline(phase1_threshold, color="#d9534f", linestyle="--", lw=1.8, label=f"Anomaly Threshold ({phase1_threshold:.2f})")
    ax.axvspan(f_start, f_end, color="#ff4d4d", alpha=0.35, label=f"Failure #{failure_id} ({target_f['type']})")

    # 24h warning horizon boundary
    h24_start = f_start - pd.Timedelta(hours=24)
    ax.axvspan(h24_start, f_start, color="#ffa726", alpha=0.2, label="24h Impending Horizon")

    ax.set_title(f"Phase-1 Anomaly Progression Around Failure #{failure_id} ({f_start.strftime('%Y-%m-%d %H:%M')})", fontweight="bold", pad=12)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Sequence MSE")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[visualizer] Saved Plot (Failure #{failure_id}) to: {save_path}")
    plt.close(fig)


def plot_4_dominant_sensor_errors(
    timestamps: np.ndarray,
    sensor_errors: np.ndarray,
    failures: List[Dict] = KNOWN_FAILURES,
    sensor_names: List[str] = ANALOGUE_SENSORS,
    save_path: Optional[str] = None,
):
    """Plot 4: Sensor-level reconstruction errors around test failures (#3 & #4)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    test_fails = [f for f in failures if f["id"] in [3, 4]]
    ts = pd.to_datetime(timestamps)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    for ax_idx, f in enumerate(test_fails):
        ax = axes[ax_idx]
        f_start = pd.Timestamp(f["start"])
        f_end = pd.Timestamp(f["end"])
        t_min = f_start - pd.Timedelta(days=2.5)
        t_max = f_end + pd.Timedelta(hours=12)

        mask = (ts >= t_min) & (ts <= t_max)
        sub_ts = ts[mask]
        sub_err = sensor_errors[mask]

        for s_idx, s_name in enumerate(sensor_names):
            ax.plot(sub_ts, sub_err[:, s_idx], lw=1.5, color=colors[s_idx % len(colors)], label=s_name)

        ax.axvspan(f_start, f_end, color="#ff4d4d", alpha=0.3, label="Failure Period")
        ax.set_title(f"Sensor Error Decomposition around Failure #{f['id']} ({f_start.strftime('%Y-%m-%d')})", fontweight="bold")
        ax.set_ylabel("Sensor Reconstruction MSE")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.legend(loc="upper left", ncol=4, fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[visualizer] Saved Plot 4 to: {save_path}")
    plt.close(fig)


def plot_5_temporal_degradation_trend(
    timestamps: np.ndarray,
    features_df: pd.DataFrame,
    failures: List[Dict] = KNOWN_FAILURES,
    save_path: Optional[str] = None,
):
    """Plot 5: Temporal degradation trends (short, med, long slopes, acceleration) before test failures."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    test_fails = [f for f in failures if f["id"] in [3, 4]]
    ts = pd.to_datetime(timestamps)

    for i, f in enumerate(test_fails):
        f_start = pd.Timestamp(f["start"])
        t_min = f_start - pd.Timedelta(days=2.0)
        t_max = f_start

        mask = (ts >= t_min) & (ts <= t_max)
        sub_ts = ts[mask]
        sub_feat = features_df.loc[mask]

        # Row i, Col 0: Anomaly Slopes
        ax_slope = axes[i, 0]
        ax_slope.plot(sub_ts, sub_feat["anomaly_slope_short"], label="Slope (Short ~30m)", color="#1f77b4", lw=1.5)
        ax_slope.plot(sub_ts, sub_feat["anomaly_slope_medium"], label="Slope (Med ~2h)", color="#ff7f0e", lw=1.5)
        ax_slope.plot(sub_ts, sub_feat["anomaly_slope_long"], label="Slope (Long ~6h)", color="#2ca02c", lw=1.5)
        ax_slope.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax_slope.set_title(f"Failure #{f['id']}: Pre-Failure Anomaly Slopes", fontweight="bold")
        ax_slope.set_ylabel("Slope (MSE / step)")
        ax_slope.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax_slope.legend(loc="upper left", fontsize=9)

        # Row i, Col 1: Acceleration & Persistence
        ax_accel = axes[i, 1]
        ax_accel.plot(sub_ts, sub_feat["anomaly_acceleration"], label="Acceleration (Δ Slope)", color="#d62728", lw=1.5)
        ax_accel.plot(sub_ts, sub_feat["fraction_above_threshold_recent"], label="Persistence (Frac > Threshold)", color="#9467bd", lw=1.5)
        ax_accel.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax_accel.set_title(f"Failure #{f['id']}: Anomaly Acceleration & Persistence", fontweight="bold")
        ax_accel.set_ylabel("Metric Value")
        ax_accel.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax_accel.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[visualizer] Saved Plot 5 to: {save_path}")
    plt.close(fig)


def plot_6_risk_score_timeline(
    timestamps: np.ndarray,
    risk_probs: np.ndarray,
    frozen_threshold: float,
    failures: List[Dict] = KNOWN_FAILURES,
    save_path: Optional[str] = None,
):
    """Plot 6: Phase 2B failure risk score/probability timeline around test failures with alarm threshold."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    test_fails = [f for f in failures if f["id"] in [3, 4]]
    ts = pd.to_datetime(timestamps)

    for ax_idx, f in enumerate(test_fails):
        ax = axes[ax_idx]
        f_start = pd.Timestamp(f["start"])
        f_end = pd.Timestamp(f["end"])
        t_min = f_start - pd.Timedelta(days=3.0)
        t_max = f_end + pd.Timedelta(hours=12)

        mask = (ts >= t_min) & (ts <= t_max)
        sub_ts = ts[mask]
        sub_probs = risk_probs[mask]

        ax.plot(sub_ts, sub_probs, color="#800080", lw=2.0, label="Phase 2B Failure Risk Probability")
        ax.axhline(frozen_threshold, color="#d9534f", linestyle="--", lw=1.8, label=f"Frozen Operating Threshold ({frozen_threshold:.2f})")
        ax.axvspan(f_start - pd.Timedelta(hours=24), f_start, color="#ffa726", alpha=0.25, label="24h Impending Failure Horizon")
        ax.axvspan(f_start, f_end, color="#ff4d4d", alpha=0.35, label=f"Failure #{f['id']} Incident")

        ax.set_title(f"Phase 2B Failure Risk Probability Timeline — Failure #{f['id']}", fontweight="bold")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Predicted Risk Probability $P(y=1)$")
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[visualizer] Saved Plot 6 to: {save_path}")
    plt.close(fig)
