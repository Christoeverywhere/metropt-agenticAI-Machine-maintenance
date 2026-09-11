"""
Comprehensive Visualization Module for MetroPT-3 Anomaly Detection.
Generates publication-quality figures saved to reports/figures/.
"""
import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.config import ANALOGUE_SENSORS, KNOWN_FAILURES
except ImportError:
    from config import ANALOGUE_SENSORS, KNOWN_FAILURES

# Modern presentation color palette
PALETTE = ["#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED", "#DB2777"]
sns.set_theme(style="whitegrid", font_scale=1.05)


def plot_loss_curves(
    all_histories: Dict[str, Dict[str, List[float]]],
    save_path: str,
):
    """Plots training and validation loss curves for all models."""
    n_models = len(all_histories)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for idx, (m_name, hist) in enumerate(all_histories.items()):
        ax = axes[idx]
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], label="Train MSE", color="#2563EB", lw=2, marker="o", ms=4)
        ax.plot(epochs, hist["val_loss"], label="Val MSE", color="#DC2626", lw=2, marker="s", ms=4)
        ax.set_title(m_name, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=11)
        if idx == 0:
            ax.set_ylabel("MSE Loss (log scale)", fontsize=11)
        ax.set_yscale("log")
        ax.legend(frameon=True, facecolor="white", loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Training & Validation Loss Progression across Autoencoder Architectures", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved loss curves to: {save_path}")


def plot_error_timeline(
    timestamps: np.ndarray,
    seq_errors_dict: Dict[str, np.ndarray],
    thresholds_dict: Dict[str, float],
    failures: List[Dict] = KNOWN_FAILURES,
    save_path: str = "reports/figures/reconstruction_error_timeline.png",
):
    """Plots test stream reconstruction error over time with threshold lines and shaded failure intervals."""
    ts = pd.to_datetime(timestamps)
    n_models = len(seq_errors_dict)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 3.2 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    for idx, (m_name, errs) in enumerate(seq_errors_dict.items()):
        ax = axes[idx]
        thresh = thresholds_dict.get(m_name, 1.0)

        # Plot rolling smoothed error to keep chart readable
        err_series = pd.Series(errs, index=ts).rolling("10min", min_periods=1).mean()

        ax.plot(ts, err_series.values, label="Reconstruction Error (10m rolling)", color=PALETTE[idx % len(PALETTE)], lw=1.2)
        ax.axhline(thresh, color="#DC2626", linestyle="--", lw=1.8, label=f"Threshold ({thresh:.3f})")

        # Shade failure windows
        for f in failures:
            f_start = pd.Timestamp(f["start"])
            f_end = pd.Timestamp(f["end"])
            ax.axvspan(f_start, f_end, color="#FCA5A5", alpha=0.45, label="Failure Event" if f["id"] == 1 and idx == 0 else "")

        ax.set_title(f"{m_name} — Test Stream Reconstruction Error", fontsize=12, fontweight="bold")
        ax.set_ylabel("Reconstruction MSE", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, facecolor="white")

    axes[-1].set_xlabel("Timestamp", fontsize=11)
    plt.suptitle("MetroPT-3 Test Stream Reconstruction Anomaly Timeline", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved error timeline to: {save_path}")


def plot_error_distribution(
    normal_errors: np.ndarray,
    anomaly_errors: np.ndarray,
    threshold: float,
    model_name: str,
    save_path: str,
):
    """Plots histogram & KDE comparing normal vs anomaly reconstruction error distributions."""
    plt.figure(figsize=(9, 5))

    sns.histplot(normal_errors, color="#2563EB", label="Normal Operation", kde=True, stat="density", bins=50, alpha=0.5)
    if len(anomaly_errors) > 0:
        sns.histplot(anomaly_errors, color="#DC2626", label="Anomaly / Failure Periods", kde=True, stat="density", bins=50, alpha=0.5)

    plt.axvline(threshold, color="#111827", linestyle="--", lw=2, label=f"Calibrated Threshold ({threshold:.4f})")
    plt.title(f"Reconstruction Error Distribution: Normal vs Anomaly ({model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Reconstruction Error (MSE)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(frameon=True, facecolor="white", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved error distribution to: {save_path}")


def plot_confusion_matrices(
    all_metrics: Dict[str, Dict[str, Any]],
    save_path: str,
):
    """Plots a multi-panel grid of confusion matrices across all evaluated models."""
    n_models = len(all_metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(3.8 * n_models, 3.8))
    if n_models == 1:
        axes = [axes]

    for idx, (m_name, metrics) in enumerate(all_metrics.items()):
        ax = axes[idx]
        cm_dict = metrics["confusion_matrix"]
        cm_array = np.array([
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]],
        ])

        sns.heatmap(
            cm_array,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred Normal", "Pred Anomaly"],
            yticklabels=["True Normal", "True Anomaly"],
            annot_kws={"size": 11, "fontweight": "bold"},
        )
        f1_val = metrics["f1_score"]
        ax.set_title(f"{m_name}\nF1: {f1_val:.3f} | Rec: {metrics['recall']:.3f}", fontsize=11, fontweight="bold", pad=8)

    plt.suptitle("Confusion Matrix Comparison Across Autoencoder Variants", fontsize=14, fontweight="bold", y=1.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved confusion matrices to: {save_path}")


def plot_sensor_explainability(
    sensor_contributions: Dict[str, Any],
    save_path: str,
):
    """Plots sensor-wise anomaly contribution breakdown and error amplification ratios."""
    breakdown = sensor_contributions["sensor_breakdown"]
    sensors = list(breakdown.keys())
    normal_errs = [breakdown[s]["mean_normal_error"] for s in sensors]
    anomaly_errs = [breakdown[s]["mean_anomaly_error"] for s in sensors]
    pcts = [breakdown[s]["contribution_pct"] for s in sensors]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Bar chart: Normal vs Anomaly MSE
    x = np.arange(len(sensors))
    width = 0.35
    ax1.bar(x - width/2, normal_errs, width, label="Normal Operation", color="#2563EB", alpha=0.85)
    ax1.bar(x + width/2, anomaly_errs, width, label="Failure Anomaly", color="#DC2626", alpha=0.85)
    ax1.set_title("Per-Sensor Reconstruction Error (Normal vs Failure)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sensors, rotation=35, ha="right", fontsize=10)
    ax1.set_ylabel("Mean Reconstruction MSE", fontsize=11)
    ax1.legend(frameon=True, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Pie/Horizontal Bar: Percentage contribution to failure anomaly
    colors = sns.color_palette("rocket", len(sensors))
    y_pos = np.arange(len(sensors))
    sorted_indices = np.argsort(pcts)

    ax2.barh(
        [sensors[i] for i in sorted_indices],
        [pcts[i] for i in sorted_indices],
        color=[colors[i] for i in sorted_indices],
    )
    for i, idx_s in enumerate(sorted_indices):
        val = pcts[idx_s]
        ax2.text(val + 0.8, i, f"{val:.1f}%", va="center", fontweight="bold", fontsize=10)

    ax2.set_title("Sensor Root-Cause Contribution to Anomaly Error", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Contribution Percentage (%)", fontsize=11)
    ax2.set_xlim(0, max(pcts) + 8)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Sensor-Level Anomaly Explainability & Attribution Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved sensor explainability chart to: {save_path}")


def plot_benchmark_comparison(
    all_metrics: Dict[str, Dict[str, Any]],
    save_path: str,
):
    """Plots comparative bar charts across Precision, Recall, F1, ROC-AUC, PR-AUC."""
    models = list(all_metrics.keys())
    metric_keys = ["precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    metric_labels = ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]

    data = {label: [all_metrics[m][k] for m in models] for k, label in zip(metric_keys, metric_labels)}
    df = pd.DataFrame(data, index=models)

    ax = df.plot(kind="bar", figsize=(13, 5.8), colormap="viridis", width=0.8, edgecolor="black", linewidth=0.5)
    plt.title("Experimental Comparison of 5 Autoencoder Architectures on MetroPT-3", fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("Metric Score [0.0 - 1.0]", fontsize=11)
    plt.xlabel("Model Architecture", fontsize=11)
    plt.xticks(rotation=15, ha="right", fontsize=10, fontweight="bold")
    plt.ylim(0, 1.15)
    plt.legend(loc="upper right", frameon=True, facecolor="white", ncol=5)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Add numeric labels above bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8, rotation=45)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved benchmark comparison chart to: {save_path}")


def plot_normal_vs_anomalous_reconstruction(
    normal_window: np.ndarray,
    normal_recon: np.ndarray,
    anomaly_window: np.ndarray,
    anomaly_recon: np.ndarray,
    sensor_names: List[str] = ANALOGUE_SENSORS,
    save_path: str = "reports/figures/normal_vs_anomalous_reconstruction.png",
):
    """Plots a 7-sensor side-by-side comparison of Actual vs Reconstructed signals for
    a Normal healthy window (left) vs an Anomalous failure window (right)."""
    n_sensors = len(sensor_names)
    fig, axes = plt.subplots(n_sensors, 2, figsize=(15, 2.2 * n_sensors), sharex=True)

    time_steps = np.arange(normal_window.shape[0])

    for i, s_name in enumerate(sensor_names):
        # Left: Normal Window
        ax_l = axes[i, 0]
        ax_l.plot(time_steps, normal_window[:, i], color="#2563EB", lw=1.8, label="Actual Sensor Signal")
        ax_l.plot(time_steps, normal_recon[:, i], color="#059669", linestyle="--", lw=1.6, label="Reconstructed Signal")
        ax_l.set_ylabel(f"{s_name}\n(scaled)", fontsize=10, fontweight="bold")
        ax_l.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax_l.set_title("Normal Operational Window (Healthy Baseline)", fontsize=12, fontweight="bold", pad=8)
            ax_l.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)

        # Right: Anomaly Window
        ax_r = axes[i, 1]
        ax_r.plot(time_steps, anomaly_window[:, i], color="#DC2626", lw=1.8, label="Actual Sensor Signal")
        ax_r.plot(time_steps, anomaly_recon[:, i], color="#D97706", linestyle="--", lw=1.6, label="Reconstructed Signal")
        ax_r.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax_r.set_title("Anomalous Pre-Failure Window (Air Leak Event)", fontsize=12, fontweight="bold", pad=8)
            ax_r.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)

    axes[-1, 0].set_xlabel("Timestep within Window (t = 0 to 180)", fontsize=11)
    axes[-1, 1].set_xlabel("Timestep within Window (t = 0 to 180)", fontsize=11)

    plt.suptitle("Sensor Signal Reconstruction Fidelity: Normal vs Anomalous Failure Window", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved normal vs anomaly reconstruction comparison to: {save_path}")


def plot_anomaly_episodes_timeline(
    timestamps: np.ndarray,
    seq_errors: np.ndarray,
    threshold: float,
    episodes: List[Dict[str, Any]],
    failures: List[Dict[str, Any]] = KNOWN_FAILURES,
    save_path: str = "reports/figures/anomaly_episodes_timeline.png",
):
    """Plots detected anomaly episodes clustered along the timeline with lead-time indicators and known failure bands."""
    ts = pd.to_datetime(timestamps)
    err_series = pd.Series(seq_errors, index=ts).rolling("15min", min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(ts, err_series.values, color="#2563EB", lw=1.2, label="Reconstruction MSE (15m rolling)")
    ax.axhline(threshold, color="#DC2626", linestyle="--", lw=1.8, label=f"Calibrated Threshold ({threshold:.3f})")

    # Shade known failure windows
    for idx_f, f in enumerate(failures, 1):
        f_start = pd.Timestamp(f["start"])
        f_end = pd.Timestamp(f["end"])
        ax.axvspan(f_start, f_end, color="#F87171", alpha=0.45, label="Documented Failure" if idx_f == 1 else "")
        ax.text(f_start, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 5.0, f"Fail #{f['id']}", rotation=90, va="top", fontsize=9, fontweight="bold", color="#991B1B")

    # Mark detected episodes
    for ep in episodes:
        ep_start = pd.Timestamp(ep["start_time"])
        ep_end = pd.Timestamp(ep["end_time"])
        if ep["lead_time_hours"] is not None and ep["lead_time_hours"] > 0:
            ax.axvspan(ep_start, ep_end, color="#FBBF24", alpha=0.6, label="Detected Anomaly Episode (Pre-Failure)" if "Detected Anomaly Episode" not in [l.get_label() for l in ax.lines + ax.patches] else "")

    ax.set_title("Detected Anomaly Episodes, Lead Times, and Documented Compressor Failure Intervals", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date / Time", fontsize=11)
    ax.set_ylabel("Reconstruction Error (MSE)", fontsize=11)
    ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved anomaly episodes timeline to: {save_path}")

