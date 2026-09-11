"""
Phase 2 Research Visualizer: Generates Publication-Quality Figures Saved to reports/figures/.
"""
import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)

try:
    from src.config import KNOWN_FAILURES, FIGURES_DIR
except ImportError:
    from config import KNOWN_FAILURES, FIGURES_DIR

PALETTE = ["#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED", "#DB2777"]
sns.set_theme(style="whitegrid", font_scale=1.05)


def plot_phase2_probability_timeline(
    timestamps: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    thresholds_dict: Dict[str, float],
    failures: List[Dict] = KNOWN_FAILURES,
    save_path: str = "reports/figures/phase2_probability_timeline.png",
):
    """Plots continuous failure probability timeline vs threshold with shaded failure intervals."""
    ts = pd.to_datetime(timestamps)
    n_models = len(probs_dict)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 3.4 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    for idx, (m_name, probs) in enumerate(probs_dict.items()):
        ax = axes[idx]
        thresh = thresholds_dict.get(m_name, 0.5)

        prob_series = pd.Series(probs, index=ts).rolling("15min", min_periods=1).mean()

        ax.plot(ts, prob_series.values, color=PALETTE[idx % len(PALETTE)], lw=1.5, label=f"Predicted P(Failure in 24h) [15m roll]")
        ax.axhline(thresh, color="#DC2626", linestyle="--", lw=1.8, label=f"Calibrated Threshold ({thresh:.2f})")

        # Shade failure windows
        for f in failures:
            f_start = pd.Timestamp(f["start"])
            f_end = pd.Timestamp(f["end"])
            ax.axvspan(f_start, f_end, color="#FCA5A5", alpha=0.45, label="Documented Failure" if f["id"] == 1 and idx == 0 else "")

        ax.set_title(f"{m_name} — Impending Failure Probability Timeline (24-Hour Horizon)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Failure Probability", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, facecolor="white")

    axes[-1].set_xlabel("Timestamp", fontsize=11)
    plt.suptitle("MetroPT-3 Test Stream Failure Probability Predictions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved failure probability timeline to: {save_path}")


def plot_phase2_lead_times(
    all_lead_times: Dict[str, List[Dict[str, Any]]],
    save_path: str = "reports/figures/phase2_lead_times.png",
):
    """Bar chart comparing predictive warning lead times per failure event across models."""
    models = list(all_lead_times.keys())
    failure_ids = [f"Failure #{f['failure_id']}" for f in all_lead_times[models[0]]]

    data = {m: [f["lead_time_hours"] if f["detected"] else 0.0 for f in all_lead_times[m]] for m in models}
    df = pd.DataFrame(data, index=failure_ids)

    ax = df.plot(kind="bar", figsize=(10, 5.5), colormap="crest", width=0.7, edgecolor="black", linewidth=0.5)
    plt.title("Predictive Warning Lead Time Comparison Across Models", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Predictive Lead Time (Hours in Advance)", fontsize=11)
    plt.xlabel("Documented Compressor Failure Event", fontsize=11)
    plt.xticks(rotation=0, fontsize=10, fontweight="bold")
    plt.legend(frameon=True, facecolor="white")
    plt.grid(True, linestyle="--", alpha=0.6)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f h", padding=3, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved predictive lead times chart to: {save_path}")


def plot_phase2_model_comparison(
    all_metrics: Dict[str, Dict[str, Any]],
    save_path: str = "reports/figures/phase2_model_comparison.png",
):
    """Bar chart comparing Logistic Regression, LSTM, and GRU across primary metrics."""
    models = list(all_metrics.keys())
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "pr_auc", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "PR-AUC", "ROC-AUC"]

    data = {label: [all_metrics[m][k] for m in models] for k, label in zip(metric_keys, metric_labels)}
    df = pd.DataFrame(data, index=models)

    ax = df.plot(kind="bar", figsize=(12, 5.5), colormap="viridis", width=0.75, edgecolor="black", linewidth=0.5)
    plt.title("Supervised Failure Predictor Benchmark: Baseline vs LSTM vs GRU (24h Horizon)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Score [0.0 - 1.0]", fontsize=11)
    plt.xlabel("Predictive Model Architecture", fontsize=11)
    plt.xticks(rotation=0, fontsize=10, fontweight="bold")
    plt.ylim(0, 1.15)
    plt.legend(loc="upper right", frameon=True, facecolor="white", ncol=6)
    plt.grid(True, linestyle="--", alpha=0.6)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved Phase 2 model comparison to: {save_path}")


def plot_phase2_horizon_comparison(
    horizon_results: List[Dict[str, Any]],
    save_path: str = "reports/figures/phase2_horizon_comparison.png",
):
    """Line plots showing metric trends across 6h, 12h, 24h, and 48h horizons."""
    df = pd.DataFrame(horizon_results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(df["horizon_hours"], df["f1_score"], marker="o", lw=2, color="#2563EB", label="F1-Score")
    ax1.plot(df["horizon_hours"], df["pr_auc"], marker="s", lw=2, color="#059669", label="PR-AUC")
    ax1.plot(df["horizon_hours"], df["recall"], marker="^", lw=2, color="#D97706", label="Recall")
    ax1.set_title("Prediction Performance vs. Warning Horizon", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Prediction Horizon (Hours in Advance)", fontsize=11)
    ax1.set_ylabel("Metric Score", fontsize=11)
    ax1.set_xticks(df["horizon_hours"])
    ax1.legend(frameon=True, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Lead times vs horizon
    ax2.plot(df["horizon_hours"], df["mean_lead_time_hours"], marker="d", lw=2.5, color="#7C3AED", label="Mean Detection Lead Time")
    ax2.set_title("Achieved Lead Time vs. Configured Horizon", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Prediction Horizon (Hours in Advance)", fontsize=11)
    ax2.set_ylabel("Mean Lead Time (Hours)", fontsize=11)
    ax2.set_xticks(df["horizon_hours"])
    ax2.legend(frameon=True, facecolor="white")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Prediction Horizon Sensitivity & Trade-Off Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved horizon comparison to: {save_path}")


def plot_phase2_integration_comparison(
    integration_results: List[Dict[str, Any]],
    save_path: str = "reports/figures/phase2_integration_comparison.png",
):
    """Bar chart comparing Exp A (Sensors-only) vs Exp B (Sensors+Score) vs Exp C (Sensors+Score+Errors)."""
    df = pd.DataFrame(integration_results)
    labels = df["experiment_name"].tolist()

    metric_keys = ["f1_score", "pr_auc", "roc_auc", "recall"]
    metric_labels = ["F1-Score", "PR-AUC", "ROC-AUC", "Recall"]

    data = {label: df[k].tolist() for k, label in zip(metric_keys, metric_labels)}
    plot_df = pd.DataFrame(data, index=labels)

    ax = plot_df.plot(kind="bar", figsize=(11, 5.2), colormap="plasma", width=0.7, edgecolor="black", linewidth=0.5)
    plt.title("Phase 1 Feature Integration Ablation (GRU Predictor, 24h Horizon)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Metric Score", fontsize=11)
    plt.xlabel("Feature Representation Strategy", fontsize=11)
    plt.xticks(rotation=0, fontsize=10, fontweight="bold")
    plt.ylim(0, 1.15)
    plt.legend(loc="upper right", frameon=True, facecolor="white", ncol=4)
    plt.grid(True, linestyle="--", alpha=0.6)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved Phase 1 integration comparison to: {save_path}")


def plot_phase2_confusion_matrices(
    all_metrics: Dict[str, Dict[str, Any]],
    save_path: str = "reports/figures/phase2_confusion_matrices.png",
):
    """Plots confusion matrices across Phase 2 models."""
    n_models = len(all_metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.0))
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
            cmap="Greens",
            cbar=False,
            ax=ax,
            xticklabels=["Pred Normal", "Pred Failure Risk"],
            yticklabels=["True Normal", "True Failure Risk"],
            annot_kws={"size": 11, "fontweight": "bold"},
        )
        f1_val = metrics["f1_score"]
        ax.set_title(f"{m_name}\nF1: {f1_val:.3f} | Rec: {metrics['recall']:.3f}", fontsize=11, fontweight="bold", pad=8)

    plt.suptitle("Confusion Matrices for Impending Failure Prediction (24h Horizon)", fontsize=14, fontweight="bold", y=1.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved Phase 2 confusion matrices to: {save_path}")


def plot_phase2_pr_roc_curves(
    y_test: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    save_path: str = "reports/figures/phase2_pr_roc_curves.png",
):
    """Plots Precision-Recall and ROC Curves across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for idx, (m_name, probs) in enumerate(probs_dict.items()):
        color = PALETTE[idx % len(PALETTE)]
        # PR Curve
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        ax1.plot(rec, prec, lw=2, color=color, label=f"{m_name} (PR-AUC = {pr_auc:.3f})")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)
        ax2.plot(fpr, tpr, lw=2, color=color, label=f"{m_name} (ROC-AUC = {roc_auc:.3f})")

    ax1.set_title("Precision-Recall Curves (24h Horizon)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Recall", fontsize=11)
    ax1.set_ylabel("Precision", fontsize=11)
    ax1.legend(frameon=True, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5, label="Random Guess")
    ax2.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax2.legend(frameon=True, facecolor="white")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Classification Trade-Off Analysis Across Predictive Architectures", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved PR & ROC curves to: {save_path}")


def plot_phase2_risk_assessment_demo(
    norm_result: Dict[str, Any],
    anom_result: Dict[str, Any],
    save_path: str = "reports/figures/phase2_risk_assessment_demo.png",
):
    """Visualizes combined Phase 1 + Phase 2 assessment output comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Normal Assessment
    norm_sensors = list(norm_result["anomaly"]["sensor_contributions"].keys())
    norm_pcts = list(norm_result["anomaly"]["sensor_contributions"].values())
    ax1.barh(norm_sensors, norm_pcts, color="#2563EB", alpha=0.85)
    ax1.set_title(
        f"Normal Operation State\n"
        f"Anomaly Score: {norm_result['anomaly']['anomaly_score']:.3f} (Thresh: {norm_result['anomaly']['threshold']:.3f})\n"
        f"Failure Prob: {norm_result['failure_prediction']['failure_probability']*100:.1f}% | Risk: {norm_result['failure_prediction']['risk_level']}",
        fontsize=11, fontweight="bold", pad=10
    )
    ax1.set_xlabel("Sensor Contribution (%)", fontsize=10)
    ax1.set_xlim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Right: Pre-Failure Assessment
    anom_sensors = list(anom_result["anomaly"]["sensor_contributions"].keys())
    anom_pcts = list(anom_result["anomaly"]["sensor_contributions"].values())
    ax2.barh(anom_sensors, anom_pcts, color="#DC2626", alpha=0.85)
    ax2.set_title(
        f"Pre-Failure Degradation State (24h Warning)\n"
        f"Anomaly Score: {anom_result['anomaly']['anomaly_score']:.3f} (Thresh: {anom_result['anomaly']['threshold']:.3f})\n"
        f"Failure Prob: {anom_result['failure_prediction']['failure_probability']*100:.1f}% | Risk: {anom_result['failure_prediction']['risk_level']}",
        fontsize=11, fontweight="bold", pad=10
    )
    ax2.set_xlabel("Sensor Contribution (%)", fontsize=10)
    ax2.set_xlim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Combined Phase 1 Anomaly Attribution + Phase 2 Impending Failure Assessment", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Saved risk assessment demo to: {save_path}")
