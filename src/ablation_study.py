"""
Systematic Hyperparameter Tuning and Ablation Studies:
  1. Sequence Length (T = 60, 120, 180)
  2. Latent Bottleneck Size (8, 16, 32)
  3. Denoising Noise Level (sigma = 0.01, 0.05, 0.10)
  4. Anomaly Threshold Strategies (P95, P97.5, P99, P99.5, mu+3sig, mu+4sig, IQR)
"""
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.config import N_FEATURES, INITIAL_LR, KNOWN_FAILURES
    from src.preprocessing import (
        load_and_clean_data,
        create_temporal_splits,
        fit_scaler,
        transform_data,
        generate_ground_truth_labels,
    )
    from src.dataset import build_sliding_windows
    from src.models.attention_lstm_ae import AttentionDenoisingLSTMAutoencoder
    from src.models.lstm_ae import LSTMAutoencoder
    from src.train_engine import train_model
    from src.thresholds import calibrate_thresholds_suite
    from src.evaluator import compute_reconstruction_errors, evaluate_predictions
except ImportError:
    from config import N_FEATURES, INITIAL_LR, KNOWN_FAILURES
    from preprocessing import (
        load_and_clean_data,
        create_temporal_splits,
        fit_scaler,
        transform_data,
        generate_ground_truth_labels,
    )
    from dataset import build_sliding_windows
    from models.attention_lstm_ae import AttentionDenoisingLSTMAutoencoder
    from models.lstm_ae import LSTMAutoencoder
    from train_engine import train_model
    from thresholds import calibrate_thresholds_suite
    from evaluator import compute_reconstruction_errors, evaluate_predictions


REPORTS_DIR = os.path.join(_project_root, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_ablation_study(epochs: int = 5, batch_size: int = 256):
    print("=" * 95)
    print("                     METROPT-3 ABLATION & HYPERPARAMETER TUNING")
    print("=" * 95)

    raw_df = load_and_clean_data()
    healthy_train, healthy_val, test_stream = create_temporal_splits(raw_df, val_days=5)

    scaler = fit_scaler(healthy_train)
    scaled_train = transform_data(healthy_train, scaler)
    scaled_val = transform_data(healthy_val, scaler)
    scaled_test = transform_data(test_stream, scaler)

    test_ground_truth = generate_ground_truth_labels(test_stream)

    # -------------------------------------------------------------
    # 1. Ablation: Sequence Length T in [60, 120, 180]
    # -------------------------------------------------------------
    print("\n[Ablation 1/4] Evaluating Sequence Lengths (T = 60, 120, 180)...")
    seq_len_results = []
    for seq_len in [60, 120, 180]:
        train_win = build_sliding_windows(scaled_train, seq_len=seq_len, stride=35)
        val_win = build_sliding_windows(scaled_val, seq_len=seq_len, stride=25)
        test_win = build_sliding_windows(scaled_test, seq_len=seq_len, stride=25)

        # Label windows
        test_lbls = np.zeros(len(test_win), dtype=np.int32)
        for idx in range(len(test_win)):
            s_i = idx * 25
            test_lbls[idx] = int(np.any(test_ground_truth[s_i : s_i + seq_len]))

        model = AttentionDenoisingLSTMAutoencoder(seq_len=seq_len, latent_size=16)
        trained_m, _ = train_model(
            model, train_win, val_win, model_name=f"Attn-LSTM (T={seq_len})",
            epochs=epochs, lr=INITIAL_LR, batch_size=batch_size, noise_std=0.05, mask_prob=0.05
        )

        val_errs, _ = compute_reconstruction_errors(trained_m, val_win)
        thresh = float(np.percentile(val_errs, 99.0))

        test_errs, _ = compute_reconstruction_errors(trained_m, test_win)
        m = evaluate_predictions(test_lbls, test_errs, threshold=thresh)

        seq_len_results.append({
            "sequence_length": seq_len,
            "val_recon_mse": float(np.mean(val_errs)),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
        })

    # -------------------------------------------------------------
    # 2. Ablation: Latent Bottleneck Dimension in [8, 16, 32]
    # -------------------------------------------------------------
    print("\n[Ablation 2/4] Evaluating Latent Dimensions (z = 8, 16, 32)...")
    latent_results = []
    train_win_180 = build_sliding_windows(scaled_train, seq_len=180, stride=35)
    val_win_180 = build_sliding_windows(scaled_val, seq_len=180, stride=25)
    test_win_180 = build_sliding_windows(scaled_test, seq_len=180, stride=25)

    test_lbls_180 = np.zeros(len(test_win_180), dtype=np.int32)
    for idx in range(len(test_win_180)):
        s_i = idx * 25
        test_lbls_180[idx] = int(np.any(test_ground_truth[s_i : s_i + 180]))

    for latent_dim in [8, 16, 32]:
        model = AttentionDenoisingLSTMAutoencoder(seq_len=180, latent_size=latent_dim)
        trained_m, _ = train_model(
            model, train_win_180, val_win_180, model_name=f"Attn-LSTM (Latent={latent_dim})",
            epochs=epochs, lr=INITIAL_LR, batch_size=batch_size, noise_std=0.05, mask_prob=0.05
        )

        val_errs, _ = compute_reconstruction_errors(trained_m, val_win_180)
        thresh = float(np.percentile(val_errs, 99.0))

        test_errs, _ = compute_reconstruction_errors(trained_m, test_win_180)
        m = evaluate_predictions(test_lbls_180, test_errs, threshold=thresh)

        latent_results.append({
            "latent_dim": latent_dim,
            "val_recon_mse": float(np.mean(val_errs)),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
        })

    # -------------------------------------------------------------
    # 3. Ablation: Denoising Noise Level sigma in [0.00, 0.02, 0.05, 0.10]
    # -------------------------------------------------------------
    print("\n[Ablation 3/4] Evaluating Noise Level (sigma = 0.00, 0.02, 0.05, 0.10)...")
    noise_results = []
    for noise_std in [0.00, 0.02, 0.05, 0.10]:
        model = AttentionDenoisingLSTMAutoencoder(seq_len=180, latent_size=16)
        trained_m, _ = train_model(
            model, train_win_180, val_win_180, model_name=f"Attn-LSTM (sigma={noise_std})",
            epochs=epochs, lr=INITIAL_LR, batch_size=batch_size, noise_std=noise_std, mask_prob=0.05 if noise_std > 0 else 0.0
        )

        val_errs, _ = compute_reconstruction_errors(trained_m, val_win_180)
        thresh = float(np.percentile(val_errs, 99.0))

        test_errs, _ = compute_reconstruction_errors(trained_m, test_win_180)
        m = evaluate_predictions(test_lbls_180, test_errs, threshold=thresh)

        noise_results.append({
            "noise_std": noise_std,
            "val_recon_mse": float(np.mean(val_errs)),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
        })

    # -------------------------------------------------------------
    # 4. Threshold Calibration Strategy Comparison
    # -------------------------------------------------------------
    print("\n[Ablation 4/4] Comparing Anomaly Threshold Calibration Strategies...")
    val_errs, _ = compute_reconstruction_errors(trained_m, val_win_180)
    thresh_suite = calibrate_thresholds_suite(val_errs)
    threshold_comp = []

    for t_name, t_val in thresh_suite.items():
        m = evaluate_predictions(test_lbls_180, test_errs, threshold=t_val)
        threshold_comp.append({
            "strategy": t_name,
            "threshold_value": round(t_val, 6),
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1_score": round(m["f1_score"], 4),
            "specificity": round(m["specificity"], 4),
            "fpr": round(m["fpr"], 4),
            "fnr": round(m["fnr"], 4),
        })

    # Print Threshold Calibration Comparison Table
    print("\n" + "=" * 105)
    print("                    THRESHOLD CALIBRATION STRATEGY COMPARISON")
    print("=" * 105)
    print(f"{'Strategy':<15} | {'Threshold':<12} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Specificity':<11} | {'FPR':<7}")
    print("-" * 105)
    for tc in threshold_comp:
        print(
            f"{tc['strategy']:<15} | {tc['threshold_value']:<12.6f} | "
            f"{tc['accuracy']:<9.4f} | {tc['precision']:<9.4f} | "
            f"{tc['recall']:<8.4f} | {tc['f1_score']:<8.4f} | "
            f"{tc['specificity']:<11.4f} | {tc['fpr']:<7.4f}"
        )
    print("=" * 105)

    # Save Ablation Results
    ablation_payload = {
        "sequence_length_study": seq_len_results,
        "latent_dimension_study": latent_results,
        "noise_level_study": noise_results,
        "threshold_strategy_study": threshold_comp,
    }

    out_json = os.path.join(REPORTS_DIR, "ablation_study_results.json")
    with open(out_json, "w") as f:
        json.dump(ablation_payload, f, indent=2)
    print(f"\n[Ablation] Saved ablation study results to: {out_json}")

    # Plot Ablation Figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Seq len plot
    df_seq = pd.DataFrame(seq_len_results)
    axes[0].plot(df_seq["sequence_length"], df_seq["f1_score"], marker="o", color="#2563EB", lw=2, label="F1-Score")
    axes[0].plot(df_seq["sequence_length"], df_seq["pr_auc"], marker="s", color="#059669", lw=2, label="PR-AUC")
    axes[0].set_title("Impact of Window Sequence Length (T)", fontweight="bold")
    axes[0].set_xlabel("Sequence Length (timesteps)")
    axes[0].set_ylabel("Metric Score")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Latent dim plot
    df_lat = pd.DataFrame(latent_results)
    axes[1].plot(df_lat["latent_dim"], df_lat["f1_score"], marker="o", color="#2563EB", lw=2, label="F1-Score")
    axes[1].plot(df_lat["latent_dim"], df_lat["pr_auc"], marker="s", color="#059669", lw=2, label="PR-AUC")
    axes[1].set_title("Impact of Latent Bottleneck Dimension", fontweight="bold")
    axes[1].set_xlabel("Latent Bottleneck Size")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Noise plot
    df_noise = pd.DataFrame(noise_results)
    axes[2].plot(df_noise["noise_std"], df_noise["f1_score"], marker="o", color="#2563EB", lw=2, label="F1-Score")
    axes[2].plot(df_noise["noise_std"], df_noise["pr_auc"], marker="s", color="#059669", lw=2, label="PR-AUC")
    axes[2].set_title("Impact of Denoising Noise Level (sigma)", fontweight="bold")
    axes[2].set_xlabel("Gaussian Noise Std (sigma)")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "hyperparameter_ablation_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Ablation] Saved ablation curves plot to: {plot_path}")

    return ablation_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hyperparameter & Ablation Studies.")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per ablation config")
    args = parser.parse_args()
    run_ablation_study(epochs=args.epochs)
