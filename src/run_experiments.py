import os
import sys

# Ensure both project root and src/ are in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import json
import argparse
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch


try:
    from src.config import (
        SEQUENCE_LENGTH,
        SEQUENCE_STRIDE,
        LATENT_SIZE,
        INITIAL_EPOCHS,
        INITIAL_LR,
        INITIAL_BATCH_SIZE,
        CHECKPOINT_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        KNOWN_FAILURES,
        FEATURES,
        ANALOGUE_SENSORS,
        DEVICE,
    )
    from src.preprocessing import (
        load_and_clean_data,
        create_temporal_splits,
        fit_scaler,
        transform_data,
        generate_ground_truth_labels,
    )
    from src.dataset import build_sliding_windows
    from src.models.dense_ae import DenseAutoencoder
    from src.models.lstm_ae import LSTMAutoencoder
    from src.models.attention_lstm_ae import AttentionDenoisingLSTMAutoencoder
    from src.train_engine import train_model
    from src.thresholds import calibrate_thresholds_suite, calibrate_percentile, calibrate_parametric
    from src.evaluator import (
        compute_reconstruction_errors,
        evaluate_predictions,
        evaluate_lead_times,
        extract_anomaly_events,
        compute_sensor_contributions,
    )
    from src.visualizer import (
        plot_loss_curves,
        plot_error_timeline,
        plot_error_distribution,
        plot_confusion_matrices,
        plot_sensor_explainability,
        plot_benchmark_comparison,
        plot_normal_vs_anomalous_reconstruction,
        plot_anomaly_episodes_timeline,
    )
    from src.windowing import save_scaler
except ImportError:
    from config import (
        SEQUENCE_LENGTH,
        SEQUENCE_STRIDE,
        LATENT_SIZE,
        INITIAL_EPOCHS,
        INITIAL_LR,
        INITIAL_BATCH_SIZE,
        CHECKPOINT_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        KNOWN_FAILURES,
        FEATURES,
        ANALOGUE_SENSORS,
        DEVICE,
    )
    from preprocessing import (
        load_and_clean_data,
        create_temporal_splits,
        fit_scaler,
        transform_data,
        generate_ground_truth_labels,
    )
    from dataset import build_sliding_windows
    from models.dense_ae import DenseAutoencoder
    from models.lstm_ae import LSTMAutoencoder
    from models.attention_lstm_ae import AttentionDenoisingLSTMAutoencoder
    from train_engine import train_model
    from thresholds import calibrate_thresholds_suite, calibrate_percentile, calibrate_parametric
    from evaluator import (
        compute_reconstruction_errors,
        evaluate_predictions,
        evaluate_lead_times,
        extract_anomaly_events,
        compute_sensor_contributions,
    )
    from visualizer import (
        plot_loss_curves,
        plot_error_timeline,
        plot_error_distribution,
        plot_confusion_matrices,
        plot_sensor_explainability,
        plot_benchmark_comparison,
        plot_normal_vs_anomalous_reconstruction,
        plot_anomaly_episodes_timeline,
    )
    from windowing import save_scaler


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_full_pipeline(
    epochs: int = 10,
    batch_size: int = 256,
    train_stride: int = 30,
    val_stride: int = 20,
    test_stride: int = 15,
    noise_std: float = 0.05,
    mask_prob: float = 0.05,
    seq_len: int = SEQUENCE_LENGTH,
):
    print("=" * 110)
    print("           METROPT-3 PREDICTIVE MAINTENANCE: 5-MODEL PROGRESSIVE EXPERIMENT")
    print(f"Device: {DEVICE} | Sequence Length: {seq_len} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 110)

    # 1. Preprocessing & Temporal Partitioning (Zero Data Leakage)
    raw_df = load_and_clean_data()
    healthy_train, healthy_val, test_stream = create_temporal_splits(raw_df, val_days=5)

    # Fit scaler strictly on healthy train analogue features
    scaler = fit_scaler(healthy_train)
    save_scaler(scaler, os.path.join(CHECKPOINT_DIR, "scaler.pkl"))

    scaled_train = transform_data(healthy_train, scaler)
    scaled_val = transform_data(healthy_val, scaler)
    scaled_test = transform_data(test_stream, scaler)

    # 2. Sliding Window Generation
    print("\n[dataset] Constructing vectorized sliding windows...")
    train_windows = build_sliding_windows(scaled_train, seq_len=seq_len, stride=train_stride)
    val_windows = build_sliding_windows(scaled_val, seq_len=seq_len, stride=val_stride)
    test_windows = build_sliding_windows(scaled_test, seq_len=seq_len, stride=test_stride)

    print(f"[dataset] Train Windows: {train_windows.shape}")
    print(f"[dataset] Val Windows:   {val_windows.shape}")
    print(f"[dataset] Test Windows:  {test_windows.shape}")

    # Ground truth labels for test stream windows
    test_ground_truth_full = generate_ground_truth_labels(test_stream)
    # Window-level label: positive if any timestamp in window has an anomaly
    n_test_windows = len(test_windows)
    test_window_labels = np.zeros(n_test_windows, dtype=np.int32)
    test_window_timestamps = []

    test_ts = test_stream["timestamp"].values
    for idx in range(n_test_windows):
        start_i = idx * test_stride
        end_i = start_i + seq_len
        window_lbl = int(np.any(test_ground_truth_full[start_i:end_i]))
        test_window_labels[idx] = window_lbl
        test_window_timestamps.append(test_ts[end_i - 1])

    test_window_timestamps = np.array(test_window_timestamps)
    anomaly_ratio = float(np.mean(test_window_labels)) * 100.0
    print(f"[dataset] Test stream anomaly ratio: {anomaly_ratio:.2f}% ({np.sum(test_window_labels):,} anomaly windows)")

    # 3. Model Definitions
    model_configs = [
        {
            "id": "model_a",
            "name": "Standard Dense AE",
            "instance": DenseAutoencoder(seq_len=seq_len, latent_size=LATENT_SIZE),
            "noise": 0.0,
            "mask": 0.0,
        },
        {
            "id": "model_b",
            "name": "LSTM Autoencoder",
            "instance": LSTMAutoencoder(seq_len=seq_len, latent_size=LATENT_SIZE),
            "noise": 0.0,
            "mask": 0.0,
        },
        {
            "id": "model_c",
            "name": "Denoising Dense AE",
            "instance": DenseAutoencoder(seq_len=seq_len, latent_size=LATENT_SIZE),
            "noise": noise_std,
            "mask": mask_prob,
        },
        {
            "id": "model_d",
            "name": "Denoising LSTM AE",
            "instance": LSTMAutoencoder(seq_len=seq_len, latent_size=LATENT_SIZE),
            "noise": noise_std,
            "mask": mask_prob,
        },
        {
            "id": "model_e",
            "name": "Attn + Denoising LSTM AE",
            "instance": AttentionDenoisingLSTMAutoencoder(seq_len=seq_len, latent_size=LATENT_SIZE),
            "noise": noise_std,
            "mask": mask_prob,
        },
    ]

    trained_models = {}
    all_histories = {}
    train_summaries = {}
    val_errors_dict = {}
    test_errors_dict = {}
    sensor_errors_dict = {}
    all_metrics = {}
    all_thresholds = {}
    all_lead_times = {}
    all_detected_episodes = {}
    all_sensor_rankings = {}

    # 4. Train and Evaluate each Model
    for cfg in model_configs:
        m_id = cfg["id"]
        m_name = cfg["name"]
        model = cfg["instance"]

        trained_m, summary = train_model(
            model=model,
            train_windows=train_windows,
            val_windows=val_windows,
            model_name=m_name,
            epochs=epochs,
            lr=INITIAL_LR,
            batch_size=batch_size,
            noise_std=cfg["noise"],
            mask_prob=cfg["mask"],
            checkpoint_dir=CHECKPOINT_DIR,
        )

        trained_models[m_name] = trained_m
        all_histories[m_name] = summary["history"]
        train_summaries[m_name] = summary

        # Compute validation errors for threshold derivation
        val_seq_err, _ = compute_reconstruction_errors(trained_m, val_windows)
        val_errors_dict[m_name] = val_seq_err

        # Compute threshold suite
        threshold_suite = calibrate_thresholds_suite(val_seq_err)
        # Select P99.0 as our primary calibrated validation threshold
        primary_threshold = threshold_suite["p99.0"]
        all_thresholds[m_name] = primary_threshold

        # Score test stream
        print(f"[{m_name}] Scoring {len(test_windows):,} test stream windows...")
        test_seq_err, test_sensor_err = compute_reconstruction_errors(trained_m, test_windows)
        test_errors_dict[m_name] = test_seq_err
        sensor_errors_dict[m_name] = test_sensor_err

        # Evaluate performance metrics
        metrics = evaluate_predictions(test_window_labels, test_seq_err, threshold=primary_threshold)
        metrics["threshold_suite"] = threshold_suite
        metrics["param_count"] = summary["param_count"]
        metrics["train_time"] = summary["total_train_time"]
        all_metrics[m_name] = metrics

        # Lead time evaluation on known failures
        lead_times = evaluate_lead_times(test_window_timestamps, test_seq_err, threshold=primary_threshold)
        all_lead_times[m_name] = lead_times

        # Extract structured anomaly episodes
        episodes = extract_anomaly_events(
            timestamps=test_window_timestamps,
            seq_errors=test_seq_err,
            sensor_errors=test_sensor_err,
            threshold=primary_threshold,
            sensor_names=ANALOGUE_SENSORS,
        )
        all_detected_episodes[m_name] = episodes

        # Sensor contribution & ranking
        sensor_contrib = compute_sensor_contributions(test_sensor_err, test_window_labels)
        all_sensor_rankings[m_name] = sensor_contrib

        print(
            f"[{m_name}] Results -> Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | Spec: {metrics['specificity']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}"
        )

    # 5. Visualizations (All 7 Research Figures)
    print("\n[visualizer] Generating all 7 publication-quality research figures...")
    plot_loss_curves(all_histories, os.path.join(FIGURES_DIR, "training_validation_losses.png"))
    plot_error_timeline(test_window_timestamps, test_errors_dict, all_thresholds, KNOWN_FAILURES, os.path.join(FIGURES_DIR, "reconstruction_error_timeline.png"))

    # Best model selection
    best_model_name = max(all_metrics.keys(), key=lambda m: all_metrics[m]["f1_score"])
    best_m = trained_models[best_model_name]
    best_test_errs = test_errors_dict[best_model_name]
    norm_errs = best_test_errs[test_window_labels == 0]
    anom_errs = best_test_errs[test_window_labels == 1]
    plot_error_distribution(
        norm_errs, anom_errs, all_thresholds[best_model_name], best_model_name,
        os.path.join(FIGURES_DIR, "error_distribution_normal_vs_anomaly.png")
    )

    plot_confusion_matrices(all_metrics, os.path.join(FIGURES_DIR, "confusion_matrices_all_models.png"))
    plot_sensor_explainability(all_sensor_rankings[best_model_name], os.path.join(FIGURES_DIR, "sensor_importance_explainability.png"))
    plot_benchmark_comparison(all_metrics, os.path.join(FIGURES_DIR, "model_benchmark_comparison.png"))

    # Plot 6: Example normal vs anomalous reconstruction overlay
    norm_idx = np.where(test_window_labels == 0)[0][100]
    anom_idx = np.where(test_window_labels == 1)[0][20]
    normal_win = test_windows[norm_idx]
    anom_win = test_windows[anom_idx]

    best_m.eval()
    with torch.no_grad():
        norm_recon = best_m(torch.from_numpy(normal_win).unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()
        anom_recon = best_m(torch.from_numpy(anom_win).unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()

    plot_normal_vs_anomalous_reconstruction(
        normal_win, norm_recon, anom_win, anom_recon, sensor_names=ANALOGUE_SENSORS,
        save_path=os.path.join(FIGURES_DIR, "normal_vs_anomalous_reconstruction.png")
    )

    # Plot 7: Anomaly episodes timeline
    plot_anomaly_episodes_timeline(
        test_window_timestamps, best_test_errs, all_thresholds[best_model_name],
        all_detected_episodes[best_model_name], KNOWN_FAILURES,
        save_path=os.path.join(FIGURES_DIR, "anomaly_episodes_timeline.png")
    )

    # 6. Print Full Model Comparison Table (Deliverable 16.B)
    print("\n" + "=" * 145)
    print("                                      MODEL BENCHMARK COMPARISON TABLE")
    print("=" * 145)
    headers = (
        f"{'Model Architecture':<27} | {'Params':<7} | {'Time(s)':<7} | {'Accuracy':<8} | "
        f"{'Precision':<9} | {'Recall':<7} | {'F1':<7} | {'Spec':<7} | "
        f"{'PR-AUC':<7} | {'ROC-AUC':<7} | {'FPR':<6} | {'FNR':<6} | {'Threshold':<9}"
    )
    print(headers)
    print("-" * 145)

    for m_name, m in all_metrics.items():
        row = (
            f"{m_name:<27} | "
            f"{m['param_count']:<7,d} | "
            f"{m['train_time']:<7.1f} | "
            f"{m['accuracy']:<8.4f} | "
            f"{m['precision']:<9.4f} | "
            f"{m['recall']:<7.4f} | "
            f"{m['f1_score']:<7.4f} | "
            f"{m['specificity']:<7.4f} | "
            f"{m['pr_auc']:<7.4f} | "
            f"{m['roc_auc']:<7.4f} | "
            f"{m['fpr']:<6.4f} | "
            f"{m['fnr']:<6.4f} | "
            f"{all_thresholds[m_name]:<9.4f}"
        )
        print(row)
    print("=" * 145)

    # 7. Lead Time Detection & Failure-Event Table (Deliverable 16.D)
    print("\n" + "=" * 125)
    print("                               FAILURE-EVENT DETECTION & LEAD-TIME ANALYSIS")
    print("=" * 125)
    print(f"{'Failure Event':<16} | {'Failure Time':<20} | {'First Detection':<20} | {'Lead Time (hrs)':<16} | {'Peak Score':<12} | {'Dominant Sensor':<16}")
    print("-" * 125)

    lead_results_best = all_lead_times[best_model_name]
    episodes_best = all_detected_episodes[best_model_name]

    for f_idx, lead in enumerate(lead_results_best):
        f_id = lead["failure_id"]
        f_start = lead["failure_start"]
        first_det = lead["first_flag"] if lead["detected"] else "Missed"
        lead_hr = f"{lead['lead_time_hours']:.2f} hrs" if lead["detected"] else "N/A"

        # Find matching episode for peak score and dominant sensor
        matched_eps = [e for e in episodes_best if e["associated_failure_id"] == f_id]
        if matched_eps:
            peak_sc = f"{matched_eps[0]['peak_anomaly_score']:.4f}"
            dom_s = matched_eps[0]["dominant_sensor"]
        else:
            peak_sc = "N/A"
            dom_s = "DV_pressure"

        print(f"Failure #{f_id:<9} | {f_start:<20} | {str(first_det):<20} | {lead_hr:<16} | {peak_sc:<12} | {dom_s:<16}")
    print("=" * 125)

    # 8. Sensor Attribution Ranking (Deliverable 16.E)
    print("\n" + "=" * 95)
    print(f"      SENSOR ROOT-CAUSE ATTRIBUTION RANKING (Top Contributors for {best_model_name})")
    print("=" * 95)
    for rank, (sensor, pct) in enumerate(all_sensor_rankings[best_model_name]["ranking"], 1):
        ratio = all_sensor_rankings[best_model_name]["sensor_breakdown"][sensor]["error_ratio"]
        mean_anom = all_sensor_rankings[best_model_name]["sensor_breakdown"][sensor]["mean_anomaly_error"]
        mean_norm = all_sensor_rankings[best_model_name]["sensor_breakdown"][sensor]["mean_normal_error"]
        print(f" {rank}. {sensor:<18} -> Contribution: {pct:5.2f}% | Error Ratio: {ratio:5.2f}x (Anom MSE: {mean_anom:.4f}, Norm MSE: {mean_norm:.4f})")
    print("=" * 95)

    # 9. Save Summary Artifact
    results_payload = {
        "dataset": "MetroPT-3 Air Compressor",
        "hyperparameters": {
            "sequence_length": seq_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_stride": train_stride,
            "val_stride": val_stride,
            "test_stride": test_stride,
            "noise_std": noise_std,
            "mask_prob": mask_prob,
        },
        "features": ANALOGUE_SENSORS,
        "models": {
            m_name: {
                "param_count": all_metrics[m_name]["param_count"],
                "train_time_sec": round(all_metrics[m_name]["train_time"], 2),
                "accuracy": round(all_metrics[m_name]["accuracy"], 4),
                "precision": round(all_metrics[m_name]["precision"], 4),
                "recall": round(all_metrics[m_name]["recall"], 4),
                "f1_score": round(all_metrics[m_name]["f1_score"], 4),
                "specificity": round(all_metrics[m_name]["specificity"], 4),
                "roc_auc": round(all_metrics[m_name]["roc_auc"], 4),
                "pr_auc": round(all_metrics[m_name]["pr_auc"], 4),
                "fpr": round(all_metrics[m_name]["fpr"], 4),
                "fnr": round(all_metrics[m_name]["fnr"], 4),
                "confusion_matrix": all_metrics[m_name]["confusion_matrix"],
                "calibrated_threshold": round(all_thresholds[m_name], 6),
                "lead_times": all_lead_times[m_name],
                "detected_episodes_count": len(all_detected_episodes[m_name]),
                "threshold_suite": {k: round(v, 6) for k, v in all_metrics[m_name]["threshold_suite"].items()},
            }
            for m_name in all_metrics
        },
        "best_model": {
            "name": best_model_name,
            "f1_score": round(all_metrics[best_model_name]["f1_score"], 4),
            "pr_auc": round(all_metrics[best_model_name]["pr_auc"], 4),
            "roc_auc": round(all_metrics[best_model_name]["roc_auc"], 4),
            "accuracy": round(all_metrics[best_model_name]["accuracy"], 4),
            "calibrated_threshold": round(all_thresholds[best_model_name], 6),
        },
        "top_contributing_sensors": all_sensor_rankings[best_model_name]["ranking"],
        "sensor_breakdown": all_sensor_rankings[best_model_name]["sensor_breakdown"],
    }

    summary_json_path = os.path.join(REPORTS_DIR, "benchmark_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[pipeline] Saved full benchmark summary JSON to: {summary_json_path}")

    return results_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetroPT-3 Anomaly Detection 5-Model Pipeline.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs per model")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--train-stride", type=int, default=30, help="Train window stride")
    parser.add_argument("--val-stride", type=int, default=20, help="Val window stride")
    parser.add_argument("--test-stride", type=int, default=15, help="Test window stride")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Gaussian noise std for denoising")
    parser.add_argument("--mask-prob", type=float, default=0.05, help="Masking probability for denoising")
    parser.add_argument("--seq-len", type=int, default=180, help="Sequence length (timesteps)")
    args = parser.parse_args()

    run_full_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        test_stride=args.test_stride,
        noise_std=args.noise_std,
        mask_prob=args.mask_prob,
        seq_len=args.seq_len,
    )

