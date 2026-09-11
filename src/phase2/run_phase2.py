"""
Phase 2 Benchmark Pipeline: Supervised Impending Failure Prediction (24-Hour Horizon).
Compares Logistic Regression Baseline vs. 2-Layer LSTM vs. 2-Layer GRU.
"""
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
_project_root = os.path.dirname(_src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import json
import argparse
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

try:
    from src.config import (
        ANALOGUE_SENSORS,
        FEATURES,
        KNOWN_FAILURES,
        SEQUENCE_LENGTH,
        CHECKPOINT_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        SCALER_PATH,
        DEVICE,
    )
    from src.preprocessing import load_and_clean_data, transform_data
    from src.windowing import load_scaler
    from src.phase2.dataset import create_phase2_event_splits, build_phase2_supervised_windows
    from src.phase2.models import LogisticRegressionBaseline, LSTMFailurePredictor, GRUFailurePredictor
    from src.phase2.train import train_phase2_model
    from src.phase2.evaluator import (
        predict_probabilities,
        calibrate_probability_threshold,
        evaluate_classifier_predictions,
        evaluate_failure_lead_times,
    )
    from src.phase2.visualizer import (
        plot_phase2_probability_timeline,
        plot_phase2_lead_times,
        plot_phase2_model_comparison,
        plot_phase2_confusion_matrices,
        plot_phase2_pr_roc_curves,
        plot_phase2_risk_assessment_demo,
    )
    from src.detector import detect_anomaly
    from src.phase2.assessor import assess_failure_risk
except ImportError:
    from config import (
        ANALOGUE_SENSORS,
        FEATURES,
        KNOWN_FAILURES,
        SEQUENCE_LENGTH,
        CHECKPOINT_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        SCALER_PATH,
        DEVICE,
    )
    from preprocessing import load_and_clean_data, transform_data
    from windowing import load_scaler
    from phase2.dataset import create_phase2_event_splits, build_phase2_supervised_windows
    from phase2.models import LogisticRegressionBaseline, LSTMFailurePredictor, GRUFailurePredictor
    from phase2.train import train_phase2_model
    from phase2.evaluator import (
        predict_probabilities,
        calibrate_probability_threshold,
        evaluate_classifier_predictions,
        evaluate_failure_lead_times,
    )
    from phase2.visualizer import (
        plot_phase2_probability_timeline,
        plot_phase2_lead_times,
        plot_phase2_model_comparison,
        plot_phase2_confusion_matrices,
        plot_phase2_pr_roc_curves,
        plot_phase2_risk_assessment_demo,
    )
    from detector import detect_anomaly
    from phase2.assessor import assess_failure_risk


def run_phase2_pipeline(
    epochs: int = 10,
    batch_size: int = 256,
    train_stride: int = 25,
    val_stride: int = 15,
    test_stride: int = 10,
    horizon_hours: float = 24.0,
    lr: float = 1e-3,
):
    print("=" * 110)
    print(f"       METROPT-3 PREDICTIVE MAINTENANCE — PHASE 2: IMPENDING FAILURE PREDICTION ({horizon_hours:.0f}H HORIZON)")
    print(f"Device: {DEVICE} | Horizon: {horizon_hours:.0f} Hours | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 110)

    # 1. Load data and create zero-leakage event splits
    raw_df = load_and_clean_data()
    train_df, val_df, test_df = create_phase2_event_splits(raw_df)

    # Load Phase 1 fitted scaler
    scaler = load_scaler(SCALER_PATH)
    scaled_train = transform_data(train_df, scaler)
    scaled_val = transform_data(val_df, scaler)
    scaled_test = transform_data(test_df, scaler)

    # 2. Build supervised windows with 24-hour impending failure labels
    print(f"\n[dataset] Generating supervised sliding windows with {horizon_hours:.0f}h failure horizon...")
    train_win, train_lbls, train_ts = build_phase2_supervised_windows(
        scaled_train, train_df["timestamp"], horizon_hours=horizon_hours, stride=train_stride
    )
    val_win, val_lbls, val_ts = build_phase2_supervised_windows(
        scaled_val, val_df["timestamp"], horizon_hours=horizon_hours, stride=val_stride
    )
    test_win, test_lbls, test_ts = build_phase2_supervised_windows(
        scaled_test, test_df["timestamp"], horizon_hours=horizon_hours, stride=test_stride
    )

    print(f"[dataset] Train Windows: {train_win.shape} | Positive: {train_lbls.sum():,} ({train_lbls.mean()*100:.2f}%)")
    print(f"[dataset] Val Windows:   {val_win.shape} | Positive: {val_lbls.sum():,} ({val_lbls.mean()*100:.2f}%)")
    print(f"[dataset] Test Windows:  {test_win.shape} | Positive: {test_lbls.sum():,} ({test_lbls.mean()*100:.2f}%)")

    # 3. Models to Benchmark
    models_to_run = [
        {
            "name": "Logistic Regression",
            "instance": LogisticRegressionBaseline(input_dim=len(ANALOGUE_SENSORS)),
        },
        {
            "name": "LSTM",
            "instance": LSTMFailurePredictor(input_dim=len(ANALOGUE_SENSORS)),
        },
        {
            "name": "GRU",
            "instance": GRUFailurePredictor(input_dim=len(ANALOGUE_SENSORS)),
        },
    ]

    trained_models = {}
    train_summaries = {}
    val_probs_dict = {}
    test_probs_dict = {}
    calibrated_thresholds = {}
    all_metrics = {}
    all_lead_times = {}

    # 4. Train and Evaluate each Model
    for m_cfg in models_to_run:
        m_name = m_cfg["name"]
        model = m_cfg["instance"]

        trained_m, summary = train_phase2_model(
            model=model,
            train_windows=train_win,
            train_labels=train_lbls,
            val_windows=val_win,
            val_labels=val_lbls,
            model_name=m_name,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            checkpoint_dir=CHECKPOINT_DIR,
        )

        trained_models[m_name] = trained_m
        train_summaries[m_name] = summary

        # 1. Validation Predictions & Threshold Calibration
        val_probs = predict_probabilities(trained_m, val_win, device=DEVICE)
        val_probs_dict[m_name] = val_probs

        calib_res = calibrate_probability_threshold(val_lbls, val_probs)
        opt_thresh = calib_res["calibrated_threshold"]
        calibrated_thresholds[m_name] = opt_thresh
        print(f"[{m_name}] Validation Optimal Threshold: {opt_thresh:.4f} (Best Val F1: {calib_res['best_val_f1']:.4f})")

        # 2. Test Stream Evaluation
        test_probs = predict_probabilities(trained_m, test_win, device=DEVICE)
        test_probs_dict[m_name] = test_probs

        metrics = evaluate_classifier_predictions(test_lbls, test_probs, threshold=opt_thresh)
        metrics["param_count"] = summary["param_count"]
        metrics["train_time"] = summary["total_train_time"]
        all_metrics[m_name] = metrics

        # 3. Lead Time Evaluation across all 4 documented failures
        lead_times = evaluate_failure_lead_times(test_ts, test_probs, threshold=opt_thresh)
        all_lead_times[m_name] = lead_times

        print(
            f"[{m_name}] Test Results -> Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | Spec: {metrics['specificity']:.4f} | "
            f"PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}"
        )

    # 5. Visualizations
    print("\n[visualizer] Generating Phase 2 publication-quality figures...")
    plot_phase2_probability_timeline(
        test_ts, test_probs_dict, calibrated_thresholds, KNOWN_FAILURES,
        os.path.join(FIGURES_DIR, "phase2_probability_timeline.png")
    )
    plot_phase2_lead_times(
        all_lead_times, os.path.join(FIGURES_DIR, "phase2_lead_times.png")
    )
    plot_phase2_model_comparison(
        all_metrics, os.path.join(FIGURES_DIR, "phase2_model_comparison.png")
    )
    plot_phase2_confusion_matrices(
        all_metrics, os.path.join(FIGURES_DIR, "phase2_confusion_matrices.png")
    )
    plot_phase2_pr_roc_curves(
        test_lbls, test_probs_dict, os.path.join(FIGURES_DIR, "phase2_pr_roc_curves.png")
    )

    # Select Best Phase 2 Model based on balanced PR-AUC, F1, and computational efficiency
    best_model_name = max(all_metrics.keys(), key=lambda m: (all_metrics[m]["pr_auc"] + all_metrics[m]["f1_score"]))

    # Save calibrated threshold config for API loading
    threshold_cfg_path = os.path.join(CHECKPOINT_DIR, "phase2_threshold.json")
    with open(threshold_cfg_path, "w") as f:
        json.dump({
            "selected_model": best_model_name,
            "calibrated_threshold": calibrated_thresholds[best_model_name],
            "all_thresholds": calibrated_thresholds,
            "prediction_horizon_hours": int(horizon_hours),
        }, f, indent=2)

    # Combined assessment demo figure
    normal_window = scaled_test[1000 : 1000 + SEQUENCE_LENGTH]
    anom_window = scaled_test[np.where(test_lbls == 1)[0][20] * test_stride : np.where(test_lbls == 1)[0][20] * test_stride + SEQUENCE_LENGTH]

    norm_assessment = assess_failure_risk(normal_window, is_raw=False)
    anom_assessment = assess_failure_risk(anom_window, is_raw=False)

    plot_phase2_risk_assessment_demo(
        norm_assessment, anom_assessment, os.path.join(FIGURES_DIR, "phase2_risk_assessment_demo.png")
    )

    # 6. Print Model Comparison Table (Deliverable 17)
    print("\n" + "=" * 145)
    print("                              PHASE 2 MODEL COMPARISON TABLE (24-HOUR HORIZON)")
    print("=" * 145)
    headers = (
        f"{'Model Architecture':<24} | {'Horizon':<7} | {'Params':<8} | {'Time(s)':<7} | "
        f"{'Accuracy':<8} | {'Precision':<9} | {'Recall':<7} | {'F1':<7} | {'Spec':<7} | "
        f"{'PR-AUC':<7} | {'ROC-AUC':<7} | {'FPR':<6} | {'FNR':<6} | {'Threshold':<9}"
    )
    print(headers)
    print("-" * 145)

    for m_name, m in all_metrics.items():
        row = (
            f"{m_name:<24} | "
            f"{int(horizon_hours):<3}h    | "
            f"{m['param_count']:<8,d} | "
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
            f"{calibrated_thresholds[m_name]:<9.4f}"
        )
        print(row)
    print("=" * 145)

    # 7. Print Failure Lead-Time Analysis Table
    print("\n" + "=" * 115)
    print("                        PHASE 2 PREDICTIVE WARNING LEAD TIMES ON TEST FAILURES")
    print("=" * 115)
    print(f"{'Failure Event':<16} | {'Failure Start Time':<20} | {'First Prediction Time':<22} | {'Lead Time (hrs)':<16} | {'Peak Prob':<10}")
    print("-" * 115)

    best_lead_times = all_lead_times[best_model_name]
    for lead in best_lead_times:
        f_id = lead["failure_id"]
        f_start = lead["failure_start"]
        first_flag = lead["first_flag"] if lead["detected"] else "Missed"
        lead_hr = f"{lead['lead_time_hours']:.2f} hrs" if lead["detected"] else "N/A"
        peak_p = f"{lead['peak_probability']*100:.1f}%" if lead["peak_probability"] > 0 else "0.0%"
        print(f"Failure #{f_id:<9} | {f_start:<20} | {str(first_flag):<22} | {lead_hr:<16} | {peak_p:<10}")
    print("=" * 115)

    # 8. Save Metrics Artifacts
    metrics_payload = {
        "prediction_horizon_hours": horizon_hours,
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
                "calibrated_threshold": round(calibrated_thresholds[m_name], 4),
                "confusion_matrix": all_metrics[m_name]["confusion_matrix"],
                "lead_times": all_lead_times[m_name],
            }
            for m_name in all_metrics
        },
        "best_model": {
            "name": best_model_name,
            "threshold": round(calibrated_thresholds[best_model_name], 4),
            "f1_score": round(all_metrics[best_model_name]["f1_score"], 4),
            "pr_auc": round(all_metrics[best_model_name]["pr_auc"], 4),
            "roc_auc": round(all_metrics[best_model_name]["roc_auc"], 4),
        },
    }

    metrics_json_path = os.path.join(REPORTS_DIR, "phase2_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\n[pipeline] Saved Phase 2 metrics summary JSON to: {metrics_json_path}")

    lead_times_path = os.path.join(REPORTS_DIR, "phase2_lead_times.json")
    with open(lead_times_path, "w") as f:
        json.dump(all_lead_times, f, indent=2)
    print(f"[pipeline] Saved Phase 2 lead times JSON to: {lead_times_path}")

    return metrics_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Impending Failure Prediction Benchmark.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--horizon", type=float, default=24.0, help="Prediction horizon in hours")
    parser.add_argument("--train-stride", type=int, default=25, help="Train window stride")
    parser.add_argument("--val-stride", type=int, default=15, help="Val window stride")
    parser.add_argument("--test-stride", type=int, default=10, help="Test window stride")
    args = parser.parse_args()

    run_phase2_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        horizon_hours=args.horizon,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        test_stride=args.test_stride,
    )
