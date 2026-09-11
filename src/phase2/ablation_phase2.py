"""
Phase 2 Ablation Studies:
  1. Prediction Horizon Ablation (6h, 12h, 24h, 48h)
  2. Phase 1 Feature Integration Ablation (Exp A: Sensors-only, Exp B: Sensors+Score, Exp C: Sensors+Score+Errors)
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
    from src.detector import AnomalyDetector
    from src.phase2.dataset import create_phase2_event_splits, build_phase2_supervised_windows
    from src.phase2.models import GRUFailurePredictor
    from src.phase2.train import train_phase2_model
    from src.phase2.evaluator import (
        predict_probabilities,
        calibrate_probability_threshold,
        evaluate_classifier_predictions,
        evaluate_failure_lead_times,
    )
    from src.phase2.visualizer import (
        plot_phase2_horizon_comparison,
        plot_phase2_integration_comparison,
    )
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
    from detector import AnomalyDetector
    from phase2.dataset import create_phase2_event_splits, build_phase2_supervised_windows
    from phase2.models import GRUFailurePredictor
    from phase2.train import train_phase2_model
    from phase2.evaluator import (
        predict_probabilities,
        calibrate_probability_threshold,
        evaluate_classifier_predictions,
        evaluate_failure_lead_times,
    )
    from phase2.visualizer import (
        plot_phase2_horizon_comparison,
        plot_phase2_integration_comparison,
    )


def run_phase2_ablations(epochs: int = 5, batch_size: int = 256):
    print("=" * 110)
    print("                     PHASE 2 SYSTEMATIC ABLATION STUDIES")
    print("=" * 110)

    raw_df = load_and_clean_data()
    train_df, val_df, test_df = create_phase2_event_splits(raw_df)

    scaler = load_scaler(SCALER_PATH)
    scaled_train = transform_data(train_df, scaler)
    scaled_val = transform_data(val_df, scaler)
    scaled_test = transform_data(test_df, scaler)

    # --------------------------------------------------------------------------
    # 1. Prediction Horizon Ablation (6h, 12h, 24h, 48h)
    # --------------------------------------------------------------------------
    print("\n[Ablation 1/2] Evaluating Prediction Horizons (6h, 12h, 24h, 48h)...")
    horizons = [6.0, 12.0, 24.0, 48.0]
    horizon_results = []

    for h in horizons:
        print(f"\n--- Horizon: {h:.0f} Hours ---")
        train_win, train_lbls, _ = build_phase2_supervised_windows(scaled_train, train_df["timestamp"], horizon_hours=h, stride=35)
        val_win, val_lbls, _ = build_phase2_supervised_windows(scaled_val, val_df["timestamp"], horizon_hours=h, stride=25)
        test_win, test_lbls, test_ts = build_phase2_supervised_windows(scaled_test, test_df["timestamp"], horizon_hours=h, stride=20)

        model = GRUFailurePredictor(input_dim=len(ANALOGUE_SENSORS))
        trained_m, _ = train_phase2_model(
            model=model,
            train_windows=train_win,
            train_labels=train_lbls,
            val_windows=val_win,
            val_labels=val_lbls,
            model_name=f"GRU ({h:.0f}h)",
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_dir=None,
        )

        val_probs = predict_probabilities(trained_m, val_win, device=DEVICE)
        calib_res = calibrate_probability_threshold(val_lbls, val_probs)
        opt_thresh = calib_res["calibrated_threshold"]

        test_probs = predict_probabilities(trained_m, test_win, device=DEVICE)
        metrics = evaluate_classifier_predictions(test_lbls, test_probs, threshold=opt_thresh)
        lead_times = evaluate_failure_lead_times(test_ts, test_probs, threshold=opt_thresh)

        detected_leads = [l["lead_time_hours"] for l in lead_times if l["detected"] and l["lead_time_hours"] is not None]
        mean_lead = float(np.mean(detected_leads)) if detected_leads else 0.0

        horizon_results.append({
            "horizon_hours": int(h),
            "threshold": round(opt_thresh, 4),
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1_score"], 4),
            "specificity": round(metrics["specificity"], 4),
            "pr_auc": round(metrics["pr_auc"], 4),
            "roc_auc": round(metrics["roc_auc"], 4),
            "fpr": round(metrics["fpr"], 4),
            "fnr": round(metrics["fnr"], 4),
            "mean_lead_time_hours": round(mean_lead, 2),
            "lead_times": lead_times,
        })

    # Save horizon results & plot
    out_horizon_json = os.path.join(REPORTS_DIR, "phase2_horizon_results.json")
    with open(out_horizon_json, "w") as f:
        json.dump(horizon_results, f, indent=2)
    print(f"\n[Ablation 1/2] Saved horizon results to: {out_horizon_json}")

    plot_phase2_horizon_comparison(horizon_results, os.path.join(FIGURES_DIR, "phase2_horizon_comparison.png"))

    # --------------------------------------------------------------------------
    # 2. Phase 1 Feature Integration Ablation (24h Horizon)
    # --------------------------------------------------------------------------
    print("\n[Ablation 2/2] Evaluating Phase 1 Feature Integration Strategies...")
    detector = AnomalyDetector()

    # Pre-extract Phase 1 features
    def extract_phase1_dense_features(df_subset, scaled_subset):
        # Stride=1 block scoring or window scoring
        # Extract per-row anomaly scores and per-sensor errors
        n_rows = len(scaled_subset)
        score_feat = np.zeros((n_rows, 1), dtype=np.float32)
        decomposed_feat = np.zeros((n_rows, len(ANALOGUE_SENSORS)), dtype=np.float32)

        # Batch compute for speed
        block_size = 500
        for s_idx in range(0, n_rows - SEQUENCE_LENGTH + 1, block_size):
            end_idx = min(s_idx + block_size + SEQUENCE_LENGTH, n_rows)
            sub_scaled = scaled_subset[s_idx:end_idx]
            if len(sub_scaled) < SEQUENCE_LENGTH:
                continue
            # sliding windows
            from src.dataset import build_sliding_windows
            w = build_sliding_windows(sub_scaled, seq_len=SEQUENCE_LENGTH, stride=1)
            if len(w) == 0:
                continue
            with torch.no_grad():
                t = torch.from_numpy(w).float().to(DEVICE)
                recon = detector.model(t)
                sq = (t - recon) ** 2
                seq_err = sq.mean(dim=(1, 2)).cpu().numpy().reshape(-1, 1)
                sens_err = sq.mean(dim=1).cpu().numpy()

            assign_len = len(seq_err)
            tgt_start = s_idx + SEQUENCE_LENGTH - 1
            tgt_end = tgt_start + assign_len
            score_feat[tgt_start:tgt_end] = seq_err
            decomposed_feat[tgt_start:tgt_end] = sens_err

        return score_feat, decomposed_feat

    print("[Ablation 2/2] Extracting Phase 1 anomaly scores on train, val, and test partitions...")
    train_score, train_decomp = extract_phase1_dense_features(train_df, scaled_train)
    val_score, val_decomp = extract_phase1_dense_features(val_df, scaled_val)
    test_score, test_decomp = extract_phase1_dense_features(test_df, scaled_test)

    integration_experiments = [
        {
            "id": "exp_a",
            "name": "Exp A: Sensor-only (7 dims)",
            "extra_train": None,
            "extra_val": None,
            "extra_test": None,
            "input_dim": 7,
        },
        {
            "id": "exp_b",
            "name": "Exp B: Sensor + Anomaly Score (8 dims)",
            "extra_train": train_score,
            "extra_val": val_score,
            "extra_test": test_score,
            "input_dim": 8,
        },
        {
            "id": "exp_c",
            "name": "Exp C: Sensor + Score + Errors (15 dims)",
            "extra_train": np.hstack([train_score, train_decomp]),
            "extra_val": np.hstack([val_score, val_decomp]),
            "extra_test": np.hstack([test_score, test_decomp]),
            "input_dim": 15,
        },
    ]

    integration_results = []

    for exp in integration_experiments:
        print(f"\n--- Running {exp['name']} ---")
        train_win, train_lbls, _ = build_phase2_supervised_windows(
            scaled_train, train_df["timestamp"], horizon_hours=24.0, stride=35, phase1_features=exp["extra_train"]
        )
        val_win, val_lbls, _ = build_phase2_supervised_windows(
            scaled_val, val_df["timestamp"], horizon_hours=24.0, stride=25, phase1_features=exp["extra_val"]
        )
        test_win, test_lbls, test_ts = build_phase2_supervised_windows(
            scaled_test, test_df["timestamp"], horizon_hours=24.0, stride=20, phase1_features=exp["extra_test"]
        )

        model = GRUFailurePredictor(input_dim=exp["input_dim"])
        trained_m, _ = train_phase2_model(
            model=model,
            train_windows=train_win,
            train_labels=train_lbls,
            val_windows=val_win,
            val_labels=val_lbls,
            model_name=exp["name"],
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_dir=None,
        )

        val_probs = predict_probabilities(trained_m, val_win, device=DEVICE)
        calib_res = calibrate_probability_threshold(val_lbls, val_probs)
        opt_thresh = calib_res["calibrated_threshold"]

        test_probs = predict_probabilities(trained_m, test_win, device=DEVICE)
        metrics = evaluate_classifier_predictions(test_lbls, test_probs, threshold=opt_thresh)
        lead_times = evaluate_failure_lead_times(test_ts, test_probs, threshold=opt_thresh)

        detected_leads = [l["lead_time_hours"] for l in lead_times if l["detected"] and l["lead_time_hours"] is not None]
        mean_lead = float(np.mean(detected_leads)) if detected_leads else 0.0

        integration_results.append({
            "experiment_id": exp["id"],
            "experiment_name": exp["name"],
            "input_dim": exp["input_dim"],
            "threshold": round(opt_thresh, 4),
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1_score"], 4),
            "specificity": round(metrics["specificity"], 4),
            "pr_auc": round(metrics["pr_auc"], 4),
            "roc_auc": round(metrics["roc_auc"], 4),
            "fpr": round(metrics["fpr"], 4),
            "fnr": round(metrics["fnr"], 4),
            "mean_lead_time_hours": round(mean_lead, 2),
            "lead_times": lead_times,
        })

    # Save integration results & plot
    out_integ_json = os.path.join(REPORTS_DIR, "phase2_integration_results.json")
    with open(out_integ_json, "w") as f:
        json.dump(integration_results, f, indent=2)
    print(f"\n[Ablation 2/2] Saved integration ablation results to: {out_integ_json}")

    plot_phase2_integration_comparison(integration_results, os.path.join(FIGURES_DIR, "phase2_integration_comparison.png"))

    # Print Summary Tables
    print("\n" + "=" * 115)
    print("                          PREDICTION HORIZON ABLATION SUMMARY")
    print("=" * 115)
    print(f"{'Horizon':<10} | {'Threshold':<10} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'PR-AUC':<8} | {'Mean Lead (h)':<14}")
    print("-" * 115)
    for hr in horizon_results:
        print(f"{hr['horizon_hours']:<3}h       | {hr['threshold']:<10.4f} | {hr['accuracy']:<9.4f} | {hr['precision']:<9.4f} | {hr['recall']:<8.4f} | {hr['f1_score']:<8.4f} | {hr['pr_auc']:<8.4f} | {hr['mean_lead_time_hours']:<14.2f}")
    print("=" * 115)

    print("\n" + "=" * 115)
    print("                     PHASE 1 FEATURE INTEGRATION ABLATION SUMMARY")
    print("=" * 115)
    print(f"{'Experiment':<40} | {'Dims':<6} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'PR-AUC':<8} | {'ROC-AUC':<8}")
    print("-" * 115)
    for ir in integration_results:
        print(f"{ir['experiment_name']:<40} | {ir['input_dim']:<6} | {ir['accuracy']:<9.4f} | {ir['precision']:<9.4f} | {ir['recall']:<8.4f} | {ir['f1_score']:<8.4f} | {ir['pr_auc']:<8.4f} | {ir['roc_auc']:<8.4f}")
    print("=" * 115)

    return {
        "horizon_study": horizon_results,
        "integration_study": integration_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Horizon & Integration Ablation Studies.")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per ablation model")
    args = parser.parse_args()
    run_phase2_ablations(epochs=args.epochs)
