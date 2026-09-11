"""
Phase 2B Master Execution Script: Temporal Degradation & Impending Failure Risk Pipeline.
Runs multi-horizon experiments (6h, 12h, 24h, 48h), validates causality,
evaluates persistence rules, compares Phase 2A vs 2B, and outputs all reports.
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    DEVICE,
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
    train_and_evaluate_phase2b,
)
from src.phase2.visualize_phase2b import (
    plot_1_anomaly_score_timeline,
    plot_2_and_3_anomaly_around_failure,
    plot_4_dominant_sensor_errors,
    plot_5_temporal_degradation_trend,
    plot_6_risk_score_timeline,
)


def run_full_phase2b_pipeline(
    stride: int = 10,
    primary_horizon: float = 24.0,
    all_horizons: List[float] = [6.0, 12.0, 24.0, 48.0],
):
    print("=" * 110)
    print("       METROPT-3 PREDICTIVE MAINTENANCE — PHASE 2B: TEMPORAL DEGRADATION & FAILURE RISK PIPELINE")
    print("=" * 110)

    # 1. Load data and create chronological zero-leakage splits
    raw_df = load_and_clean_data()
    train_df, val_df, test_df = create_phase2_event_splits(raw_df)

    scaler = load_scaler(SCALER_PATH)
    scaled_train = transform_data(train_df, scaler)
    scaled_val = transform_data(val_df, scaler)
    scaled_test = transform_data(test_df, scaler)

    # 2. Load Phase 1 Denoising Dense Autoencoder Detector
    phase1_ckpt_path = os.path.join(CHECKPOINT_DIR, "denoising_dense_ae.pt")
    phase1_threshold = 2.913448  # Calibrated P99 from Phase 1 benchmark
    detector = AnomalyDetector(
        model_path=phase1_ckpt_path,
        scaler=scaler,
        model_type="dense",
        threshold=phase1_threshold,
    )
    print(f"\n[phase1] Loaded Phase 1 Detector: Denoising Dense AE (Operational P99 Threshold = {phase1_threshold:.4f})")

    # 3. Compute Phase 1 Reconstruction MSE and per-sensor errors
    print("\n[features] Extracting Phase-1 sequence MSE and per-sensor decomposition across splits...")
    tr_ts, tr_mse, tr_sens_err, tr_win = compute_phase1_stream_scores(scaled_train, train_df["timestamp"], detector, stride=stride)
    va_ts, va_mse, va_sens_err, va_win = compute_phase1_stream_scores(scaled_val, val_df["timestamp"], detector, stride=stride)
    te_ts, te_mse, te_sens_err, te_win = compute_phase1_stream_scores(scaled_test, test_df["timestamp"], detector, stride=stride)

    print(f"[features] Train Stream: {len(tr_ts):,} windows ({tr_ts[0]} -> {tr_ts[-1]})")
    print(f"[features] Val Stream:   {len(va_ts):,} windows ({va_ts[0]} -> {va_ts[-1]})")
    print(f"[features] Test Stream:  {len(te_ts):,} windows ({te_ts[0]} -> {te_ts[-1]})")

    # 4. Construct Causal Temporal Degradation Features
    print("\n[features] Building causal temporal degradation features (rolling stats, persistence, slopes, acceleration, contributions)...")
    X_train, feat_names = build_temporal_degradation_features(tr_ts, tr_mse, tr_sens_err, phase1_threshold=phase1_threshold)
    X_val, _ = build_temporal_degradation_features(va_ts, va_mse, va_sens_err, phase1_threshold=phase1_threshold)
    X_test, _ = build_temporal_degradation_features(te_ts, te_mse, te_sens_err, phase1_threshold=phase1_threshold)

    print(f"[features] Extracted {len(feat_names)} temporal features per window.")
    print(f"[features] Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 5. Strict Causality Audit
    verify_strict_causality(te_ts, n_checks=10)

    # 6. Multi-Horizon Experiments
    horizon_results = {}
    sweep_dfs = []
    
    print("\n" + "=" * 110)
    print("                     PHASE 2B MULTI-HORIZON RISK EXPERIMENTS (6h, 12h, 24h, 48h)")
    print("=" * 110)

    primary_res = None

    for h in all_horizons:
        y_train = generate_impending_failure_labels(tr_ts, horizon_hours=h)
        y_val = generate_impending_failure_labels(va_ts, horizon_hours=h)
        y_test = generate_impending_failure_labels(te_ts, horizon_hours=h)

        feature_scaler = StandardScaler()
        res = train_and_evaluate_phase2b(
            train_features=X_train,
            train_labels=y_train,
            val_features=X_val,
            val_labels=y_val,
            test_features=X_test,
            test_labels=y_test,
            feature_scaler=feature_scaler,
            val_timestamps=va_ts,
            test_timestamps=te_ts,
            horizon_hours=h,
        )

        horizon_results[f"{int(h)}h"] = res
        if h == primary_horizon:
            primary_res = res

        # Collect validation sweep results
        for sw in res["val_sweep_results"]:
            sweep_dfs.append({
                "horizon": f"{int(h)}h",
                "persistence_rule": sw["persistence"],
                "threshold": sw["threshold"],
                "val_precision": round(sw["precision"], 4),
                "val_recall": round(sw["recall"], 4),
                "val_f1": round(sw["f1"], 4),
            })

        m = res["metrics"]
        print(
            f"Horizon: {int(h):2d}h | Train: {y_train.sum():4d}/{len(y_train):,} ({y_train.mean()*100:.2f}%) | "
            f"Val: {y_val.sum():3d}/{len(y_val):,} ({y_val.mean()*100:.2f}%) | "
            f"Test: {y_test.sum():4d}/{len(y_test):,} ({y_test.mean()*100:.2f}%) | "
            f"ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} (Base: {m['positive_prevalence_baseline']:.4f}) | "
            f"F1: {m['f1_score']:.4f} | Rec: {m['recall']:.4f} | Prec: {m['precision']:.4f} | "
            f"FPR: {m['fpr']:.4f} | Thresh: {res['frozen_threshold']:.2f} (P={res['frozen_persistence']})"
        )

    # 7. Generate Visualizations
    print("\n[visualizer] Generating all 6 publication-quality figures...")
    
    # Concatenate full stream for Plot 1
    full_ts = np.concatenate([tr_ts, va_ts, te_ts])
    full_mse = np.concatenate([tr_mse, va_mse, te_mse])
    plot_1_anomaly_score_timeline(
        full_ts, full_mse, phase1_threshold, save_path=os.path.join(FIGURES_DIR, "phase2b_plot1_anomaly_score_timeline.png")
    )

    plot_2_and_3_anomaly_around_failure(
        te_ts, te_mse, failure_id=3, phase1_threshold=phase1_threshold,
        save_path=os.path.join(FIGURES_DIR, "phase2b_plot2_failure3_anomaly.png")
    )
    plot_2_and_3_anomaly_around_failure(
        te_ts, te_mse, failure_id=4, phase1_threshold=phase1_threshold,
        save_path=os.path.join(FIGURES_DIR, "phase2b_plot3_failure4_anomaly.png")
    )

    plot_4_dominant_sensor_errors(
        te_ts, te_sens_err, save_path=os.path.join(FIGURES_DIR, "phase2b_plot4_dominant_sensor_errors.png")
    )

    features_test_df = pd.DataFrame(X_test, columns=feat_names)
    plot_5_temporal_degradation_trend(
        te_ts, features_test_df, save_path=os.path.join(FIGURES_DIR, "phase2b_plot5_temporal_degradation_trend.png")
    )

    plot_6_risk_score_timeline(
        te_ts, primary_res["test_probs"], frozen_threshold=primary_res["frozen_threshold"],
        save_path=os.path.join(FIGURES_DIR, "phase2b_plot6_risk_score_timeline.png")
    )

    # 8. Save Required Output Files
    print("\n[storage] Saving all required Phase 2B data artifacts and configs...")
    
    # 1) phase2b_features.csv (sample first 10,000 and around test failures to keep file lightweight and persistent)
    features_export_df = pd.DataFrame(X_test, columns=feat_names)
    features_export_df.insert(0, "timestamp", te_ts)
    features_export_path = os.path.join(REPORTS_DIR, "phase2b_features.csv")
    features_export_df.head(5000).to_csv(features_export_path, index=False)
    print(f"[storage] Saved sample features to: {features_export_path}")

    # 2) phase2b_predictions.csv
    y_test_24h = generate_impending_failure_labels(te_ts, horizon_hours=24.0)
    preds_df = pd.DataFrame({
        "timestamp": te_ts,
        "ground_truth_label_24h": y_test_24h,
        "predicted_risk_probability": primary_res["test_probs"],
        "predicted_class": primary_res["test_preds"],
        "anomaly_score": te_mse,
    })
    preds_path = os.path.join(REPORTS_DIR, "phase2b_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"[storage] Saved predictions to: {preds_path}")

    # 3) phase2b_threshold_analysis.csv
    threshold_analysis_df = pd.DataFrame(sweep_dfs)
    thresh_csv_path = os.path.join(REPORTS_DIR, "phase2b_threshold_analysis.csv")
    threshold_analysis_df.to_csv(thresh_csv_path, index=False)
    print(f"[storage] Saved threshold analysis to: {thresh_csv_path}")

    # 4) phase2b_event_detection.json
    event_det_path = os.path.join(REPORTS_DIR, "phase2b_event_detection.json")
    with open(event_det_path, "w") as f:
        json.dump(primary_res["event_detection"], f, indent=2)
    print(f"[storage] Saved event detection results to: {event_det_path}")

    # 5) phase2b_metrics.json
    metrics_payload = {
        "pipeline": "Phase 2B — Phase-1-Derived Temporal Degradation Model",
        "base_detector": "Denoising Dense Autoencoder (Phase 1)",
        "phase1_p99_threshold": phase1_threshold,
        "primary_horizon_hours": primary_horizon,
        "horizons": {
            h_str: {
                "frozen_threshold": horizon_results[h_str]["frozen_threshold"],
                "frozen_persistence_windows": horizon_results[h_str]["frozen_persistence"],
                "best_val_f1": round(horizon_results[h_str]["best_val_f1"], 4),
                "metrics": horizon_results[h_str]["metrics"],
                "event_detection": horizon_results[h_str]["event_detection"],
            }
            for h_str in horizon_results
        },
    }
    metrics_json_path = os.path.join(REPORTS_DIR, "phase2b_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[storage] Saved Phase 2B metrics summary to: {metrics_json_path}")

    # 6) Checkpoint configurations
    feature_cfg_path = os.path.join(CHECKPOINT_DIR, "phase2b_feature_config.json")
    with open(feature_cfg_path, "w") as f:
        json.dump({
            "feature_count": len(feat_names),
            "feature_names": feat_names,
            "lookback_steps": {"short": 6, "medium": 24, "long": 72},
            "phase1_threshold": phase1_threshold,
        }, f, indent=2)

    thresh_cfg_path = os.path.join(CHECKPOINT_DIR, "phase2b_threshold.json")
    with open(thresh_cfg_path, "w") as f:
        json.dump({
            "selected_model": "Logistic Regression (Temporal Degradation)",
            "primary_horizon_hours": int(primary_horizon),
            "frozen_threshold": primary_res["frozen_threshold"],
            "frozen_persistence": primary_res["frozen_persistence"],
            "all_horizon_thresholds": {
                h_str: {
                    "threshold": horizon_results[h_str]["frozen_threshold"],
                    "persistence": horizon_results[h_str]["frozen_persistence"],
                }
                for h_str in horizon_results
            },
        }, f, indent=2)

    return horizon_results, feat_names, primary_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2B Temporal Degradation & Risk Pipeline.")
    parser.add_argument("--stride", type=int, default=10, help="Stride between sequence windows")
    parser.add_argument("--horizon", type=float, default=24.0, help="Primary prediction horizon (hours)")
    args = parser.parse_args()

    run_full_phase2b_pipeline(stride=args.stride, primary_horizon=args.horizon)
