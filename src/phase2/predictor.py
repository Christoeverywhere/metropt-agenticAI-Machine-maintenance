"""
Phase 2 Standalone API: Impending Failure Predictor.
"""
import os
import json
from typing import Dict, Any, Optional, Union, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

try:
    from src.config import (
        ANALOGUE_SENSORS,
        FEATURES,
        SEQUENCE_LENGTH,
        CHECKPOINT_DIR,
        REPORTS_DIR,
        DEVICE,
    )
    from src.windowing import load_scaler
    from src.phase2.models import GRUFailurePredictor, LSTMFailurePredictor, LogisticRegressionBaseline
except ImportError:
    from config import (
        ANALOGUE_SENSORS,
        FEATURES,
        SEQUENCE_LENGTH,
        CHECKPOINT_DIR,
        REPORTS_DIR,
        DEVICE,
    )
    from windowing import load_scaler
    from phase2.models import GRUFailurePredictor, LSTMFailurePredictor, LogisticRegressionBaseline


class FailurePredictor:
    """Production-grade Impending Failure Predictor wrapping a trained LSTM/GRU model,
    fitted StandardScaler, and calibrated probability threshold.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        scaler: Optional[StandardScaler] = None,
        threshold: Optional[float] = None,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        model_type: str = "GRU",
        horizon_hours: int = 24,
        sensor_names: List[str] = ANALOGUE_SENSORS,
        device: Union[str, torch.device] = DEVICE,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.sensor_names = sensor_names
        self.seq_len = SEQUENCE_LENGTH
        self.n_features = len(sensor_names)
        self.horizon_hours = horizon_hours
        self.model_type = model_type

        # 1. Scaler
        if scaler is not None:
            self.scaler = scaler
        elif scaler_path and os.path.exists(scaler_path):
            self.scaler = load_scaler(scaler_path)
        else:
            default_scaler_path = os.path.join(CHECKPOINT_DIR, "scaler.pkl")
            if os.path.exists(default_scaler_path):
                self.scaler = load_scaler(default_scaler_path)
            else:
                self.scaler = None

        # 2. Model
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        elif model_path and os.path.exists(model_path):
            self.model = self._load_model_from_checkpoint(model_path, model_type)
        else:
            default_ckpt = os.path.join(CHECKPOINT_DIR, "phase2_gru.pt")
            if os.path.exists(default_ckpt):
                self.model = self._load_model_from_checkpoint(default_ckpt, "GRU")
            else:
                self.model = None

        # 3. Probability threshold
        if threshold is not None:
            self.threshold = float(threshold)
        else:
            threshold_cfg = os.path.join(CHECKPOINT_DIR, "phase2_threshold.json")
            if os.path.exists(threshold_cfg):
                with open(threshold_cfg, "r") as f:
                    data = json.load(f)
                    self.threshold = float(data.get("calibrated_threshold", 0.50))
            else:
                self.threshold = 0.50

    def _load_model_from_checkpoint(self, path: str, model_type: str) -> nn.Module:
        """Instantiates architecture and loads weights from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        if "lstm" in model_type.lower():
            model = LSTMFailurePredictor(input_dim=self.n_features)
        elif "linear" in model_type.lower() or "logistic" in model_type.lower():
            model = LogisticRegressionBaseline(input_dim=self.n_features)
        else:
            model = GRUFailurePredictor(input_dim=self.n_features)

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict_failure(
        self,
        window: np.ndarray,
        is_raw: bool = True,
    ) -> Dict[str, Any]:
        """Predicts the probability of an impending failure occurring within the next 24 hours.

        Args:
            window: (180, 7) or (1, 180, 7) array of sensor observations
            is_raw: True if window contains unscaled physical sensor values

        Returns:
            {
                "failure_probability": float,
                "prediction_horizon_hours": int,
                "threshold": float,
                "is_failure_risk": bool,
                "risk_level": "LOW" | "MODERATE" | "HIGH" | "CRITICAL",
                "model": str
            }
        """
        if self.model is None:
            raise RuntimeError("FailurePredictor model is not initialized or loaded.")

        arr = np.asarray(window, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape != (self.seq_len, self.n_features):
            raise ValueError(
                f"Expected window shape ({self.seq_len}, {self.n_features}), got {arr.shape}"
            )

        if is_raw:
            if self.scaler is None:
                raise RuntimeError("StandardScaler is required to transform raw window.")
            scaled_arr = self.scaler.transform(arr).astype(np.float32)
        else:
            scaled_arr = arr

        tensor_in = torch.from_numpy(scaled_arr).unsqueeze(0).to(self.device)  # (1, T, D)
        logits = self.model(tensor_in)
        prob = float(torch.sigmoid(logits).item())

        is_risk = bool(prob >= self.threshold)

        # Multi-tiered risk categorization
        if prob < 0.30:
            risk_level = "LOW"
        elif prob < self.threshold:
            risk_level = "MODERATE"
        elif prob < 0.80:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return {
            "failure_probability": round(prob, 4),
            "prediction_horizon_hours": self.horizon_hours,
            "threshold": round(self.threshold, 4),
            "is_failure_risk": is_risk,
            "risk_level": risk_level,
            "model": self.model_type,
        }


# Global singleton instance for functional API
_global_predictor: Optional[FailurePredictor] = None


def get_predictor() -> FailurePredictor:
    """Returns or lazily initializes the default Phase 2 FailurePredictor."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = FailurePredictor()
    return _global_predictor


def predict_failure(
    window: np.ndarray,
    model: Optional[nn.Module] = None,
    scaler: Optional[StandardScaler] = None,
    threshold: Optional[float] = None,
    horizon_hours: int = 24,
    is_raw: bool = True,
) -> Dict[str, Any]:
    """Convenience functional API for Phase 2 Impending Failure Prediction."""
    if model is not None or scaler is not None or threshold is not None:
        predictor = FailurePredictor(
            model=model, scaler=scaler, threshold=threshold, horizon_hours=horizon_hours
        )
    else:
        predictor = get_predictor()

    return predictor.predict_failure(window, is_raw=is_raw)
