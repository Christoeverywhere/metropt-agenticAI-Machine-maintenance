"""
Phase 1 Explainable Anomaly Detection API for MetroPT-3 Compressor.

Provides:
  - AnomalyDetector class for high-throughput and real-time streaming inference
  - detect_anomaly(window) functional interface
  - Sequence and sensor-level error decomposition
  - Dynamic sensor attribution and dominant sensor identification
  - Multi-tier risk categorization (NORMAL, WARNING, CRITICAL)
"""
import os
from typing import Dict, Any, Optional, Tuple, Union, List

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
        DEVICE,
    )
    from src.windowing import load_scaler
    from src.models.attention_lstm_ae import AttentionDenoisingLSTMAutoencoder
    from src.models.lstm_ae import LSTMAutoencoder
    from src.models.dense_ae import DenseAutoencoder
except ImportError:
    from config import (
        ANALOGUE_SENSORS,
        FEATURES,
        SEQUENCE_LENGTH,
        CHECKPOINT_DIR,
        DEVICE,
    )
    from windowing import load_scaler
    from models.attention_lstm_ae import AttentionDenoisingLSTMAutoencoder
    from models.lstm_ae import LSTMAutoencoder
    from models.dense_ae import DenseAutoencoder


class AnomalyDetector:
    """Production-grade Anomaly Detector wrapping a trained Autoencoder model,
    fitted StandardScaler, and calibrated normal validation threshold.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        scaler: Optional[StandardScaler] = None,
        threshold: Optional[float] = None,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        model_type: str = "attn_lstm",
        sensor_names: List[str] = ANALOGUE_SENSORS,
        device: Union[str, torch.device] = DEVICE,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.sensor_names = sensor_names
        self.seq_len = SEQUENCE_LENGTH
        self.n_features = len(sensor_names)

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
            default_ckpt = os.path.join(CHECKPOINT_DIR, "attn_+_denoising_lstm_ae.pt")
            if os.path.exists(default_ckpt):
                self.model = self._load_model_from_checkpoint(default_ckpt, "attn_lstm")
            else:
                self.model = None

        # 3. Calibrated threshold
        self.threshold = float(threshold) if threshold is not None else 1.0

    def _load_model_from_checkpoint(self, path: str, model_type: str) -> nn.Module:
        """Instantiates architecture and loads weights from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        if "attn" in model_type.lower() or "attention" in path.lower():
            model = AttentionDenoisingLSTMAutoencoder(
                n_features=self.n_features, seq_len=self.seq_len, latent_size=16
            )
        elif "dense" in model_type.lower() or "dense" in path.lower():
            model = DenseAutoencoder(
                n_features=self.n_features, seq_len=self.seq_len, latent_size=16
            )
        else:
            model = LSTMAutoencoder(
                n_features=self.n_features, seq_len=self.seq_len, latent_size=16
            )

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    def set_threshold(self, threshold: float) -> None:
        """Sets the decision threshold."""
        self.threshold = float(threshold)

    @torch.no_grad()
    def score_window(
        self,
        window: np.ndarray,
        is_raw: bool = True,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Calculates sequence reconstruction MSE and per-sensor errors for a single window.

        Args:
            window: Array of shape (seq_len, n_features) or (1, seq_len, n_features)
            is_raw: If True, applies fitted StandardScaler before model inference

        Returns:
            seq_error: Scalar sequence MSE
            sensor_errors: (n_features,) array of MSE per sensor
            recon: Reconstructed window (seq_len, n_features)
        """
        if self.model is None:
            raise RuntimeError("AnomalyDetector model is not initialized or loaded.")

        arr = np.asarray(window, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]  # Take (seq_len, n_features)
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
        recon_tensor = self.model(tensor_in)  # (1, T, D)

        sq_diff = (tensor_in - recon_tensor) ** 2  # (1, T, D)
        seq_error = float(sq_diff.mean().item())
        sensor_errors = sq_diff.mean(dim=(0, 1)).cpu().numpy()  # (D,)
        recon_np = recon_tensor.squeeze(0).cpu().numpy()

        return seq_error, sensor_errors, recon_np

    def detect_anomaly(
        self,
        window: np.ndarray,
        is_raw: bool = True,
    ) -> Dict[str, Any]:
        """Core Phase 1 Inference API. Evaluates a single 180x7 sensor window.

        Returns:
            {
                "anomaly_score": float,
                "threshold": float,
                "is_anomaly": bool,
                "risk_level": "NORMAL" | "WARNING" | "CRITICAL",
                "dominant_sensor": str,
                "sensor_contributions": {
                    "DV_pressure": float (pct),
                    "H1": float (pct),
                    ...
                }
            }
        """
        seq_error, sensor_errors, _ = self.score_window(window, is_raw=is_raw)

        is_anomaly = bool(seq_error > self.threshold)

        # Multi-tiered operational risk classification
        if seq_error <= self.threshold:
            risk_level = "NORMAL"
        elif seq_error <= 1.5 * self.threshold:
            risk_level = "WARNING"
        else:
            risk_level = "CRITICAL"

        # Normalized sensor contributions (%)
        total_sensor_err = float(np.sum(sensor_errors))
        sensor_contrib = {}
        for idx, sensor_name in enumerate(self.sensor_names):
            pct = float((sensor_errors[idx] / max(total_sensor_err, 1e-8)) * 100.0)
            sensor_contrib[sensor_name] = round(pct, 2)

        dominant_sensor = max(sensor_contrib.items(), key=lambda x: x[1])[0]

        return {
            "anomaly_score": round(seq_error, 6),
            "threshold": round(self.threshold, 6),
            "is_anomaly": is_anomaly,
            "risk_level": risk_level,
            "dominant_sensor": dominant_sensor,
            "sensor_contributions": sensor_contrib,
        }


# Global singleton detector instance for functional usage
_global_detector: Optional[AnomalyDetector] = None


def get_detector() -> AnomalyDetector:
    """Returns or lazily initializes the default Phase 1 AnomalyDetector."""
    global _global_detector
    if _global_detector is None:
        _global_detector = AnomalyDetector()
    return _global_detector


def detect_anomaly(
    window: np.ndarray,
    model: Optional[nn.Module] = None,
    scaler: Optional[StandardScaler] = None,
    threshold: Optional[float] = None,
    is_raw: bool = True,
) -> Dict[str, Any]:
    """Convenience functional API for Phase 1 Anomaly Detection.

    Args:
        window: (180, 7) array of sensor observations
        model: Optional trained model override
        scaler: Optional fitted StandardScaler override
        threshold: Optional decision threshold override
        is_raw: True if window contains unscaled raw physical measurements

    Returns:
        Structured dictionary matching Phase 1 output specification.
    """
    if model is not None or scaler is not None or threshold is not None:
        detector = AnomalyDetector(model=model, scaler=scaler, threshold=threshold)
    else:
        detector = get_detector()

    return detector.detect_anomaly(window, is_raw=is_raw)
