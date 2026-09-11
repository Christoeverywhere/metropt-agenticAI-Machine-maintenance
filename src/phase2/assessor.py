"""
Combined Phase 1 + Phase 2 Operational Risk Assessor.
Integrates Unsupervised Explainable Anomaly Detection with Supervised Impending Failure Prediction.
"""
from typing import Dict, Any, Optional
import numpy as np

try:
    from src.detector import detect_anomaly, AnomalyDetector
    from src.phase2.predictor import predict_failure, FailurePredictor
except ImportError:
    from detector import detect_anomaly, AnomalyDetector
    from phase2.predictor import predict_failure, FailurePredictor


class RiskAssessor:
    """Integrated Risk Assessor evaluating both current operational anomaly state
    and 24-hour impending failure risk.
    """

    def __init__(
        self,
        detector: Optional[AnomalyDetector] = None,
        predictor: Optional[FailurePredictor] = None,
    ):
        self.detector = detector or AnomalyDetector()
        self.predictor = predictor or FailurePredictor()

    def assess_failure_risk(
        self,
        window: np.ndarray,
        is_raw: bool = True,
    ) -> Dict[str, Any]:
        """Runs integrated assessment across Phase 1 and Phase 2."""
        anomaly_res = self.detector.detect_anomaly(window, is_raw=is_raw)
        failure_res = self.predictor.predict_failure(window, is_raw=is_raw)

        return {
            "anomaly": anomaly_res,
            "failure_prediction": failure_res,
        }


# Global singleton instance
_global_assessor: Optional[RiskAssessor] = None


def get_assessor() -> RiskAssessor:
    global _global_assessor
    if _global_assessor is None:
        _global_assessor = RiskAssessor()
    return _global_assessor


def assess_failure_risk(
    window: np.ndarray,
    is_raw: bool = True,
) -> Dict[str, Any]:
    """Convenience functional API for full Phase 1 + Phase 2 Risk Assessment.

    Returns:
        {
            "anomaly": {
                "anomaly_score": float,
                "threshold": float,
                "is_anomaly": bool,
                "risk_level": str,
                "dominant_sensor": str,
                "sensor_contributions": dict
            },
            "failure_prediction": {
                "failure_probability": float,
                "prediction_horizon_hours": int,
                "threshold": float,
                "is_failure_risk": bool,
                "risk_level": str,
                "model": str
            }
        }
    """
    assessor = get_assessor()
    return assessor.assess_failure_risk(window, is_raw=is_raw)
