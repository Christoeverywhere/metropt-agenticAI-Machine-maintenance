"""
Threshold Calibration Strategies Derived from Normal Validation Reconstruction Errors.
"""
from typing import Dict, Any
import numpy as np


def calibrate_parametric(val_errors: np.ndarray, k: float = 3.0) -> float:
    """Computes mu + k * sigma threshold on normal validation errors."""
    mean = float(np.mean(val_errors))
    std = float(np.std(val_errors))
    return mean + k * std


def calibrate_percentile(val_errors: np.ndarray, percentile: float = 99.0) -> float:
    """Computes non-parametric percentile threshold on normal validation errors."""
    return float(np.percentile(val_errors, percentile))


def calibrate_iqr(val_errors: np.ndarray, factor: float = 1.5) -> float:
    """Computes Tukey's fence IQR threshold: Q3 + factor * (Q3 - Q1)."""
    q75, q25 = np.percentile(val_errors, [75, 25])
    iqr = q75 - q25
    return float(q75 + factor * iqr)


def calibrate_thresholds_suite(val_errors: np.ndarray) -> Dict[str, float]:
    """Computes all standard thresholding strategies for comparative analysis."""
    mean = float(np.mean(val_errors))
    std = float(np.std(val_errors))

    return {
        "mu+2sigma": mean + 2.0 * std,
        "mu+3sigma": mean + 3.0 * std,
        "mu+4sigma": mean + 4.0 * std,
        "p95.0": float(np.percentile(val_errors, 95.0)),
        "p97.5": float(np.percentile(val_errors, 97.5)),
        "p99.0": float(np.percentile(val_errors, 99.0)),
        "p99.5": float(np.percentile(val_errors, 99.5)),
        "p99.9": float(np.percentile(val_errors, 99.9)),
        "iqr_1.5": calibrate_iqr(val_errors, factor=1.5),
        "iqr_3.0": calibrate_iqr(val_errors, factor=3.0),
    }
