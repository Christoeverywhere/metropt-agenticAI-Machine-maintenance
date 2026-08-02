"""
Central configuration for the MetroPT-3 LSTM autoencoder pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "MetroPT3(AirCompressor).csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"
MODEL_CHECKPOINT_PATH = CHECKPOINT_DIR / "lstm_ae.pt"

# ---------------------------------------------------------------------------
# Sensor columns
# Analogue (continuous) sensors — these are what the autoencoder reconstructs.
# The digital signals are kept for context but excluded from reconstruction
# since they are near-binary and would dominate a shared MSE loss.
# ---------------------------------------------------------------------------
ANALOGUE_FEATURES = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

DIGITAL_FEATURES = [
    "COMP",
    "DV_electric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulse",
]

TIMESTAMP_COL = "timestamp"

# ---------------------------------------------------------------------------
# Known failure windows (from the UCI dataset card / Data Description PDF).
# Used ONLY for evaluation and as an exclusion mask during training —
# never as labels fed into the model, since we are training unsupervised.
# ---------------------------------------------------------------------------
KNOWN_FAILURES = [
    {
        "id": 1,
        "start": "2020-04-18 00:00:00",
        "end": "2020-04-18 23:59:00",
        "type": "Air leak",
        "severity": "High stress",
    },
    {
        "id": 2,
        "start": "2020-05-29 23:30:00",
        "end": "2020-05-30 06:00:00",
        "type": "Air leak",
        "severity": "High stress",
    },
    {
        "id": 3,
        "start": "2020-06-05 10:00:00",
        "end": "2020-06-07 14:30:00",
        "type": "Air leak",
        "severity": "High stress",
    },
    {
        "id": 4,
        "start": "2020-07-15 14:30:00",
        "end": "2020-07-15 19:00:00",
        "type": "Air leak",
        "severity": "High stress",
    },
]

# ---------------------------------------------------------------------------
# Windowing / sequence hyperparameters
# Data is logged at 1Hz. A 180-step window = 3 minutes of context per sample.
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 180
SEQUENCE_STRIDE = 30          # 30s stride between overlapping windows during training
INFERENCE_STRIDE = 1          # dense scoring at inference time

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 64
LATENT_SIZE = 16
NUM_LSTM_LAYERS = 1
DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
INITIAL_EPOCHS = 15
INITIAL_LR = 1e-3
INITIAL_BATCH_SIZE = 256

# Incremental (batch-by-batch) fine-tuning
FINETUNE_CHUNK = "7D"         # re-fit on ~weekly chunks of new data as they "arrive"
FINETUNE_EPOCHS = 2           # light touch per chunk — avoid catastrophic forgetting
FINETUNE_LR = 2e-4
FINETUNE_BATCH_SIZE = 256

# A chunk is skipped for fine-tuning (but still scored) if its mean
# reconstruction error is already above this multiple of the running
# healthy-baseline error — protects the model from "learning" a fault as normal.
FINETUNE_SKIP_ERROR_MULTIPLE = 2.5

# ---------------------------------------------------------------------------
# Anomaly thresholding
# Threshold = mean(healthy_val_error) + THRESHOLD_K * std(healthy_val_error)
# ---------------------------------------------------------------------------
THRESHOLD_K = 4.0
ROLLING_SCORE_WINDOW = 60      # smooth per-timestep error over 60s before thresholding
