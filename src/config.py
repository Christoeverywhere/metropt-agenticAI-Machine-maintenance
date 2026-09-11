"""
Central configuration: paths, hyperparameters, sensor schema, known failures.
Every other module imports from here — no magic numbers/strings elsewhere.
"""
import os

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    # PyTorch is required to actually train/run the model (see train.py /
    # evaluate.py / model.py). It is NOT required for data_loader.py or
    # windowing.py, which only depend on pandas/numpy/scikit-learn. This
    # fallback lets those modules (and this config) be imported/tested in
    # a PyTorch-less environment without crashing.
    _TORCH_AVAILABLE = False

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

RAW_CSV_PATH = os.path.join(DATA_DIR, "MetroPT3(AirCompressor).csv")

MODEL_PATH = os.path.join(CHECKPOINT_DIR, "lstm_ae.pt")
SCALER_PATH = os.path.join(CHECKPOINT_DIR, "scaler.pkl")
SCORED_STREAM_PATH = os.path.join(CHECKPOINT_DIR, "scored_stream.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Sensor schema
# --------------------------------------------------------------------------
# NOTE: the real CSV header has a couple of quirks vs. the tidy names used
# in documentation - normalize on load in data_loader.py, but the *names
# we standardize to* are defined here and used everywhere downstream.
#
# Raw header quirks observed in data/MetroPT3(AirCompressor).csv:
#   - leading unnamed index column            -> dropped
#   - "DV_eletric"      (typo, missing "c")   -> renamed to DV_electric
#   - "Caudal_impulses" (plural)              -> renamed to Caudal_impulse

RAW_TO_STANDARD_COLUMN_MAP = {
    "DV_eletric": "DV_electric",
    "Caudal_impulses": "Caudal_impulse",
}

TIMESTAMP_COL = "timestamp"

# 7 continuous sensors used for reconstruction (encoder + decoder target)
FEATURES = [
    "DV_pressure",
    "H1",
    "TP2",
    "TP3",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature",
]
ANALOGUE_SENSORS = FEATURES

# 8 near-binary sensors excluded from the reconstruction loss
DIGITAL_SENSORS = [
    "COMP",
    "DV_electric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulse",
]

ALL_SENSORS = ANALOGUE_SENSORS + DIGITAL_SENSORS
N_FEATURES = len(ANALOGUE_SENSORS)  # 7

# --------------------------------------------------------------------------
# Known failure events (evaluation only - never used as training labels)
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Windowing / sequence hyperparameters
# --------------------------------------------------------------------------
SEQUENCE_LENGTH = 180        # 3 minutes of 1Hz-equivalent samples per window
SEQUENCE_STRIDE = 30         # stride between windows during training
INFERENCE_STRIDE = 1         # dense stride during evaluation/scoring

# --------------------------------------------------------------------------
# Model hyperparameters
# --------------------------------------------------------------------------
HIDDEN_SIZE = 64
LATENT_SIZE = 16
NUM_LSTM_LAYERS = 1
DROPOUT = 0.1

# --------------------------------------------------------------------------
# Stage 1: initial fit
# --------------------------------------------------------------------------
INITIAL_EPOCHS = 15
INITIAL_LR = 1e-3
INITIAL_BATCH_SIZE = 256

# --------------------------------------------------------------------------
# Stage 2: incremental fine-tuning
# --------------------------------------------------------------------------
FINETUNE_CHUNK = "7D"                  # pandas offset alias: 7-day chunks
FINETUNE_EPOCHS = 2
FINETUNE_LR = 2e-4
FINETUNE_BATCH_SIZE = 256
FINETUNE_SKIP_ERROR_MULTIPLE = 2.5     # skip fine-tuning if chunk err > 2.5x baseline

# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------
THRESHOLD_K = 4.0             # k-sigma multiplier
ROLLING_SCORE_WINDOW = 60     # seconds, for smoothing dense scores

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
RANDOM_SEED = 42

if _TORCH_AVAILABLE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = "cpu"  # placeholder; real training requires PyTorch installed
