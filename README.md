# MetroPT-3 LSTM Autoencoder — Predictive Maintenance Pipeline

> Unsupervised anomaly detection for the **MetroPT-3 Air Production Unit (compressor)** using an LSTM Autoencoder trained batch-by-batch: an initial fit on the first month of healthy operating data, followed by incremental fine-tuning as later weekly chunks of the sensor stream "arrive."

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
  - [Data Source](#data-source)
  - [Sensor Schema](#sensor-schema)
  - [Known Failure Events](#known-failure-events)
- [Technical Architecture](#technical-architecture)
  - [Pipeline Overview](#pipeline-overview)
  - [Data Preprocessing](#data-preprocessing)
  - [LSTM Autoencoder Model](#lstm-autoencoder-model)
  - [Two-Stage Training Strategy](#two-stage-training-strategy)
  - [Anomaly Scoring & Thresholding](#anomaly-scoring--thresholding)
  - [Sensor-Level Explainability](#sensor-level-explainability)
- [Hyperparameters Reference](#hyperparameters-reference)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Output Artifacts](#output-artifacts)
- [Tuning Guide](#tuning-guide)
- [Design Decisions & Rationale](#design-decisions--rationale)

---

## Overview

This project implements a **predictive maintenance pipeline** for a metro train compressor using an **LSTM sequence autoencoder**. The core idea is simple: train the model to reconstruct what *normal* sensor behaviour looks like, then flag any future time window where the reconstruction error is abnormally high.

The pipeline is **fully unsupervised** — the 4 confirmed failure events across ~6 months of data are used only for evaluation (measuring detection lead times), never as training labels.

Key capabilities:
- **Batch-by-batch incremental learning** — the model adapts to operational drift without absorbing fault patterns
- **Per-sensor explainability** — every anomaly flag comes with a ranking of which sensors drove it, enabling the downstream agent to generate human-readable justifications
- **Lead-time benchmarking** — detection performance is measured against published results (97 minutes to ~16 hours)

---

## Project Structure

```
metropt-agentic-maintenance/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git exclusion rules
│
├── data/                              # Dataset directory
│   ├── .gitkeep
│   ├── MetroPT3(AirCompressor).csv    # Raw sensor data (~218 MB, ~1.5M rows)
│   └── Data Description_Metro.pdf     # UCI dataset documentation
│
├── checkpoints/                       # Model artifacts (auto-created by train.py)
│   ├── .gitkeep
│   ├── lstm_ae.pt                     # Saved PyTorch model weights
│   ├── scaler.pkl                     # Fitted StandardScaler (pickle)
│   └── scored_stream.csv             # Dense anomaly scores (output of evaluate.py)
│
└── src/                               # Source code
    ├── config.py                      # All paths, hyperparameters, failure windows
    ├── data_loader.py                 # CSV loading, cleaning, train/stream split
    ├── windowing.py                   # Scaling + sliding-window sequence construction
    ├── model.py                       # LSTM autoencoder architecture
    ├── train.py                       # Stage 1 (initial fit) + Stage 2 (incremental)
    └── evaluate.py                    # Threshold calibration, lead-time eval, explanations
```

---

## Dataset

### Data Source

**MetroPT-3 Dataset** from the UCI Machine Learning Repository:

- **Repository URL:** https://archive.ics.uci.edu/dataset/791/metropt+3+dataset
- **Direct download:** https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip
- **Alternative (programmatic):**

```python
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=791)
X = ds.data.features
```

The dataset contains **~1.5 million rows** of 1 Hz sensor readings from a metro train air production unit (compressor), spanning approximately **February 2020 – August 2020**.

### Sensor Schema

The dataset contains **15 sensor columns** plus a timestamp. This pipeline separates them into two groups:

#### Analogue (Continuous) Sensors — *Used for Reconstruction*

| Sensor            | Description                                      |
|-------------------|--------------------------------------------------|
| `TP2`             | Pressure at compressor output (bar)              |
| `TP3`             | Pressure generated at pneumatic panel (bar)      |
| `H1`              | Humidity inside the electrical panel (%)         |
| `DV_pressure`     | Discharge valve pressure (bar)                   |
| `Reservoirs`      | Air tank pressure (bar)                          |
| `Oil_temperature` | Compressor oil temperature (°C)                  |
| `Motor_current`   | Motor current draw (A)                           |

#### Digital (Near-Binary) Sensors — *Excluded from Reconstruction*

| Sensor            | Description                                      |
|-------------------|--------------------------------------------------|
| `COMP`            | Compressor on/off                                |
| `DV_electric`     | Discharge valve electric signal                  |
| `Towers`          | Air dryer tower state                            |
| `MPG`             | Micro-pulse generator                            |
| `LPS`             | Low-pressure switch                              |
| `Pressure_switch` | Pressure switch state                            |
| `Oil_level`       | Oil level indicator                              |
| `Caudal_impulse`  | Flow rate impulse signal                         |

### Known Failure Events

Four documented air-leak failures are used for **evaluation only** (never as training labels):

| # | Start               | End                 | Type     | Severity    |
|---|---------------------|---------------------|----------|-------------|
| 1 | 2020-04-18 00:00:00 | 2020-04-18 23:59:00 | Air leak | High stress |
| 2 | 2020-05-29 23:30:00 | 2020-05-30 06:00:00 | Air leak | High stress |
| 3 | 2020-06-05 10:00:00 | 2020-06-07 14:30:00 | Air leak | High stress |
| 4 | 2020-07-15 14:30:00 | 2020-07-15 19:00:00 | Air leak | High stress |

---

## Technical Architecture

### Pipeline Overview

```
┌─────────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│   Raw CSV        │────▶│  Data Loading &     │────▶│  Scaling & Sliding  │
│  (1.5M rows)     │     │  Train/Stream Split │     │  Window Construction│
└─────────────────┘     └────────────────────┘     └────────┬────────────┘
                                                            │
                               ┌────────────────────────────┘
                               ▼
                   ┌───────────────────────┐
                   │  Stage 1: Initial Fit │  (Month 1 healthy data)
                   │  15 epochs, lr=1e-3   │
                   └───────────┬───────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │  Stage 2: Incremental │  (Remaining ~5 months, weekly chunks)
                   │  Fine-tuning          │
                   │  Score → Skip/Update  │
                   └───────────┬───────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │  Evaluation           │
                   │  Threshold → Scoring  │
                   │  → Lead Times         │
                   │  → Sensor Explanations│
                   └───────────────────────┘
```

### Data Preprocessing

1. **Loading & Cleaning** (`data_loader.py`):
   - Parse timestamps, normalise column names (strip whitespace, replace spaces with underscores)
   - Drop unnamed index columns if present
   - Sort chronologically

2. **Train/Stream Split** (`data_loader.py`):
   - **healthy_train**: First calendar month of data (starting from the first row's timestamp), with any rows inside known failure windows removed as a safety net
   - **stream**: Everything after the first month — walked in weekly chunks for incremental fine-tuning and scoring

3. **Feature Scaling** (`windowing.py`):
   - A `StandardScaler` is fitted **only on healthy training data** (zero-mean, unit-variance normalisation). This defines the model's notion of "normal"
   - All subsequent data (stream chunks, evaluation data) is transformed using this same scaler

4. **Sliding Window Construction** (`windowing.py`):
   - Sensor readings are reshaped into overlapping fixed-length sequences: `(n_sequences, seq_len, n_features)`
   - **Training stride**: 30 steps (30 seconds at 1 Hz) — reduces memory while maintaining overlap
   - **Inference stride**: 1 step — dense scoring for maximum temporal resolution

### LSTM Autoencoder Model

The model (`model.py`) is a **sequence-to-sequence LSTM autoencoder**:

```
Input: (batch, 180, 7)
         │
         ▼
┌─────────────────────┐
│  Encoder LSTM        │  hidden_size=64, num_layers=1
│  (batch, 180, 7)     │──────▶ h_n: (1, batch, 64)
└──────────┬──────────┘
           │ take last hidden state
           ▼
┌─────────────────────┐
│  Linear → Latent     │  64 → 16
│  (batch, 16)         │  ← bottleneck / compressed representation
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Linear → Hidden     │  16 → 64
│  Repeat × 180        │  → (batch, 180, 64)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Decoder LSTM        │  hidden_size=64, num_layers=1
│  (batch, 180, 64)    │──────▶ (batch, 180, 64)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Output Linear       │  64 → 7
│  (batch, 180, 7)     │  ← reconstruction
└─────────────────────┘
```

**Key design choices:**
- The encoder processes the full input sequence and the **last hidden state** is compressed through a linear bottleneck to a 16-dimensional latent vector
- The decoder receives the latent vector repeated across all 180 time steps, forcing it to reconstruct the entire sequence from this compressed representation
- The bottleneck creates an information constraint — normal patterns are learned efficiently, while anomalous patterns produce high reconstruction error

### Two-Stage Training Strategy

#### Stage 1 — Initial Fit (`train.py: initial_fit()`)
- Trains from scratch on the **first month** of healthy data
- **15 epochs**, Adam optimizer with **lr = 1e-3**, batch size **256**
- Loss function: **MSE** between input and reconstructed sequences
- After training, computes a **healthy baseline error** — the mean reconstruction error on the training set itself

#### Stage 2 — Incremental Batch-by-Batch Fine-Tuning (`train.py: incremental_finetune()`)
- Walks the remaining ~5 months in **weekly chunks** (configurable via `FINETUNE_CHUNK`)
- For each arriving chunk:
  1. **Score**: compute the mean reconstruction error with the current model (eval mode, no gradients)
  2. **Decision gate**: if chunk error > `baseline_error × FINETUNE_SKIP_ERROR_MULTIPLE` (default 2.5×):
     - **SKIP** fine-tuning (the chunk looks anomalous — training on it would teach the model that faults are "normal")
     - The chunk is still scored for evaluation purposes
  3. Otherwise: **fine-tune** for 2 epochs at a reduced learning rate (**lr = 2e-4**) to absorb operational drift
- This protects against **catastrophic forgetting** and **concept corruption** simultaneously

### Anomaly Scoring & Thresholding

#### Threshold Calibration (`evaluate.py: calibrate_threshold()`)
- A **held-out validation slice** is carved from the tail of the healthy training month (last 5 days)
- The threshold is set using a **k-sigma rule**:

```
threshold = μ_healthy + k × σ_healthy
```

where `k = 4.0` (configurable via `THRESHOLD_K`), and μ/σ are the mean/std of reconstruction error on the validation slice.

#### Dense Scoring (`evaluate.py: score_dataframe()`)
- The entire post-training stream is scored with **stride = 1** (every single timestep)
- Each sequence's reconstruction error is attributed to its **last timestep** (the most "current" reading — matches how a live stream would be scored)
- A **rolling mean** (window = 60 seconds) smooths per-timestep scores to reduce noise before threshold comparison

#### Lead-Time Evaluation (`evaluate.py: evaluate_lead_times()`)
- For each of the 4 known failures, the evaluator finds the **first threshold-crossing** within 2 days before the failure's documented start
- The lead time = `failure_start - first_flag_timestamp`
- This is benchmarked against published results for this dataset (~97 minutes to ~16 hours)

### Sensor-Level Explainability

The model provides **per-feature reconstruction error** without needing any external explainability tool (no SHAP/LIME):

```python
per_feature_error = mean((x - x̂)², dim=time)   # → (batch, n_features)
```

For any flagged timestamp, the `top_contributing_sensors()` function ranks the 7 analogue sensors by their individual reconstruction error, producing explanations like:

> *"Flagged due to rising Motor_current and Oil_temperature reconstruction error"*

This is directly usable by a downstream scheduling/decision agent.

---

## Hyperparameters Reference

All hyperparameters are centralised in `src/config.py`:

### Windowing

| Parameter          | Value | Description                                          |
|--------------------|-------|------------------------------------------------------|
| `SEQUENCE_LENGTH`  | 180   | Window length in timesteps (= 3 minutes at 1 Hz)    |
| `SEQUENCE_STRIDE`  | 30    | Stride between training windows (30 seconds)         |
| `INFERENCE_STRIDE` | 1     | Stride at inference (dense, every second)            |

### Model Architecture

| Parameter         | Value | Description                        |
|-------------------|-------|------------------------------------|
| `HIDDEN_SIZE`     | 64    | LSTM hidden dimension              |
| `LATENT_SIZE`     | 16    | Bottleneck latent dimension        |
| `NUM_LSTM_LAYERS` | 1     | Number of stacked LSTM layers      |
| `DROPOUT`         | 0.1   | Dropout rate (only if layers > 1)  |

### Stage 1 Training

| Parameter              | Value | Description              |
|------------------------|-------|--------------------------|
| `INITIAL_EPOCHS`       | 15    | Training epochs          |
| `INITIAL_LR`           | 1e-3  | Adam learning rate       |
| `INITIAL_BATCH_SIZE`   | 256   | Mini-batch size          |

### Stage 2 Incremental Fine-Tuning

| Parameter                       | Value  | Description                                           |
|---------------------------------|--------|-------------------------------------------------------|
| `FINETUNE_CHUNK`                | `"7D"` | Chunk frequency (weekly)                              |
| `FINETUNE_EPOCHS`               | 2      | Epochs per chunk                                      |
| `FINETUNE_LR`                   | 2e-4   | Reduced learning rate to prevent catastrophic forgetting |
| `FINETUNE_BATCH_SIZE`           | 256    | Mini-batch size                                       |
| `FINETUNE_SKIP_ERROR_MULTIPLE`  | 2.5    | Skip threshold multiplier over baseline error         |

### Anomaly Detection

| Parameter              | Value | Description                                                |
|------------------------|-------|------------------------------------------------------------|
| `THRESHOLD_K`          | 4.0   | k-sigma multiplier for threshold calibration               |
| `ROLLING_SCORE_WINDOW` | 60    | Rolling average window (seconds) for smoothing scores      |

---

## Getting Started

### Prerequisites

- **Python** ≥ 3.9
- **CUDA** (optional) — the pipeline auto-detects GPU availability; runs on CPU otherwise

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd metropt-agentic-maintenance

# Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):
| Package       | Version  | Purpose                                   |
|---------------|----------|-------------------------------------------|
| `torch`       | ≥ 2.1   | LSTM autoencoder, training loop, GPU accel |
| `pandas`      | ≥ 2.0   | Data loading, time-series manipulation     |
| `numpy`       | ≥ 1.24  | Array operations, sequence construction    |
| `scikit-learn` | ≥ 1.3  | `StandardScaler` for feature normalisation |
| `ucimlrepo`   | ≥ 0.0.7 | (Optional) Programmatic dataset download   |

### Data Setup

Download the dataset and place the CSV in the `data/` directory:

```
data/MetroPT3(AirCompressor).csv
```

### Training

```bash
python src/train.py
```

**What happens:**
1. Loads and cleans the raw CSV (~1.5M rows)
2. Splits into healthy training set (month 1) and stream (months 2–6)
3. **Stage 1**: Fits the autoencoder on healthy data (15 epochs)
4. Computes healthy baseline reconstruction error
5. **Stage 2**: Walks the stream in weekly chunks, scoring and selectively fine-tuning
6. Saves `checkpoints/lstm_ae.pt` and `checkpoints/scaler.pkl`

**Expected console output:**
```
[stage 1] healthy_train rows: ~2,600,000
[stage 1] training sequences: (N, 180, 7)
[stage 1] epoch 1/15 — train MSE: 0.XXXXXX
...
[baseline] healthy reconstruction error: 0.XXXXXX
[stage 2] streaming N rows in '7D' chunks
[stage 2] 2020-03-XX → 2020-03-XX: mean err 0.XXXXXX — fine-tuned (2 epochs)
...
```

### Evaluation

```bash
python src/evaluate.py
```

**What happens:**
1. Loads the saved model and scaler from `checkpoints/`
2. Calibrates an anomaly threshold from a held-out validation slice (last 5 days of month 1)
3. Densely scores the full stream (stride = 1)
4. Reports **detection lead time** for each of the 4 known failures
5. Reports **top contributing sensors** for each failure
6. Saves all scores to `checkpoints/scored_stream.csv`

**Expected console output:**
```
[threshold] healthy mean=0.XXXXXX std=0.XXXXXX -> threshold=0.XXXXXX (k=4.0)
[scoring] dense pass over the full stream (this can take a while)...

=== Detection lead time vs. known failures ===
Failure #1 (2020-04-18 00:00:00): first flagged at ... — lead time = ...
...

=== Example explanation for each detected failure's first flag ===
Failure #1 near 2020-04-18 00:00:00: top sensors -> [('Motor_current_error', ...), ...]
```

---

## Output Artifacts

| File                            | Format  | Description                                                      |
|---------------------------------|---------|------------------------------------------------------------------|
| `checkpoints/lstm_ae.pt`        | PyTorch | Model state dict (encoder + decoder weights)                     |
| `checkpoints/scaler.pkl`        | Pickle  | Fitted `StandardScaler` — needed for inference on new data       |
| `checkpoints/scored_stream.csv` | CSV     | Per-timestep anomaly scores + per-sensor errors for the full stream |

**`scored_stream.csv` columns:**

| Column                | Description                                           |
|-----------------------|-------------------------------------------------------|
| `timestamp`           | Timestamp of the scored point                         |
| `seq_error`           | Scalar anomaly score (mean MSE across all features)   |
| `TP2_error`           | Per-feature reconstruction error for TP2              |
| `TP3_error`           | Per-feature reconstruction error for TP3              |
| `H1_error`            | Per-feature reconstruction error for H1               |
| `DV_pressure_error`   | Per-feature reconstruction error for DV_pressure      |
| `Reservoirs_error`    | Per-feature reconstruction error for Reservoirs       |
| `Oil_temperature_error` | Per-feature reconstruction error for Oil_temperature |
| `Motor_current_error` | Per-feature reconstruction error for Motor_current    |

---

## Tuning Guide

| What to change                   | Where                           | Suggestions                                                                                         |
|----------------------------------|---------------------------------|-----------------------------------------------------------------------------------------------------|
| Window length                    | `SEQUENCE_LENGTH`               | Shorter (60) = faster but less temporal context. Longer (300) = more context but more memory        |
| Threshold sensitivity            | `THRESHOLD_K`                   | Lower k (2–3) = more sensitive (more false positives). Higher k (5+) = fewer false positives        |
| Score smoothing                  | `ROLLING_SCORE_WINDOW`          | Larger window = smoother scores, delayed detection. Smaller = noisier but faster response           |
| Fine-tuning frequency            | `FINETUNE_CHUNK`                | `"1D"` for daily drift adaptation. `"14D"` if data is very stable                                   |
| Skip anomalous chunks            | `FINETUNE_SKIP_ERROR_MULTIPLE`  | Higher = more permissive (may learn mild faults). Lower = more conservative                         |
| Latent bottleneck                | `LATENT_SIZE`                   | Smaller = stronger compression, may miss subtle patterns. Larger = less constraint                  |
| Add digital sensors as context   | `config.py` / `model.py`        | Feed digital sensors as **input only** (not reconstruction target) to condition the encoder on compressor state |

---

## Design Decisions & Rationale

### Why Exclude Digital Sensors from Reconstruction?

The digital sensors (`COMP`, `DV_electric`, etc.) are near-binary signals. Including them in a shared MSE loss lets the noisier binary transitions dominate the loss, drowning out subtle changes in the continuous sensors that are more indicative of emerging faults. They can still be included as **encoder inputs** without being reconstructed.

### Why Unsupervised?

With only **4 confirmed failure events** across ~6 months, there is nowhere near enough labelled data for supervised classification. The autoencoder approach sidesteps this entirely: learn what *normal* looks like, and flag anything that deviates.

### Why Incremental Fine-Tuning with a Skip Gate?

Real-world compressor behaviour drifts over time (seasonal temperature changes, wear, load variation). A static model trained on month 1 would accumulate drift-induced false positives. The incremental strategy adapts to drift while the skip gate (error > 2.5× baseline → don't train) prevents the model from absorbing fault patterns as "normal."

### Why Attribute Error to the Last Timestep?

In a live deployment, you have data up to "now." A sequence ending at time *t* represents the model's assessment of the most recent 3-minute window. Attributing the score to *t* (the last step) matches the real-time scoring semantics where you'd act on the most current reading.

---

## Tech Stack

- **Language**: Python 3.9+
- **Deep Learning**: PyTorch ≥ 2.1
- **Data Processing**: pandas ≥ 2.0, NumPy ≥ 1.24
- **Preprocessing**: scikit-learn ≥ 1.3 (`StandardScaler`)
- **Data Acquisition**: ucimlrepo ≥ 0.0.7
