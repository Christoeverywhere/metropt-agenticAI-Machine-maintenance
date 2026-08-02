# MetroPT-3 LSTM Autoencoder — Predictive Maintenance Pipeline

Unsupervised anomaly detection for the MetroPT-3 Air Production Unit (compressor)
dataset, trained batch-by-batch: an initial fit on the first month of healthy
data, followed by incremental fine-tuning as later chunks of the stream "arrive."

## Project Structure

```
metropt_pipeline/
├── README.md                      # setup + usage instructions
├── requirements.txt               # torch, pandas, numpy, scikit-learn, ucimlrepo
├── data/                          # <- drop the CSV here
│   └── MetroPT3(AirCompressor).csv
├── checkpoints/                   # <- created automatically by train.py
│   ├── lstm_ae.pt                 # saved model weights
│   ├── scaler.pkl                 # fitted StandardScaler
│   └── scored_stream.csv          # output of evaluate.py
└── src/
    ├── config.py                  # all paths, hyperparameters, known failure windows
    ├── data_loader.py             # CSV loading, cleaning, train/stream split
    ├── windowing.py               # scaling + sliding-window sequence construction
    ├── model.py                   # LSTM autoencoder architecture
    ├── train.py                   # stage 1 (initial fit) + stage 2 (incremental fine-tune)
    └── evaluate.py                # threshold calibration, lead-time eval, sensor explanations
```

## 1. Get the data

Download from the UCI repository:
https://archive.ics.uci.edu/dataset/791/metropt+3+dataset

Direct zip: https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip

Unzip it and place `MetroPT3(AirCompressor).csv` into `data/`, so you have:

```
data/MetroPT3(AirCompressor).csv
```

Alternatively, fetch it programmatically:

```python
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=791)
X = ds.data.features
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Train

```bash
python src/train.py
```

This runs:
- **Stage 1** — fits the LSTM autoencoder from scratch on the first calendar
  month of data (recommended split per the UCI dataset card).
- **Stage 2** — walks the remaining ~5 months in weekly chunks. Each chunk is
  scored first; only chunks that still look "healthy" relative to the running
  baseline are used to lightly fine-tune the model. Chunks that already look
  anomalous are scored but skipped for fine-tuning, so the model never learns
  to treat a fault as normal.

Outputs: `checkpoints/lstm_ae.pt` (model weights) and `checkpoints/scaler.pkl`
(the fitted StandardScaler).

## 4. Evaluate

```bash
python src/evaluate.py
```

This:
- Calibrates an anomaly threshold from a held-out slice of healthy data.
- Densely scores the full post-training-month stream.
- For each of the 4 documented failures, reports the **lead time** — how long
  before the failure's recorded start time the model first flagged an anomaly.
  Compare this against the published lead times for this dataset
  (~97 minutes to ~16 hours) to see where your model lands.
- For each failure, prints the **top contributing sensors** by reconstruction
  error — this is the per-feature explanation your agent can use to justify
  a decision ("flagged due to rising Motor_current and Oil_temperature error").

Results are saved to `checkpoints/scored_stream.csv` for further analysis
or for feeding into the downstream scheduling agent.

## Notes / things to tune

- `src/config.py` holds every hyperparameter — window length, stride, hidden
  sizes, thresholding, fine-tuning cadence, etc.
- The digital (near-binary) sensors are excluded from the reconstruction
  target on purpose — mixing them with continuous sensors in one MSE loss
  lets the noisier binary signals dominate the loss. You can still bring
  them back in later as extra *input* context without reconstructing them,
  if you want the encoder to condition on compressor state.
- At ~1.5M rows total, the full dense scoring pass in `evaluate.py` can be
  slow on CPU — consider scoring a date-range subset first while iterating.
- This is unsupervised: only 4 confirmed failure events exist across ~6
  months, nowhere near enough for supervised classification. The autoencoder
  is trained to reconstruct *normal* behaviour; error magnitude is the
  anomaly signal, evaluated against those 4 known windows.
