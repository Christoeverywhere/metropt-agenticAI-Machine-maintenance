"""
Phase 2 Supervised Training Engine with Class-Weighted BCE Loss,
Early Stopping, Model Checkpointing, and Metric Logging.
"""
import os
import copy
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from src.config import DEVICE, RANDOM_SEED, CHECKPOINT_DIR
    from src.phase2.dataset import create_phase2_dataloader
except ImportError:
    from config import DEVICE, RANDOM_SEED, CHECKPOINT_DIR
    from phase2.dataset import create_phase2_dataloader


def set_seed(seed: int = RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Phase2EarlyStopping:
    """Early stopping to prevent overfitting on imbalanced failure sequences."""

    def __init__(self, patience: int = 4, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("inf")
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def restore_best_weights(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def train_phase2_model(
    model: nn.Module,
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    model_name: str = "LSTM",
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    patience: int = 4,
    checkpoint_dir: Optional[str] = CHECKPOINT_DIR,
    device: torch.device = DEVICE,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Trains a failure prediction model with class-weighted cross-entropy loss."""
    set_seed(RANDOM_SEED)
    model = model.to(device)

    # Calculate balanced positive class weight for imbalanced data (effective sqrt weighting)
    n_pos = float(np.sum(train_labels))
    n_neg = float(len(train_labels) - n_pos)
    pos_weight = float(np.sqrt(n_neg / max(n_pos, 1.0)) * 1.5)
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)

    print(f"\n==================================================================")
    print(f" Training Phase 2 Model: {model_name}")
    print(f" Train: {len(train_windows):,} ({n_pos:.0f} pos / {n_neg:.0f} neg, weight={pos_weight:.2f}x)")
    print(f" Val:   {len(val_windows):,} ({np.sum(val_labels):.0f} pos / {len(val_labels)-np.sum(val_labels):.0f} neg)")
    print(f"==================================================================")

    train_loader = create_phase2_dataloader(train_windows, train_labels, batch_size=batch_size, shuffle=True)
    val_loader = create_phase2_dataloader(val_windows, val_labels, batch_size=batch_size, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopper = Phase2EarlyStopping(patience=patience)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pr_auc": [],
        "val_roc_auc": [],
        "epoch_times": [],
        "learning_rates": [],
    }

    start_train_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # Training
        model.train()
        total_train_loss, train_batches = 0.0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        mean_train_loss = total_train_loss / max(train_batches, 1)

        # Validation
        model.eval()
        total_val_loss, val_batches = 0.0, 0
        val_probs_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                total_val_loss += loss.item()
                val_batches += 1

                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs_list.append(probs)
                val_targets_list.append(batch_y.cpu().numpy())

        mean_val_loss = total_val_loss / max(val_batches, 1)
        val_probs = np.concatenate(val_probs_list).ravel()
        val_targets = np.concatenate(val_targets_list).ravel()

        if len(np.unique(val_targets)) > 1:
            val_pr_auc = float(average_precision_score(val_targets, val_probs))
            val_roc_auc = float(roc_auc_score(val_targets, val_probs))
        else:
            val_pr_auc, val_roc_auc = 0.0, 0.0

        epoch_time = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(mean_val_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["val_roc_auc"].append(val_roc_auc)
        history["epoch_times"].append(epoch_time)
        history["learning_rates"].append(current_lr)

        scheduler.step(mean_val_loss)

        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} | "
            f"Val PR-AUC: {val_pr_auc:.4f} | Val ROC-AUC: {val_roc_auc:.4f} | "
            f"Time: {epoch_time:.2f}s | LR: {current_lr:.2e}"
        )

        if early_stopper(mean_val_loss, model, epoch):
            print(f"[{model_name}] Early stopping triggered at epoch {epoch}. Best epoch was {early_stopper.best_epoch}.")
            break

    # Restore best checkpoint weights
    early_stopper.restore_best_weights(model)
    total_duration = time.perf_counter() - start_train_time

    # Save model checkpoint
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_filename = f"phase2_{model_name.lower().replace(' ', '_')}.pt"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
        torch.save(
            {
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "best_epoch": early_stopper.best_epoch,
                "best_val_loss": early_stopper.best_score,
                "history": history,
                "pos_weight": pos_weight,
            },
            ckpt_path,
        )
        print(f"[{model_name}] Saved Phase 2 checkpoint to: {ckpt_path}")

    summary = {
        "model_name": model_name,
        "total_train_time": total_duration,
        "best_epoch": early_stopper.best_epoch,
        "best_val_loss": early_stopper.best_score,
        "final_val_pr_auc": history["val_pr_auc"][-1],
        "final_val_roc_auc": history["val_roc_auc"][-1],
        "history": history,
        "param_count": sum(p.numel() for p in model.parameters()),
    }
    return model, summary
