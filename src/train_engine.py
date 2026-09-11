"""
Unified Training Engine with Early Stopping, Model Checkpointing, and Denoising Support.
"""
import os
import copy
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from src.config import DEVICE, RANDOM_SEED
    from src.dataset import create_dataloader
except ImportError:
    from config import DEVICE, RANDOM_SEED
    from dataset import create_dataloader


def set_seed(seed: int = RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""

    def __init__(self, patience: int = 4, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
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


def train_model(
    model: nn.Module,
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    model_name: str = "Model",
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 256,
    noise_std: float = 0.0,
    mask_prob: float = 0.0,
    patience: int = 4,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Trains a model with clean/denoising objective, early stopping, and metric logging."""
    set_seed(RANDOM_SEED)
    model = model.to(DEVICE)

    # Data loaders
    train_loader = create_dataloader(
        train_windows,
        batch_size=batch_size,
        shuffle=True,
        noise_std=noise_std,
        mask_prob=mask_prob,
        is_training=True,
    )
    val_loader = create_dataloader(
        val_windows,
        batch_size=batch_size,
        shuffle=False,
        noise_std=0.0,
        mask_prob=0.0,
        is_training=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopping(patience=patience)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "epoch_times": [],
        "learning_rates": [],
    }

    print(f"\n=======================================================")
    print(f" Training {model_name} (Noise sigma={noise_std}, Mask p={mask_prob})")
    print(f" Train Windows: {len(train_windows):,} | Val Windows: {len(val_windows):,}")
    print(f"=======================================================")

    start_train_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # Training phase
        model.train()
        total_train_loss, train_batches = 0.0, 0
        for batch_in, batch_target in train_loader:
            batch_in = batch_in.to(DEVICE)
            batch_target = batch_target.to(DEVICE)

            optimizer.zero_grad()
            recon = model(batch_in)
            loss = loss_fn(recon, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        mean_train_loss = total_train_loss / max(train_batches, 1)

        # Validation phase (pure clean inputs)
        model.eval()
        total_val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch_in, batch_target in val_loader:
                batch_in = batch_in.to(DEVICE)
                batch_target = batch_target.to(DEVICE)

                recon = model(batch_in)
                loss = loss_fn(recon, batch_target)
                total_val_loss += loss.item()
                val_batches += 1

        mean_val_loss = total_val_loss / max(val_batches, 1)
        epoch_time = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(mean_val_loss)
        history["epoch_times"].append(epoch_time)
        history["learning_rates"].append(current_lr)

        scheduler.step(mean_val_loss)

        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | "
            f"Train MSE: {mean_train_loss:.6f} | Val MSE: {mean_val_loss:.6f} | "
            f"Time: {epoch_time:.2f}s | LR: {current_lr:.2e}"
        )

        if early_stopper(mean_val_loss, model, epoch):
            print(f"[{model_name}] Early stopping triggered at epoch {epoch}. Best epoch was {early_stopper.best_epoch}.")
            break

    # Restore best checkpoint weights
    early_stopper.restore_best_weights(model)
    total_duration = time.perf_counter() - start_train_time

    # Save checkpoint if requested
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_filename = f"{model_name.lower().replace(' ', '_')}.pt"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
        torch.save(
            {
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "best_epoch": early_stopper.best_epoch,
                "best_val_loss": early_stopper.best_loss,
                "history": history,
            },
            ckpt_path,
        )
        print(f"[{model_name}] Saved best model checkpoint to: {ckpt_path}")

    summary = {
        "model_name": model_name,
        "total_train_time": total_duration,
        "best_epoch": early_stopper.best_epoch,
        "best_val_loss": early_stopper.best_loss,
        "final_train_loss": history["train_loss"][-1],
        "avg_epoch_time": float(np.mean(history["epoch_times"])),
        "history": history,
        "param_count": sum(p.numel() for p in model.parameters()),
    }
    return model, summary
