
"""
Task 10 — Experiment 2: Tail-Weighted U-Net

Run from project root:
    python src/experiments/task10_train_unet_weighted.py

Differences from original U-Net:
- Same architecture.
- Same training and validation patches.
- Same target normalisation.
- Same optimiser and scheduler.
- Weighted Huber loss based on TRAINING target quantiles.
- Separate checkpoint and output files.

Test split is never loaded.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PATCH_DIR = ROOT / "data" / "processed" / "task10_patches"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs" / "tables"

TRAIN_FILE = PATCH_DIR / "train.npz"
VAL_FILE = PATCH_DIR / "validation.npz"

BEST_MODEL_FILE = MODEL_DIR / "task10_unet_weighted_best.pt"
HISTORY_FILE = OUTPUT_DIR / "10_unet_weighted_training_history.csv"
METRICS_FILE = OUTPUT_DIR / "10_unet_weighted_validation_metrics.json"

SEED = 42

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 15

INPUT_CHANNELS = 10
BASE_CHANNELS = 16
PATCH_SIZE = 16

TARGET_MEAN = 270.157498
TARGET_STD = 113.758504

NUM_WORKERS = 0

# Tail weighting experiment.
# Weights correspond to training-target quintiles.
QUANTILE_WEIGHTS = [1.5, 1.0, 0.75, 0.75, 1.5]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. LOGGING AND REPRODUCIBILITY
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger("task10_weighted_unet")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. DATASET
# ============================================================

class FloodPatchDataset(Dataset):

    def __init__(self, file_path: Path, augment=False):

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        logger.info("Loading %s", file_path.name)

        with np.load(file_path) as data:
            self.X = np.asarray(data["X"], dtype=np.float32)
            self.y = np.asarray(data["y"], dtype=np.float32)
            self.mask = np.asarray(data["mask"], dtype=np.float32)

        if self.X.ndim != 4 or self.X.shape[1] != INPUT_CHANNELS:
            raise ValueError(f"Invalid feature shape: {self.X.shape}")

        if self.y.shape != self.mask.shape:
            raise ValueError("Target and mask shapes differ.")

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Patch counts differ.")

        if self.X.shape[2:] != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Unexpected patch dimensions: {self.X.shape}")

        if not np.isfinite(self.X).all():
            raise ValueError("Features contain non-finite values.")

        if not np.isfinite(self.y).all():
            raise ValueError("Targets contain non-finite values.")

        self.augment = augment

        logger.info(
            "%s: %d patches, %d supervised pixels",
            file_path.name,
            len(self.X),
            int(self.mask.sum()),
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        x = torch.from_numpy(self.X[index].copy())
        y = torch.from_numpy(self.y[index].copy())
        mask = torch.from_numpy(self.mask[index].copy())

        if self.augment:

            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
                mask = torch.flip(mask, dims=[2])

            if random.random() < 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
                mask = torch.flip(mask, dims=[1])

            rotations = random.randint(0, 3)

            if rotations:
                x = torch.rot90(x, rotations, dims=[1, 2])
                y = torch.rot90(y, rotations, dims=[1, 2])
                mask = torch.rot90(mask, rotations, dims=[1, 2])

        return x, y, mask


# ============================================================
# 4. U-NET ARCHITECTURE
# Identical to the original model.
# ============================================================

def group_count(channels):

    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups

    return 1


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        groups = group_count(out_channels)

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self, in_channels=10, base_channels=16):

        super().__init__()

        b = base_channels

        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(b * 4, b * 8)

        self.up3 = nn.ConvTranspose2d(
            b * 8, b * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(b * 8, b * 4)

        self.up2 = nn.ConvTranspose2d(
            b * 4, b * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(b * 4, b * 2)

        self.up1 = nn.ConvTranspose2d(
            b * 2, b, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(b * 2, b)

        self.output = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        bottleneck = self.bottleneck(self.pool(e3))

        d3 = self.up3(bottleneck)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# 5. TRAINING-ONLY QUANTILE BOUNDARIES
# ============================================================

def calculate_training_quantiles(train_dataset):

    logger.info("Calculating target quantiles from training cells only.")

    values = []

    for i in range(len(train_dataset)):
        target = train_dataset.y[i, 0]
        mask = train_dataset.mask[i, 0].astype(bool)

        values.append(target[mask])

    values = np.concatenate(values)

    # Targets are stored normalised in the NPZ files.
    # Quantiles are invariant under this positive linear scaling.
    boundaries = np.quantile(
        values,
        [0.20, 0.40, 0.60, 0.80],
    )

    logger.info(
        "Training target quintile boundaries (normalised): %s",
        np.round(boundaries, 4).tolist(),
    )

    return boundaries.astype(np.float32)


# ============================================================
# 6. TAIL-WEIGHTED HUBER LOSS
# ============================================================

class TailWeightedHuberLoss(nn.Module):

    def __init__(self, boundaries, weights):

        super().__init__()

        if len(weights) != 5:
            raise ValueError("Exactly five quantile weights are required.")

        self.register_buffer(
            "boundaries",
            torch.as_tensor(boundaries, dtype=torch.float32),
        )

        self.register_buffer(
            "quantile_weights",
            torch.as_tensor(weights, dtype=torch.float32),
        )

    def forward(self, prediction, target, mask):

        # Assign each valid target to a training-derived quintile.
        # torch.bucketize returns indices 0 through 4.
        quantile_index = torch.bucketize(
            target.contiguous(),
            self.boundaries,
        )

        pixel_weights = self.quantile_weights[quantile_index]

        element_loss = F.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
            beta=1.0,
        )

        weighted_mask = pixel_weights * mask

        # Normalise by sum of valid weights so the loss scale
        # remains comparable across batches.
        denominator = weighted_mask.sum().clamp_min(1e-8)

        return (element_loss * weighted_mask).sum() / denominator


# ============================================================
# 7. METRICS
# ============================================================

def calculate_metrics(predictions, actuals):

    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)

    residual = predictions - actuals

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))

    r2 = (
        float(1.0 - ss_res / ss_tot)
        if ss_tot > 0
        else None
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "mean_actual": float(actuals.mean()),
        "mean_predicted": float(predictions.mean()),
        "actual_std": float(actuals.std()),
        "predicted_std": float(predictions.std()),
        "mean_residual": float(residual.mean()),
        "supervised_pixels": int(len(actuals)),
    }


# ============================================================
# 8. TRAINING EPOCH
# ============================================================

def train_epoch(model, loader, optimizer, loss_fn, device):

    model.train()

    total_loss = 0.0
    batches = 0

    for x, y, mask in loader:

        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(x)

        loss = loss_fn(
            prediction,
            y,
            mask,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / max(batches, 1)


# ============================================================
# 9. VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader, loss_fn, device):

    model.eval()

    total_loss = 0.0
    batches = 0

    predictions = []
    actuals = []

    for x, y, mask in loader:

        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        prediction = model(x)

        loss = loss_fn(
            prediction,
            y,
            mask,
        )

        total_loss += loss.item()
        batches += 1

        valid = mask.bool()

        pred_values = prediction[valid].cpu().numpy()
        actual_values = y[valid].cpu().numpy()

        pred_values = pred_values * TARGET_STD + TARGET_MEAN
        actual_values = actual_values * TARGET_STD + TARGET_MEAN

        predictions.append(pred_values)
        actuals.append(actual_values)

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    metrics = calculate_metrics(predictions, actuals)

    metrics["validation_loss"] = total_loss / max(batches, 1)

    return metrics


# ============================================================
# 10. MAIN
# ============================================================

def main():

    set_seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)

    if not VAL_FILE.exists():
        raise FileNotFoundError(VAL_FILE)

    logger.info("=" * 65)
    logger.info("TASK 10 — TAIL-WEIGHTED U-NET EXPERIMENT")
    logger.info("=" * 65)

    logger.info("Device: %s", DEVICE)

    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("Maximum epochs: %d", EPOCHS)
    logger.info("Quantile weights: %s", QUANTILE_WEIGHTS)

    # Load only train and validation.
    train_dataset = FloodPatchDataset(
        TRAIN_FILE,
        augment=True,
    )

    val_dataset = FloodPatchDataset(
        VAL_FILE,
        augment=False,
    )

    # Derive boundaries exclusively from training labels.
    boundaries = calculate_training_quantiles(train_dataset)

    loss_fn = TailWeightedHuberLoss(
        boundaries=boundaries,
        weights=QUANTILE_WEIGHTS,
    ).to(DEVICE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(
        in_channels=INPUT_CHANNELS,
        base_channels=BASE_CHANNELS,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            DEVICE,
        )

        val_metrics = validate(
            model,
            val_loader,
            loss_fn,
            DEVICE,
        )

        val_rmse = val_metrics["RMSE"]

        scheduler.step(val_rmse)

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_metrics["validation_loss"],
            "validation_MAE": val_metrics["MAE"],
            "validation_RMSE": val_metrics["RMSE"],
            "validation_R2": val_metrics["R2"],
            "learning_rate": current_lr,
        }

        history.append(row)

        logger.info(
            "Epoch %03d/%03d | "
            "Train loss: %.5f | "
            "Val loss: %.5f | "
            "MAE: %.3f | "
            "RMSE: %.3f | "
            "R2: %.4f | "
            "LR: %.2e",
            epoch,
            EPOCHS,
            train_loss,
            val_metrics["validation_loss"],
            val_metrics["MAE"],
            val_metrics["RMSE"],
            val_metrics["R2"],
            current_lr,
        )

        if val_rmse < best_val_rmse:

            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": val_metrics,
                    "model_config": {
                        "input_channels": INPUT_CHANNELS,
                        "base_channels": BASE_CHANNELS,
                        "patch_size": PATCH_SIZE,
                    },
                    "target_normalisation": {
                        "mean": TARGET_MEAN,
                        "std": TARGET_STD,
                    },
                    "loss_config": {
                        "name": "tail_weighted_huber",
                        "quantile_boundaries_normalised": boundaries.tolist(),
                        "quantile_weights": QUANTILE_WEIGHTS,
                    },
                    "seed": SEED,
                },
                BEST_MODEL_FILE,
            )

            logger.info(
                "Best weighted checkpoint saved. RMSE: %.4f",
                best_val_rmse,
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            logger.info(
                "Early stopping at epoch %d. Best epoch: %d",
                epoch,
                best_epoch,
            )

            break

    # Save training history.
    with open(
        HISTORY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )

        writer.writeheader()
        writer.writerows(history)

    # Read best checkpoint metrics.
    checkpoint = torch.load(
        BEST_MODEL_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    final_metrics = checkpoint["validation_metrics"]

    summary = {
        "experiment": "Tail-weighted U-Net",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_metrics": final_metrics,
        "training_quantile_boundaries_normalised": boundaries.tolist(),
        "quantile_weights": QUANTILE_WEIGHTS,
        "model_path": str(BEST_MODEL_FILE),
        "history_path": str(HISTORY_FILE),
        "train_patches": len(train_dataset),
        "validation_patches": len(val_dataset),
        "test_split_used": False,
        "target_normalisation": {
            "mean": TARGET_MEAN,
            "std": TARGET_STD,
        },
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(summary, file, indent=4)

    logger.info("=" * 65)
    logger.info("TAIL-WEIGHTED U-NET COMPLETED")
    logger.info("=" * 65)

    logger.info("Best epoch: %d", best_epoch)
    logger.info("Validation MAE: %.4f", final_metrics["MAE"])
    logger.info("Validation RMSE: %.4f", final_metrics["RMSE"])
    logger.info("Validation R2: %.6f", final_metrics["R2"])

    logger.info("Model: %s", BEST_MODEL_FILE)
    logger.info("History: %s", HISTORY_FILE)
    logger.info("Metrics: %s", METRICS_FILE)

    logger.info("Test split was not used.")


if __name__ == "__main__":
    main()
