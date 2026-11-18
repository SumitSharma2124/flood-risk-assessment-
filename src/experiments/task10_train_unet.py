
"""
Task 10 — Train a compact U-Net for KSDMA 100-year flood hazard prediction.

Run from project root:
    python src/experiments/task10_train_unet.py

Uses:
    data/processed/task10_patches/train.npz
    data/processed/task10_patches/validation.npz

The test split is deliberately NOT loaded or evaluated here.
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

BEST_MODEL_FILE = MODEL_DIR / "task10_unet_best.pt"
HISTORY_FILE = OUTPUT_DIR / "10_unet_training_history.csv"
METRICS_FILE = OUTPUT_DIR / "10_unet_validation_metrics.json"

SEED = 42

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 15

INPUT_CHANNELS = 10
BASE_CHANNELS = 16
PATCH_SIZE = 16

# Statistics calculated using training cells only.
TARGET_MEAN = 270.157498
TARGET_STD = 113.758504

NUM_WORKERS = 0  # Safe default for Windows.


# ============================================================
# 2. REPRODUCIBILITY AND LOGGING
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger("task10_unet")


# ============================================================
# 3. DATASET
# ============================================================

class FloodPatchDataset(Dataset):
    def __init__(self, file_path: Path, augment: bool = False):
        if not file_path.exists():
            raise FileNotFoundError(f"Missing patch file: {file_path}")

        logger.info("Loading %s", file_path.name)

        with np.load(file_path) as data:
            self.X = np.asarray(data["X"], dtype=np.float32)
            self.y = np.asarray(data["y"], dtype=np.float32)
            self.mask = np.asarray(data["mask"], dtype=np.float32)

        if self.X.ndim != 4:
            raise ValueError(f"Expected X to be 4D, got {self.X.shape}")

        if self.y.ndim != 4 or self.mask.ndim != 4:
            raise ValueError("Expected y and mask to be 4D arrays.")

        if self.X.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                f"Expected {INPUT_CHANNELS} input channels, "
                f"got {self.X.shape[1]}"
            )

        if self.y.shape[1] != 1 or self.mask.shape[1] != 1:
            raise ValueError("Expected one target channel and one mask channel.")

        if self.X.shape[0] != self.y.shape[0] or self.y.shape != self.mask.shape:
            raise ValueError("Inconsistent patch counts or target/mask shapes.")

        if self.X.shape[2:] != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Unexpected patch size: {self.X.shape[2:]}")

        if not np.isfinite(self.X).all():
            raise ValueError("Input features contain NaN or infinite values.")

        if not np.isfinite(self.y).all():
            raise ValueError("Target array contains NaN or infinite values.")

        self.augment = augment

        logger.info(
            "%s loaded: %d patches | supervised pixels: %d",
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

        # Spatial augmentation is applied only to training patches.
        # The same transformation is applied to features, target and mask.
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
# ============================================================

def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
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

        self.pool = nn.MaxPool2d(kernel_size=2)

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

        # Regression output: one continuous hazard value per pixel.
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
# 5. MASKED HUBER LOSS
# ============================================================

def masked_huber_loss(prediction, target, mask):
    """
    Calculate loss only for valid supervised pixels.
    Excluded pixels do not contribute to the loss.
    """

    element_loss = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=1.0,
    )

    masked_loss = element_loss * mask
    valid_count = mask.sum().clamp_min(1.0)

    return masked_loss.sum() / valid_count


# ============================================================
# 6. METRICS
# ============================================================

def calculate_metrics(predictions, actuals):
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)

    errors = predictions - actuals

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = float(np.sum(errors ** 2))
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
        "supervised_pixels": int(len(actuals)),
    }


# ============================================================
# 7. TRAINING AND VALIDATION
# ============================================================

def run_training_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_batches = 0

    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(x)
        loss = masked_huber_loss(prediction, y, mask)

        loss.backward()

        # Small safeguard against unusually large gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def run_validation(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    all_predictions = []
    all_actuals = []

    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        prediction = model(x)

        loss = masked_huber_loss(prediction, y, mask)

        total_loss += loss.item()
        total_batches += 1

        valid = mask.bool()

        pred_values = prediction[valid].detach().cpu().numpy()
        actual_values = y[valid].detach().cpu().numpy()

        # Convert from normalised target units to original KSDMA units.
        pred_values = pred_values * TARGET_STD + TARGET_MEAN
        actual_values = actual_values * TARGET_STD + TARGET_MEAN

        all_predictions.append(pred_values)
        all_actuals.append(actual_values)

    predictions = np.concatenate(all_predictions)
    actuals = np.concatenate(all_actuals)

    metrics = calculate_metrics(predictions, actuals)
    metrics["validation_loss"] = total_loss / max(total_batches, 1)

    return metrics


# ============================================================
# 8. MAIN
# ============================================================

def main():
    set_seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise FileNotFoundError(
            "Training or validation patch file is missing. "
            "Run Task 10 patch extraction first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("=" * 60)
    logger.info("TASK 10 — U-NET TRAINING")
    logger.info("=" * 60)

    logger.info("Device: %s", device)

    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    logger.info("Patch size: %d x %d", PATCH_SIZE, PATCH_SIZE)
    logger.info("Input channels: %d", INPUT_CHANNELS)
    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("Maximum epochs: %d", EPOCHS)

    # Load training and validation data only.
    train_dataset = FloodPatchDataset(
        TRAIN_FILE,
        augment=True,
    )

    val_dataset = FloodPatchDataset(
        VAL_FILE,
        augment=False,
    )

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
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    logger.info("Trainable parameters: %s", f"{parameter_count:,}")

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
        train_loss = run_training_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_metrics = run_validation(
            model,
            val_loader,
            device,
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
            "Val MAE: %.3f | "
            "Val RMSE: %.3f | "
            "Val R2: %s | "
            "LR: %.2e",
            epoch,
            EPOCHS,
            train_loss,
            val_metrics["validation_loss"],
            val_metrics["MAE"],
            val_rmse,
            (
                f"{val_metrics['R2']:.4f}"
                if val_metrics["R2"] is not None
                else "N/A"
            ),
            current_lr,
        )

        # Save the best model according to validation RMSE.
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
                    "seed": SEED,
                },
                BEST_MODEL_FILE,
            )

            logger.info(
                "Best checkpoint saved. Validation RMSE: %.4f",
                best_val_rmse,
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            logger.info(
                "Early stopping at epoch %d. "
                "Best epoch: %d",
                epoch,
                best_epoch,
            )
            break

    # Save training history.
    with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    # Reload best checkpoint and record its validation metrics.
    checkpoint = torch.load(
        BEST_MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    final_metrics = checkpoint["validation_metrics"]

    summary = {
        "task": "Task 10 — U-Net",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_metrics": final_metrics,
        "model_path": str(BEST_MODEL_FILE),
        "history_path": str(HISTORY_FILE),
        "train_patches": len(train_dataset),
        "validation_patches": len(val_dataset),
        "test_split_used": False,
        "target_normalisation": {
            "mean": TARGET_MEAN,
            "std": TARGET_STD,
        },
        "model_config": {
            "input_channels": INPUT_CHANNELS,
            "base_channels": BASE_CHANNELS,
            "patch_size": PATCH_SIZE,
            "batch_size": BATCH_SIZE,
        },
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    logger.info("=" * 60)
    logger.info("U-NET TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info("Best epoch: %d", best_epoch)
    logger.info("Best validation MAE: %.4f", final_metrics["MAE"])
    logger.info("Best validation RMSE: %.4f", final_metrics["RMSE"])
    logger.info("Best validation R2: %s", final_metrics["R2"])
    logger.info("Model: %s", BEST_MODEL_FILE)
    logger.info("History: %s", HISTORY_FILE)
    logger.info("Metrics: %s", METRICS_FILE)
    logger.info("Test split was not used.")


if __name__ == "__main__":
    main()
