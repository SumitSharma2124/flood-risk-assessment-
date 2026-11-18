
"""
Task 10 — U-Net Validation Diagnostics

Run from project root:
    python src/experiments/task10_unet_diagnostics.py

Uses validation patches only.
Does not retrain or access test data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PATCH_DIR = ROOT / "data" / "processed" / "task10_patches"
MODEL_FILE = ROOT / "models" / "task10_unet_best.pt"

OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

VAL_FILE = PATCH_DIR / "validation.npz"

PREDICTIONS_FILE = TABLE_DIR / "10_unet_validation_predictions.csv"
SUMMARY_FILE = TABLE_DIR / "10_unet_validation_diagnostics.json"
QUANTILE_FILE = TABLE_DIR / "10_unet_validation_error_by_quantile.csv"

SCATTER_PLOT = FIGURE_DIR / "10_unet_actual_vs_predicted.png"
DISTRIBUTION_PLOT = FIGURE_DIR / "10_unet_prediction_distribution.png"
RESIDUAL_PLOT = FIGURE_DIR / "10_unet_residual_distribution.png"
QUANTILE_PLOT = FIGURE_DIR / "10_unet_error_by_quantile.png"

BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger("task10_diagnostics")


# ============================================================
# 2. U-NET ARCHITECTURE
# Must match the training architecture exactly.
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
# 3. VALIDATION DATASET
# ============================================================

class ValidationDataset(Dataset):
    def __init__(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        logger.info("Loading validation patches...")

        with np.load(file_path) as data:
            self.X = np.asarray(data["X"], dtype=np.float32)
            self.y = np.asarray(data["y"], dtype=np.float32)
            self.mask = np.asarray(data["mask"], dtype=np.float32)

            # Preserve patch location and block identity.
            self.rows = np.asarray(data["row"])
            self.cols = np.asarray(data["col"])
            self.block_ids = np.asarray(data["block_id"]).astype(str)

        if self.X.ndim != 4 or self.X.shape[1] != 10:
            raise ValueError(f"Unexpected feature shape: {self.X.shape}")

        if self.y.shape != self.mask.shape:
            raise ValueError("Target and mask shapes do not match.")

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Feature and target patch counts do not match.")

        if not (
            len(self.rows)
            == len(self.cols)
            == len(self.block_ids)
            == len(self.X)
        ):
            raise ValueError("Patch metadata length mismatch.")

        logger.info("Validation patches: %d", len(self.X))
        logger.info("Supervised cells: %d", int(self.mask.sum()))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.X[index].copy()),
            torch.from_numpy(self.y[index].copy()),
            torch.from_numpy(self.mask[index].copy()),
        )


# ============================================================
# 4. METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    residual = predicted - actual

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))

    r2 = (
        float(1.0 - ss_res / ss_tot)
        if ss_tot > 0
        else None
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "mean_actual": float(actual.mean()),
        "mean_predicted": float(predicted.mean()),
        "median_actual": float(np.median(actual)),
        "median_predicted": float(np.median(predicted)),
        "actual_std": float(actual.std()),
        "predicted_std": float(predicted.std()),
        "mean_residual": float(residual.mean()),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "underprediction_cells": int(np.sum(predicted < actual)),
        "overprediction_cells": int(np.sum(predicted > actual)),
        "exact_prediction_cells": int(np.sum(predicted == actual)),
        "supervised_cells": int(len(actual)),
    }


# ============================================================
# 5. PREDICTIONS
# ============================================================

@torch.no_grad()
def generate_predictions(
    model,
    loader,
    dataset,
    device,
    target_mean,
    target_std,
):
    model.eval()

    all_actual = []
    all_predicted = []

    for x, y, mask in loader:
        x = x.to(device)

        prediction = model(x).cpu().numpy()
        y = y.numpy()
        mask = mask.numpy().astype(bool)

        for i in range(len(x)):
            valid = mask[i, 0]

            pred_values = prediction[i, 0][valid]
            actual_values = y[i, 0][valid]

            # Convert back to original target units.
            pred_values = pred_values * target_std + target_mean
            actual_values = actual_values * target_std + target_mean

            all_predicted.append(pred_values)
            all_actual.append(actual_values)

    actual = np.concatenate(all_actual)
    predicted = np.concatenate(all_predicted)

    if len(actual) != int(dataset.mask.sum()):
        raise RuntimeError(
            "Prediction count does not match supervised cell count."
        )

    # Recover exact raster positions from patch metadata and masks.
    rows = []
    cols = []
    block_ids = []

    patch_size = dataset.X.shape[-1]

    for i in range(len(dataset)):
        row0 = int(dataset.rows[i])
        col0 = int(dataset.cols[i])

        valid_positions = np.argwhere(
            dataset.mask[i, 0].astype(bool)
        )

        for local_row, local_col in valid_positions:
            rows.append(row0 + int(local_row))
            cols.append(col0 + int(local_col))
            block_ids.append(dataset.block_ids[i])

    if len(rows) != len(actual):
        raise RuntimeError("Prediction coordinate count mismatch.")

    return actual, predicted, np.asarray(rows), np.asarray(cols), np.asarray(block_ids)


# ============================================================
# 6. PLOTS
# ============================================================

def create_plots(actual, predicted, quantile_df):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    residual = predicted - actual

    # Actual vs predicted.
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(
        actual,
        predicted,
        s=7,
        alpha=0.25,
    )

    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel("Actual KSDMA hazard value")
    ax.set_ylabel("Predicted KSDMA hazard value")
    ax.set_title("U-Net: Actual vs Predicted — Validation")

    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(SCATTER_PLOT, dpi=200)
    plt.close(fig)

    # Actual and predicted distributions.
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        actual,
        bins=40,
        alpha=0.55,
        density=True,
        label="Actual",
    )

    ax.hist(
        predicted,
        bins=40,
        alpha=0.55,
        density=True,
        label="Predicted",
    )

    ax.set_xlabel("KSDMA hazard value")
    ax.set_ylabel("Density")
    ax.set_title("U-Net: Actual vs Predicted Distribution")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(DISTRIBUTION_PLOT, dpi=200)
    plt.close(fig)

    # Residual distribution.
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        residual,
        bins=50,
        alpha=0.8,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel("Residual (predicted - actual)")
    ax.set_ylabel("Number of cells")
    ax.set_title("U-Net: Validation Residual Distribution")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(RESIDUAL_PLOT, dpi=200)
    plt.close(fig)

    # Error by target quantile.
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        quantile_df["quantile"],
        quantile_df["MAE"],
    )

    ax.set_xlabel("Actual target quantile")
    ax.set_ylabel("MAE")
    ax.set_title("U-Net: Validation MAE by Actual Hazard Quantile")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(QUANTILE_PLOT, dpi=200)
    plt.close(fig)


# ============================================================
# 7. MAIN
# ============================================================

def main():
    logger.info("=" * 65)
    logger.info("TASK 10 — U-NET VALIDATION DIAGNOSTICS")
    logger.info("=" * 65)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_FILE}")

    if not VAL_FILE.exists():
        raise FileNotFoundError(f"Missing validation data: {VAL_FILE}")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s", DEVICE)

    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Load checkpoint.
    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    model_config = checkpoint["model_config"]
    target_stats = checkpoint["target_normalisation"]

    target_mean = float(target_stats["mean"])
    target_std = float(target_stats["std"])

    model = UNet(
        in_channels=model_config["input_channels"],
        base_channels=model_config["base_channels"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("Loaded checkpoint from epoch %d", checkpoint["epoch"])

    # Load validation data only.
    dataset = ValidationDataset(VAL_FILE)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    actual, predicted, rows, cols, block_ids = generate_predictions(
        model,
        loader,
        dataset,
        DEVICE,
        target_mean,
        target_std,
    )

    # Metrics.
    metrics = calculate_metrics(actual, predicted)

    logger.info("-" * 65)
    logger.info("VALIDATION METRICS")
    logger.info("-" * 65)

    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info("%s: %.6f", key, value)
        else:
            logger.info("%s: %s", key, value)

    # Prediction table.
    prediction_df = pd.DataFrame({
        "row": rows,
        "col": cols,
        "block_id": block_ids,
        "actual": actual,
        "predicted": predicted,
        "residual": predicted - actual,
        "absolute_error": np.abs(predicted - actual),
    })

    if prediction_df.duplicated(["row", "col"]).any():
        raise RuntimeError("Duplicate validation raster coordinates found.")

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # Error by actual target quantile.
    prediction_df["quantile"] = pd.qcut(
        prediction_df["actual"],
        q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        duplicates="drop",
    )

    quantile_df = (
        prediction_df
        .groupby("quantile", observed=True)
        .apply(
            lambda group: pd.Series({
                "cells": len(group),
                "mean_actual": group["actual"].mean(),
                "mean_predicted": group["predicted"].mean(),
                "mean_residual": group["residual"].mean(),
                "MAE": group["absolute_error"].mean(),
                "RMSE": np.sqrt(np.mean(group["residual"] ** 2)),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    quantile_df.to_csv(
        QUANTILE_FILE,
        index=False,
    )

    # Save summary.
    summary = {
        "task": "Task 10 — U-Net validation diagnostics",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_metrics": metrics,
        "prediction_mean_difference": float(
            predicted.mean() - actual.mean()
        ),
        "prediction_to_actual_std_ratio": float(
            predicted.std() / actual.std()
        ) if actual.std() > 0 else None,
        "test_split_used": False,
        "model_path": str(MODEL_FILE),
        "predictions_path": str(PREDICTIONS_FILE),
        "quantile_analysis_path": str(QUANTILE_FILE),
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # Plots.
    create_plots(
        actual,
        predicted,
        quantile_df,
    )

    logger.info("-" * 65)
    logger.info("DIAGNOSTIC FILES SAVED")
    logger.info("-" * 65)

    for path in [
        PREDICTIONS_FILE,
        QUANTILE_FILE,
        SUMMARY_FILE,
        SCATTER_PLOT,
        DISTRIBUTION_PLOT,
        RESIDUAL_PLOT,
        QUANTILE_PLOT,
    ]:
        logger.info("%s", path)

    logger.info("=" * 65)
    logger.info("U-NET VALIDATION DIAGNOSTICS COMPLETED")
    logger.info("Test split was not accessed.")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
