
"""
Task 10 — Final U-Net Test Evaluation

Run from project root:
    python src/experiments/task10_unet_test_evaluation.py

Evaluates the original U-Net checkpoint (best epoch 7).

Uses:
    models/task10_unet_best.pt
    data/processed/task10_patches/test.npz

Does NOT:
    - Train or fine-tune the model
    - Load training or validation patches
    - Access or modify the weighted U-Net checkpoint
    - Use test results for model selection
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PATCH_DIR = ROOT / "data" / "processed" / "task10_patches"
MODEL_FILE = ROOT / "models" / "task10_unet_best.pt"

TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"

TEST_FILE = PATCH_DIR / "test.npz"

PREDICTIONS_FILE = (
    TABLE_DIR / "10_unet_test_predictions.csv"
)

SUMMARY_FILE = (
    TABLE_DIR / "10_unet_test_metrics.json"
)

QUANTILE_FILE = (
    TABLE_DIR / "10_unet_test_error_by_quantile.csv"
)

SCATTER_PLOT = (
    FIGURE_DIR / "10_unet_test_actual_vs_predicted.png"
)

DISTRIBUTION_PLOT = (
    FIGURE_DIR / "10_unet_test_prediction_distribution.png"
)

RESIDUAL_PLOT = (
    FIGURE_DIR / "10_unet_test_residual_distribution.png"
)

QUANTILE_PLOT = (
    FIGURE_DIR / "10_unet_test_error_by_quantile.png"
)

BATCH_SIZE = 16
NUM_WORKERS = 0

EXPECTED_TEST_CELLS = 10767

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger("task10_unet_test")


# ============================================================
# 2. U-NET ARCHITECTURE
# Must match the original training architecture.
# ============================================================

def group_count(channels: int) -> int:

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

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = DoubleConv(b * 4, b * 8)

        self.up3 = nn.ConvTranspose2d(
            b * 8,
            b * 4,
            kernel_size=2,
            stride=2,
        )

        self.dec3 = DoubleConv(b * 8, b * 4)

        self.up2 = nn.ConvTranspose2d(
            b * 4,
            b * 2,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = DoubleConv(b * 4, b * 2)

        self.up1 = nn.ConvTranspose2d(
            b * 2,
            b,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv(b * 2, b)

        self.output = nn.Conv2d(
            b,
            1,
            kernel_size=1,
        )

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
# 3. TEST DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(self, file_path: Path):

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        logger.info("Loading test patches...")

        with np.load(file_path) as data:

            self.X = np.asarray(
                data["X"],
                dtype=np.float32,
            )

            self.y = np.asarray(
                data["y"],
                dtype=np.float32,
            )

            self.mask = np.asarray(
                data["mask"],
                dtype=np.float32,
            )

            self.rows = np.asarray(data["row"])
            self.cols = np.asarray(data["col"])

            self.block_ids = np.asarray(
                data["block_id"]
            ).astype(str)

        if self.X.ndim != 4 or self.X.shape[1] != 10:
            raise ValueError(
                f"Unexpected feature shape: {self.X.shape}"
            )

        if self.y.shape != self.mask.shape:
            raise ValueError(
                "Target and mask shapes do not match."
            )

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                "Feature and target patch counts differ."
            )

        if not (
            len(self.rows)
            == len(self.cols)
            == len(self.block_ids)
            == len(self.X)
        ):
            raise ValueError(
                "Patch metadata length mismatch."
            )

        if not np.isfinite(self.X).all():
            raise ValueError(
                "Features contain NaN or infinite values."
            )

        if not np.isfinite(self.y).all():
            raise ValueError(
                "Targets contain NaN or infinite values."
            )

        logger.info(
            "Test patches: %d",
            len(self.X),
        )

        logger.info(
            "Supervised test cells: %d",
            int(self.mask.sum()),
        )

        if int(self.mask.sum()) != EXPECTED_TEST_CELLS:
            raise ValueError(
                f"Expected {EXPECTED_TEST_CELLS} test cells, "
                f"found {int(self.mask.sum())}. "
                "Check the Task 10 test patches."
            )

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

    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    residual = predicted - actual

    mae = float(
        np.mean(np.abs(residual))
    )

    rmse = float(
        np.sqrt(np.mean(residual ** 2))
    )

    ss_res = float(
        np.sum(residual ** 2)
    )

    ss_tot = float(
        np.sum((actual - actual.mean()) ** 2)
    )

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
        "median_absolute_error": float(
            np.median(np.abs(residual))
        ),
        "underprediction_cells": int(
            np.sum(predicted < actual)
        ),
        "overprediction_cells": int(
            np.sum(predicted > actual)
        ),
        "supervised_cells": int(len(actual)),
    }


# ============================================================
# 5. GENERATE TEST PREDICTIONS
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

    for batch_index, (x, y, mask) in enumerate(loader, start=1):

        x = x.to(device)

        prediction = model(x).cpu().numpy()

        y = y.numpy()
        mask = mask.numpy().astype(bool)

        for i in range(len(x)):

            valid = mask[i, 0]

            pred_values = prediction[i, 0][valid]
            actual_values = y[i, 0][valid]

            # Convert normalised predictions back to original units.
            pred_values = (
                pred_values * target_std + target_mean
            )

            actual_values = (
                actual_values * target_std + target_mean
            )

            all_predicted.append(pred_values)
            all_actual.append(actual_values)

        if batch_index % 5 == 0:
            logger.info(
                "Processed %d/%d test batches",
                batch_index,
                len(loader),
            )

    actual = np.concatenate(all_actual)
    predicted = np.concatenate(all_predicted)

    if len(actual) != int(dataset.mask.sum()):
        raise RuntimeError(
            "Prediction count does not match supervised test cells."
        )

    # Recover exact raster coordinates.
    rows = []
    cols = []
    block_ids = []

    for i in range(len(dataset)):

        row0 = int(dataset.rows[i])
        col0 = int(dataset.cols[i])

        valid_positions = np.argwhere(
            dataset.mask[i, 0].astype(bool)
        )

        for local_row, local_col in valid_positions:

            rows.append(
                row0 + int(local_row)
            )

            cols.append(
                col0 + int(local_col)
            )

            block_ids.append(
                dataset.block_ids[i]
            )

    if len(rows) != len(actual):
        raise RuntimeError(
            "Test prediction coordinate count mismatch."
        )

    return (
        actual,
        predicted,
        np.asarray(rows),
        np.asarray(cols),
        np.asarray(block_ids),
    )


# ============================================================
# 6. ERROR BY ACTUAL TEST QUANTILE
# ============================================================

def analyse_quantiles(prediction_df):

    df = prediction_df.copy()

    # Quantiles are descriptive test-set evaluation groups.
    # They are NOT used for model training or model selection.
    df["quantile"] = pd.qcut(
        df["actual"],
        q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        duplicates="drop",
    )

    quantile_df = (
        df.groupby(
            "quantile",
            observed=True,
        )
        .agg(
            cells=("actual", "size"),
            mean_actual=("actual", "mean"),
            mean_predicted=("predicted", "mean"),
            mean_residual=("residual", "mean"),
            MAE=("absolute_error", "mean"),
            RMSE=(
                "residual",
                lambda x: np.sqrt(np.mean(x ** 2)),
            ),
        )
        .reset_index()
    )

    return quantile_df


# ============================================================
# 7. PLOTS
# ============================================================

def create_plots(
    actual,
    predicted,
    quantile_df,
):

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    residual = predicted - actual

    # --------------------------------------------------------
    # Actual vs predicted
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(
        actual,
        predicted,
        s=7,
        alpha=0.25,
    )

    lower = min(
        actual.min(),
        predicted.min(),
    )

    upper = max(
        actual.max(),
        predicted.max(),
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel("Actual KSDMA hazard value")
    ax.set_ylabel("Predicted KSDMA hazard value")

    ax.set_title(
        "Original U-Net: Actual vs Predicted — Test"
    )

    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(SCATTER_PLOT, dpi=200)
    plt.close(fig)

    # --------------------------------------------------------
    # Actual vs predicted distribution
    # --------------------------------------------------------

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

    ax.set_title(
        "Original U-Net: Actual vs Predicted Distribution — Test"
    )

    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(DISTRIBUTION_PLOT, dpi=200)
    plt.close(fig)

    # --------------------------------------------------------
    # Residual distribution
    # --------------------------------------------------------

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

    ax.set_xlabel(
        "Residual (predicted - actual)"
    )

    ax.set_ylabel("Number of cells")

    ax.set_title(
        "Original U-Net: Test Residual Distribution"
    )

    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(RESIDUAL_PLOT, dpi=200)
    plt.close(fig)

    # --------------------------------------------------------
    # Error by quantile
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        quantile_df["quantile"].astype(str),
        quantile_df["MAE"],
    )

    ax.set_xlabel("Actual test target quantile")
    ax.set_ylabel("MAE")

    ax.set_title(
        "Original U-Net: Test MAE by Hazard Quantile"
    )

    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(QUANTILE_PLOT, dpi=200)
    plt.close(fig)


# ============================================================
# 8. MAIN
# ============================================================

def main():

    logger.info("=" * 65)
    logger.info("TASK 10 — FINAL U-NET TEST EVALUATION")
    logger.info("=" * 65)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing model: {MODEL_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing test file: {TEST_FILE}"
        )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Device: %s", DEVICE)

    if torch.cuda.is_available():
        logger.info(
            "GPU: %s",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Load original U-Net checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    model_config = checkpoint["model_config"]
    target_stats = checkpoint["target_normalisation"]

    target_mean = float(
        target_stats["mean"]
    )

    target_std = float(
        target_stats["std"]
    )

    model = UNet(
        in_channels=model_config["input_channels"],
        base_channels=model_config["base_channels"],
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    logger.info(
        "Loaded original U-Net checkpoint from epoch %d",
        checkpoint["epoch"],
    )

    logger.info(
        "Checkpoint: %s",
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    dataset = TestDataset(TEST_FILE)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    (
        actual,
        predicted,
        rows,
        cols,
        block_ids,
    ) = generate_predictions(
        model,
        loader,
        dataset,
        DEVICE,
        target_mean,
        target_std,
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        actual,
        predicted,
    )

    logger.info("-" * 65)
    logger.info("FINAL TEST METRICS")
    logger.info("-" * 65)

    for key, value in metrics.items():

        if isinstance(value, float):
            logger.info(
                "%s: %.6f",
                key,
                value,
            )
        else:
            logger.info(
                "%s: %s",
                key,
                value,
            )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({

        "row": rows,
        "col": cols,
        "block_id": block_ids,
        "actual": actual,
        "predicted": predicted,
        "residual": predicted - actual,
        "absolute_error": np.abs(
            predicted - actual
        ),
    })

    if prediction_df.duplicated(
        ["row", "col"]
    ).any():

        raise RuntimeError(
            "Duplicate test raster coordinates found."
        )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Quantile analysis
    # --------------------------------------------------------

    quantile_df = analyse_quantiles(
        prediction_df
    )

    quantile_df.to_csv(
        QUANTILE_FILE,
        index=False,
    )

    logger.info("-" * 65)
    logger.info("TEST ERROR BY HAZARD QUANTILE")
    logger.info("-" * 65)

    for _, row in quantile_df.iterrows():

        logger.info(
            "%s | Cells: %d | Actual: %.2f | "
            "Predicted: %.2f | Residual: %.2f | "
            "MAE: %.2f | RMSE: %.2f",
            row["quantile"],
            int(row["cells"]),
            row["mean_actual"],
            row["mean_predicted"],
            row["mean_residual"],
            row["MAE"],
            row["RMSE"],
        )

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {

        "task": "Task 10 — Final U-Net test evaluation",

        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),

        "validation_metrics_from_checkpoint": (
            checkpoint["validation_metrics"]
        ),

        "test_metrics": metrics,

        "prediction_mean_difference": float(
            predicted.mean() - actual.mean()
        ),

        "prediction_to_actual_std_ratio": (
            float(
                predicted.std() / actual.std()
            )
            if actual.std() > 0
            else None
        ),

        "test_cells_expected": EXPECTED_TEST_CELLS,

        "test_cells_evaluated": int(len(actual)),

        "duplicate_test_coordinates": int(
            prediction_df.duplicated(
                ["row", "col"]
            ).sum()
        ),

        "model_path": str(MODEL_FILE),

        "predictions_path": str(
            PREDICTIONS_FILE
        ),

        "quantile_analysis_path": str(
            QUANTILE_FILE
        ),

        "test_used_for_model_selection": False,
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    create_plots(
        actual,
        predicted,
        quantile_df,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    logger.info("-" * 65)
    logger.info("TEST EVALUATION FILES SAVED")
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
    logger.info("FINAL U-NET TEST EVALUATION COMPLETED")
    logger.info("No training or model selection was performed.")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
