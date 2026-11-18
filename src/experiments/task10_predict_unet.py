
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg

DATA = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
MODEL_FILE = ROOT / "models" / "task10_unet_best.pt"

FEATURE_STATS_FILE = TABLES / "10_unet_feature_normalisation.json"
FEATURE_FILE = DATA / "features_stack.tif"
FEATURE_MASK_FILE = DATA / "features_valid_mask.tif"
DISTRICT_FILE = DATA / "district_mask.tif"

OUTPUT_DIR = ROOT / "outputs" / "rasters"
OUTPUT_FILE = OUTPUT_DIR / "10_unet_full_district_prediction.tif"
METADATA_FILE = TABLES / "10_unet_full_district_prediction.json"

FEATURE_BANDS = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]

FEATURE_NAMES = [
    "elevation", "slope", "twi", "distance_waterway",
    "trees", "cropland", "builtup", "water",
    "wetland", "mangrove",
]

PATCH_SIZE = 16
BLOCK_SIZE = 5000
NODATA = -9999.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("unet_prediction")


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
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
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

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2)
        self.dec3 = DoubleConv(b * 8, b * 4)

        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2)
        self.dec2 = DoubleConv(b * 4, b * 2)

        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, 2)
        self.dec1 = DoubleConv(b * 2, b)

        self.output = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        bottleneck = self.bottleneck(self.pool(e3))

        d3 = self.up3(bottleneck)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output(d1)


def create_block_grid(transform, height, width):
    rows, cols = np.indices((height, width))

    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e

    bx = np.floor(x / BLOCK_SIZE).astype(np.int32)
    by = np.floor(y / BLOCK_SIZE).astype(np.int32)

    return (
        np.char.add(
            np.char.add(bx.astype(str), "_"),
            by.astype(str),
        )
    )


def read_aligned(path, reference):
    with rasterio.open(path) as src:
        if (
            src.height != reference["height"]
            or src.width != reference["width"]
            or src.transform != reference["transform"]
            or src.crs != reference["crs"]
        ):
            raise ValueError(f"Grid mismatch: {path}")

        return src.read(1), src.nodata


def main():
    for path in [
        MODEL_FILE, FEATURE_STATS_FILE,
        FEATURE_FILE, FEATURE_MASK_FILE, DISTRICT_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    if OUTPUT_FILE.exists() or METADATA_FILE.exists():
        raise FileExistsError(
            "Prediction output already exists. "
            "Move or rename it before rerunning."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s", DEVICE)

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
    model.eval()

    with open(FEATURE_STATS_FILE, encoding="utf-8") as f:
        feature_stats = json.load(f)

    means = np.array(
        [feature_stats[n]["mean"] for n in FEATURE_NAMES],
        dtype=np.float32,
    )

    stds = np.array(
        [feature_stats[n]["std"] for n in FEATURE_NAMES],
        dtype=np.float32,
    )

    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise ValueError("Invalid feature normalization statistics.")

    if np.any(stds <= 0) or target_std <= 0:
        raise ValueError("Invalid standard deviation.")

    with rasterio.open(FEATURE_FILE) as src:
        features = src.read(FEATURE_BANDS).astype(np.float32)

        profile = src.profile.copy()
        transform = src.transform

        height = src.height
        width = src.width
        feature_nodata = src.nodata

        reference = {
            "height": height,
            "width": width,
            "transform": transform,
            "crs": src.crs,
        }

    feature_mask, _ = read_aligned(FEATURE_MASK_FILE, reference)
    district, _ = read_aligned(DISTRICT_FILE, reference)

    feature_finite = np.all(np.isfinite(features), axis=0)

    if feature_nodata is not None:
        feature_finite &= np.all(features != feature_nodata, axis=0)

    context_valid = (feature_mask > 0) & feature_finite
    prediction_domain = (district == 1) & context_valid

    if not np.any(prediction_domain):
        raise RuntimeError("No valid prediction cells found.")

    # Exact training-time normalization.
    features = (
        features - means[:, None, None]
    ) / stds[:, None, None]

    features[:, ~context_valid] = 0.0

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    block_grid = create_block_grid(transform, height, width)

    output = np.full((height, width), NODATA, dtype=np.float32)

    total_patches = 0
    total_predicted = 0

    logger.info("Starting block-aware full-district inference.")
    logger.info("Prediction cells: %d", int(prediction_domain.sum()))

    with torch.no_grad():
        for row in range(0, height, PATCH_SIZE):
            for col in range(0, width, PATCH_SIZE):

                row_end = min(row + PATCH_SIZE, height)
                col_end = min(col + PATCH_SIZE, width)

                tile_height = row_end - row
                tile_width = col_end - col

                block_tile = block_grid[row:row_end, col:col_end]
                district_tile = prediction_domain[row:row_end, col:col_end]

                tile_blocks = np.unique(block_tile[district_tile])

                for block in tile_blocks:
                    block_mask = block_tile == block
                    assign_mask = district_tile & block_mask

                    if not np.any(assign_mask):
                        continue

                    x_patch = np.zeros(
                        (len(FEATURE_BANDS), PATCH_SIZE, PATCH_SIZE),
                        dtype=np.float32,
                    )

                    feature_tile = features[
                        :, row:row_end, col:col_end
                    ].copy()

                    # Same geographic block-context masking as training.
                    feature_tile[:, ~block_mask] = 0.0

                    x_patch[:, :tile_height, :tile_width] = feature_tile

                    tensor = torch.from_numpy(
                        x_patch[None]
                    ).to(DEVICE)

                    prediction = model(tensor).cpu().numpy()[0, 0]

                    # Restore original KSDMA target units.
                    prediction = (
                        prediction * target_std + target_mean
                    )

                    output_tile = output[row:row_end, col:col_end]
                    output_tile[assign_mask] = prediction[:tile_height, :tile_width][assign_mask]

                    total_patches += 1
                    total_predicted += int(assign_mask.sum())

            if row % 160 == 0:
                logger.info("Processed raster rows %d/%d", row, height)

    if total_predicted != int(prediction_domain.sum()):
        raise RuntimeError(
            "Prediction coverage mismatch: "
            f"{total_predicted} vs {int(prediction_domain.sum())}"
        )

    if not np.isfinite(output[prediction_domain]).all():
        raise RuntimeError("Non-finite predictions found.")

    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
    )

    with rasterio.open(OUTPUT_FILE, "w", **profile) as dst:
        dst.write(output, 1)
        dst.set_band_description(1, "U-Net predicted KSDMA hazard index")
        dst.update_tags(
            model="Original U-Net",
            checkpoint_epoch=str(checkpoint["epoch"]),
            district=cfg.DISTRICT_NAME,
            scenario=cfg.SCENARIO_NAME,
            patch_size=str(PATCH_SIZE),
            block_size_metres=str(BLOCK_SIZE),
            prediction_domain="District cells with valid input features",
            caveat=(
                "Predictions outside the KSDMA modelled mask are "
                "extrapolations beyond the supervised target domain."
            ),
        )

    predictions = output[prediction_domain]

    metadata = {
        "model": "Original U-Net",
        "checkpoint": str(MODEL_FILE),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "output_file": str(OUTPUT_FILE),
        "feature_names": FEATURE_NAMES,
        "feature_bands_1_based": FEATURE_BANDS,
        "patch_size": PATCH_SIZE,
        "block_size_metres": BLOCK_SIZE,
        "prediction_cells": int(total_predicted),
        "inference_patches": total_patches,
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "prediction_mean": float(predictions.mean()),
        "target_mean": target_mean,
        "target_std": target_std,
        "nodata": NODATA,
        "note": (
            "Predictions outside the KSDMA modelled mask are "
            "extrapolations, not independently validated hazard values."
        ),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("U-Net prediction completed.")
    logger.info("Raster: %s", OUTPUT_FILE)
    logger.info("Metadata: %s", METADATA_FILE)


if __name__ == "__main__":
    main()
