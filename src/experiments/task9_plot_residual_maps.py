
"""
Task 9 — XGBoost validation residual visualisation.

Generates:
1. Validation residual map
2. Validation absolute-error map

Uses validation cells only. Test data is never accessed.

Run from project root:
    python src/experiments/task9_plot_residual_maps.py
"""

from pathlib import Path
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# --------------------------------------------------
# PATHS AND CONFIG
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))
import config as cfg

RASTER_DIR = ROOT / "outputs" / "rasters"
FIGURE_DIR = ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RESIDUAL_PATH = RASTER_DIR / "09_xgb_validation_residuals.tif"
ERROR_PATH = RASTER_DIR / "09_xgb_validation_absolute_error.tif"
SPLIT_PATH = ROOT / "data" / "processed" / "task9_spatial_split.tif"


# --------------------------------------------------
# RASTER READER
# --------------------------------------------------

def read_config_raster(key):
    path = Path(cfg.output_path(key))

    if not path.is_absolute():
        path = ROOT / path

    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


def read_diagnostic_raster(path):
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

    # Convert NoData to NaN.
    if nodata is not None:
        array[array == nodata] = np.nan

    # Also remove infinities.
    array[~np.isfinite(array)] = np.nan

    return array, transform, crs, profile


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Loading diagnostic rasters...")

    residual, transform, crs, residual_profile = (
        read_diagnostic_raster(RESIDUAL_PATH)
    )

    absolute_error, error_transform, error_crs, _ = (
        read_diagnostic_raster(ERROR_PATH)
    )

    print("Loading validation split...")

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Split raster not found: {SPLIT_PATH}")

    with rasterio.open(SPLIT_PATH) as src:
        split = src.read(1)
        split_transform = src.transform
        split_crs = src.crs

    # --------------------------------------------------
    # GRID VALIDATION
    # --------------------------------------------------

    if residual.shape != absolute_error.shape:
        raise ValueError("Residual and absolute-error raster shapes differ.")

    if residual.shape != split.shape:
        raise ValueError("Diagnostic and split raster shapes differ.")

    if transform != error_transform or transform != split_transform:
        raise ValueError("Raster transforms do not match.")

    if crs != error_crs or crs != split_crs:
        raise ValueError("Raster coordinate reference systems do not match.")

    # --------------------------------------------------
    # VALIDATION MASK
    # --------------------------------------------------

    # Split values:
    # 0 = excluded
    # 1 = train
    # 2 = validation
    # 3 = test

    validation_mask = split == 2

    # Ensure both maps contain only actual validation cells.
    residual[~validation_mask] = np.nan
    absolute_error[~validation_mask] = np.nan

    # Absolute error cannot be negative.
    absolute_error[absolute_error < 0] = np.nan

    # A valid residual must also have a corresponding valid absolute error.
    valid_mask = (
        validation_mask
        & np.isfinite(residual)
        & np.isfinite(absolute_error)
    )

    residual[~valid_mask] = np.nan
    absolute_error[~valid_mask] = np.nan

    valid_residual = residual[np.isfinite(residual)]
    valid_error = absolute_error[np.isfinite(absolute_error)]

    if len(valid_residual) == 0 or len(valid_error) == 0:
        raise RuntimeError("No valid validation cells available for plotting.")

    print("\n=== RASTER CHECK ===")
    print(f"Validation cells in split: {validation_mask.sum():,}")
    print(f"Valid residual cells: {len(valid_residual):,}")
    print(f"Valid absolute-error cells: {len(valid_error):,}")
    print(f"Residual minimum: {valid_residual.min():.3f}")
    print(f"Residual maximum: {valid_residual.max():.3f}")
    print(f"Absolute-error minimum: {valid_error.min():.3f}")
    print(f"Absolute-error maximum: {valid_error.max():.3f}")

    # --------------------------------------------------
    # MAP EXTENT
    # --------------------------------------------------

    height, width = residual.shape

    left = transform.c
    top = transform.f
    right = left + width * transform.a
    bottom = top + height * transform.e

    extent = [left, right, bottom, top]

    # --------------------------------------------------
    # MAP 1: RESIDUALS
    # --------------------------------------------------

    residual_limit = float(
        np.percentile(np.abs(valid_residual), 98)
    )
    residual_limit = max(residual_limit, 1.0)

    fig, ax = plt.subplots(figsize=(10, 11))

    image = ax.imshow(
        residual,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(
            vmin=-residual_limit,
            vcenter=0,
            vmax=residual_limit,
        ),
        extent=extent,
        interpolation="nearest",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Residual (actual - predicted)",
        shrink=0.8,
    )

    ax.set_title(
        "XGBoost Validation Residuals\n"
        "Red = Underprediction | Blue = Overprediction"
    )

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()

    residual_output = (
        FIGURE_DIR / "09_xgb_validation_residuals.png"
    )

    plt.savefig(
        residual_output,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------
    # MAP 2: ABSOLUTE ERROR
    # --------------------------------------------------

    error_limit = float(
        np.percentile(valid_error, 98)
    )
    error_limit = max(error_limit, 1.0)

    fig, ax = plt.subplots(figsize=(10, 11))

    image = ax.imshow(
        absolute_error,
        cmap="YlOrRd",
        vmin=0,
        vmax=error_limit,
        extent=extent,
        interpolation="nearest",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Absolute prediction error",
        shrink=0.8,
    )

    ax.set_title(
        "XGBoost Validation Absolute Error"
    )

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()

    error_output = (
        FIGURE_DIR / "09_xgb_validation_absolute_error.png"
    )

    plt.savefig(
        error_output,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------
    # COMPLETE
    # --------------------------------------------------

    print("\nMaps generated successfully:")
    print(residual_output)
    print(error_output)

    print("\nOnly validation cells were plotted.")
    print("Training and test cells were excluded.")
    print("No model was retrained.")


if __name__ == "__main__":
    main()
