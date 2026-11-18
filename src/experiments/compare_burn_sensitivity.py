
from pathlib import Path
import json

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "outputs" / "sensitivity_task4"

BASELINE = {
    "flow_accumulation": ROOT / "data/processed/flow_accumulation_90m.tif",
    "hand": ROOT / "data/processed/hand_90m.tif",
    "twi": ROOT / "data/processed/twi_90m.tif",
}

VARIANTS = ["burn_0m", "burn_2m", "burn_5m", "burn_10m"]
MASK_PATH = ROOT / "data/processed/district_mask.tif"


def read_on_reference_grid(path, reference):
    with rasterio.open(path) as src:
        array = np.full(
            (reference.height, reference.width),
            -9999.0,
            dtype="float32",
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=array,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            dst_nodata=-9999.0,
            resampling=Resampling.nearest,
        )

    return array


def read_experiment(path):
    with rasterio.open(path) as src:
        return src.read(1).astype("float64"), src.nodata


def compare(reference, candidate, mask):
    valid = (
        mask
        & np.isfinite(reference)
        & np.isfinite(candidate)
        & (reference != -9999)
        & (candidate != -9999)
    )

    a = reference[valid]
    b = candidate[valid]

    if not len(a):
        return {"valid_cells": 0}

    diff = b - a
    correlation = spearmanr(a, b).statistic

    return {
        "valid_cells": int(len(a)),
        "mean_absolute_difference": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "spearman_correlation": float(correlation),
        "changed_cells_pct": float(
            np.mean(np.abs(diff) > 1e-6) * 100
        ),
        "cells_difference_over_1": float(
            np.mean(np.abs(diff) > 1) * 100
        ),
        "reference_mean": float(np.mean(a)),
        "candidate_mean": float(np.mean(b)),
    }


def main():
    with rasterio.open(MASK_PATH) as mask_src:
        mask = mask_src.read(1) == 1
        reference_grid = mask_src

        results = {}

        for layer, baseline_path in BASELINE.items():
            baseline = read_on_reference_grid(
                baseline_path, reference_grid
            ).astype("float64")

            results[layer] = {}

            for variant in VARIANTS:
                candidate_path = EXP / variant / f"{layer}.tif"
                candidate, _ = read_experiment(candidate_path)

                if candidate.shape != mask.shape:
                    raise ValueError(
                        f"Experimental grid mismatch: {candidate_path}"
                    )

                results[layer][variant] = compare(
                    baseline, candidate, mask
                )

    output = EXP / "baseline_comparison.json"
    output.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"\nSaved comparison: {output}")


if __name__ == "__main__":
    main()
