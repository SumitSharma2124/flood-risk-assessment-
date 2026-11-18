
from pathlib import Path
import json

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs" / "tables"

def read(name):
    with rasterio.open(DATA / name) as src:
        return src.read(1), src.nodata


def main():
    target, nodata = read("ksdma_100yr_90m.tif")
    split, _ = read("task9_spatial_split.tif")
    district, _ = read("district_mask.tif")
    features, _ = read("features_valid_mask.tif")
    modelled, _ = read("ksdma_modelled_mask.tif")

    valid = (
        (split == 1)
        & (district == 1)
        & (features > 0)
        & (modelled > 0)
        & np.isfinite(target)
    )

    if nodata is not None:
        valid &= target != nodata

    values = target[valid].astype(np.float64)

    if len(values) == 0:
        raise RuntimeError("No valid training targets found.")

    mean = float(values.mean())
    std = float(values.std())

    if std == 0:
        raise RuntimeError("Target standard deviation is zero.")

    stats = {
        "target": "ksdma_100yr_90m",
        "transformation": "standard_score",
        "mean": mean,
        "std": std,
        "count": int(len(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "10_unet_target_normalisation.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print("TASK 10 — TARGET NORMALISATION")
    print("=" * 50)
    print("Training cells:", len(values))
    print(f"Mean: {mean:.6f}")
    print(f"Standard deviation: {std:.6f}")
    print(f"Minimum: {values.min():.6f}")
    print(f"Maximum: {values.max():.6f}")
    print("Saved:", path)


if __name__ == "__main__":
    main()
