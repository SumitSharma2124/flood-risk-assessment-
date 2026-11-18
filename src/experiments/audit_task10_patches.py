
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

PATCH_SIZES = [16, 24, 32, 48, 64]
MIN_LABELS = 64


def read(name):
    with rasterio.open(DATA / name) as src:
        return src.read(1)


def main():
    split = read("task9_spatial_split.tif")
    district = read("district_mask.tif") > 0
    feature_valid = read("features_valid_mask.tif") > 0
    target_valid = read("ksdma_modelled_mask.tif") > 0
    target = read("ksdma_100yr_90m.tif")

    with rasterio.open(DATA / "ksdma_100yr_90m.tif") as src:
        if src.nodata is not None:
            target_valid &= target != src.nodata

    eligible = (
        district
        & feature_valid
        & target_valid
        & np.isfinite(target)
    )

    height, width = split.shape

    print("TASK 10 — RELAXED PATCH FEASIBILITY AUDIT")
    print("=" * 75)
    print("Non-overlapping patches")
    print("Excluded/background pixels allowed")
    print("Mixed supervised splits prohibited")
    print(f"Minimum eligible labels per patch: {MIN_LABELS}\n")

    print(f"{'Size':<12}{'Train':<12}{'Validation':<15}{'Test':<12}")
    print("-" * 55)

    for size in PATCH_SIZES:
        stride = size

        counts = {1: 0, 2: 0, 3: 0}
        label_counts = {1: 0, 2: 0, 3: 0}

        for row in range(0, height - size + 1, stride):
            for col in range(0, width - size + 1, stride):

                s = split[row:row + size, col:col + size]
                e = eligible[row:row + size, col:col + size]

                # Only eligible supervised pixels determine patch ownership.
                supervised_splits = np.unique(s[e])

                # Reject patches with no labels or mixed split labels.
                if len(supervised_splits) != 1:
                    continue

                split_id = int(supervised_splits[0])

                if split_id not in counts:
                    continue

                valid_count = int((e & (s == split_id)).sum())

                if valid_count < MIN_LABELS:
                    continue

                counts[split_id] += 1
                label_counts[split_id] += valid_count

        print(
            f"{size}x{size:<8}"
            f"{counts[1]:<12}"
            f"{counts[2]:<15}"
            f"{counts[3]:<12}"
        )

        print(
            f"  Eligible label occurrences — "
            f"Train: {label_counts[1]:,}, "
            f"Validation: {label_counts[2]:,}, "
            f"Test: {label_counts[3]:,}"
        )

    print("\nAudit only — no patches saved and no data modified.")


if __name__ == "__main__":
    main()
