
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"


def read(name):
    with rasterio.open(DATA / name) as src:
        return src.read(1)


def main():
    split = read("task9_spatial_split.tif")
    district = read("district_mask.tif") > 0
    features = read("features_valid_mask.tif") > 0
    target_mask = read("ksdma_modelled_mask.tif") > 0
    target = read("ksdma_100yr_90m.tif")

    with rasterio.open(DATA / "ksdma_100yr_90m.tif") as src:
        if src.nodata is not None:
            target_mask &= target != src.nodata

    eligible = (
        district
        & features
        & target_mask
        & np.isfinite(target)
    )

    print("TASK 10 — SPLIT GEOMETRY AUDIT")
    print("=" * 55)

    for sid, name in [(1, "Train"), (2, "Validation"), (3, "Test")]:
        mask = eligible & (split == sid)
        rows, cols = np.where(mask)

        if len(rows) == 0:
            print(f"\n{name}: No eligible cells")
            continue

        print(f"\n{name}")
        print("Eligible cells:", len(rows))
        print(
            "Bounding box (row, col):",
            (rows.min(), cols.min(), rows.max(), cols.max())
        )
        print(
            "Occupied 5 km blocks:",
            len(np.unique(
                (rows // 56) * 10000 + (cols // 56)
            ))
        )

    print("\nAudit complete. No data modified.")


if __name__ == "__main__":
    main()
