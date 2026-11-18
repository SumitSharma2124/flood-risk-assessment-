
from pathlib import Path

import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

FILES = {
    "features": DATA / "features_stack.tif",
    "target": DATA / "ksdma_100yr_90m.tif",
    "target_mask": DATA / "ksdma_modelled_mask.tif",
    "feature_mask": DATA / "features_valid_mask.tif",
    "district_mask": DATA / "district_mask.tif",
    "split": DATA / "task9_spatial_split.tif",
}

FEATURE_BANDS = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]


def main():
    print("TASK 10 — U-NET INPUT AUDIT")
    print("=" * 55)

    print("\nPyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    with rasterio.open(FILES["features"]) as src:
        profile = src.profile
        print("\nFEATURE STACK")
        print("Shape:", (src.height, src.width))
        print("Bands:", src.count)
        print("CRS:", src.crs)
        print("Transform:", src.transform)

        if src.count != 12:
            raise ValueError("Expected 12 feature bands.")

        if not all(1 <= b <= src.count for b in FEATURE_BANDS):
            raise ValueError("Invalid selected feature band.")

    for name, path in FILES.items():
        with rasterio.open(path) as src:
            print(f"\n{name.upper()}")
            print("Shape:", (src.height, src.width))
            print("CRS:", src.crs)
            print("NoData:", src.nodata)

            if (src.height, src.width) != (
                profile["height"], profile["width"]
            ):
                raise ValueError(f"{name} has a different raster shape.")

            if src.crs != profile["crs"]:
                raise ValueError(f"{name} has a different CRS.")

            if src.transform != profile["transform"]:
                raise ValueError(f"{name} has a different grid transform.")

    with rasterio.open(FILES["split"]) as src:
        split = src.read(1)

    with rasterio.open(FILES["district_mask"]) as src:
        district = src.read(1) > 0

    with rasterio.open(FILES["feature_mask"]) as src:
        feature_valid = src.read(1) > 0

    with rasterio.open(FILES["target_mask"]) as src:
        target_valid = src.read(1) > 0

    with rasterio.open(FILES["target"]) as src:
        target = src.read(1).astype(np.float32)
        target_nodata = src.nodata

    target_finite = np.isfinite(target)

    if target_nodata is not None:
        target_finite &= target != target_nodata

    eligible = (
        district
        & feature_valid
        & target_valid
        & target_finite
    )

    print("\nELIGIBLE CELLS")
    print("Total:", int(eligible.sum()))

    print("\nEXISTING SPLIT COUNTS")
    for value, label in [
        (1, "Train"),
        (2, "Validation"),
        (3, "Test"),
    ]:
        count = int((eligible & (split == value)).sum())
        print(f"{label}: {count:,}")

    excluded = int((district & ~eligible).sum())
    print("District cells excluded from supervised learning:", f"{excluded:,}")

    print("\nPATCH CONFIGURATION")
    print("Patch size: 64 x 64")
    print("Input channels:", len(FEATURE_BANDS))
    print("Target: KSDMA 100-year continuous raster")
    print("Split: existing Task 9 geographic split")

    print("\nAUDIT PASSED")


if __name__ == "__main__":
    main()
