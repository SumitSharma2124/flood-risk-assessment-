
from pathlib import Path
import sys

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg

FILES = {
    "features_stack": "features_stack",
    "features_valid_mask": "features_valid_mask",
    "ksdma_target": "ksdma_100yr_90m",
    "ksdma_modelled_mask": "ksdma_modelled_mask",
    "district_mask": "district_mask",
}


def main():
    datasets = {}

    for label, key in FILES.items():
        path = cfg.output_path(key)

        if not path.exists():
            raise FileNotFoundError(path)

        datasets[label] = rasterio.open(path)

    try:
        reference = datasets["district_mask"]

        print("\n=== RASTER METADATA ===")

        for label, src in datasets.items():
            print(f"\n{label}")
            print("Shape:", (src.height, src.width))
            print("Bands:", src.count)
            print("CRS:", src.crs)
            print("Transform:", src.transform)
            print("NoData:", src.nodata)

            if (
                src.width != reference.width
                or src.height != reference.height
                or src.crs != reference.crs
                or src.transform != reference.transform
            ):
                raise RuntimeError(f"Grid mismatch: {label}")

        stack = datasets["features_stack"].read().astype("float32")
        feature_mask = datasets["features_valid_mask"].read(1)
        target = datasets["ksdma_target"].read(1).astype("float64")
        modelled_mask = datasets["ksdma_modelled_mask"].read(1)
        district = datasets["district_mask"].read(1) == 1

        print("\n=== INPUT COUNTS ===")
        print("District cells:", int(district.sum()))
        print("Feature bands:", stack.shape[0])
        print("Feature-valid cells:", int((feature_mask > 0).sum()))
        print("KSDMA-modelled cells:", int((modelled_mask > 0).sum()))

        print("\n=== FEATURE VALIDITY ===")

        for i in range(stack.shape[0]):
            band = stack[i]
            valid = district & np.isfinite(band)

            if datasets["features_stack"].nodata is not None:
                valid &= band != datasets["features_stack"].nodata

            print(
                f"Band {i + 1}: "
                f"{int(valid.sum())} valid district cells"
            )

        print("\n=== KSDMA TARGET ===")

        target_valid = (
            district
            & (modelled_mask > 0)
            & np.isfinite(target)
        )

        if datasets["ksdma_target"].nodata is not None:
            target_valid &= target != datasets["ksdma_target"].nodata

        values = target[target_valid]

        print("Valid target cells:", len(values))

        if len(values):
            print("Min:", float(values.min()))
            print("Mean:", float(values.mean()))
            print("Median:", float(np.median(values)))
            print("P90:", float(np.percentile(values, 90)))
            print("Max:", float(values.max()))

        print("\n=== MASK CONSISTENCY ===")
        print(
            "Modelled cells outside district:",
            int(((modelled_mask > 0) & ~district).sum())
        )
        print(
            "Modelled cells with invalid target:",
            int(((modelled_mask > 0) & district & ~target_valid).sum())
        )

        print("\nInput audit completed.")

    finally:
        for src in datasets.values():
            src.close()


if __name__ == "__main__":
    main()
