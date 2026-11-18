
from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "XGBoost": ROOT / "outputs/rasters/09_xgb_full_district_prediction.tif",
    "U-Net": ROOT / "outputs/rasters/10_unet_full_district_prediction.tif",
    "Exposure": ROOT / "data/processed/exposure_score.tif",
    "Vulnerability": ROOT / "data/processed/vulnerability_score.tif",
}

datasets = {}

try:
    for name, path in FILES.items():
        print(f"\n{'=' * 55}")
        print(f"{name}: {path}")

        if not path.exists():
            print("ERROR: File not found")
            continue

        ds = rasterio.open(path)
        datasets[name] = ds

        arr = ds.read(1, masked=True)
        values = arr.compressed()

        print("Shape:", ds.height, "x", ds.width)
        print("CRS:", ds.crs)
        print("Transform:", ds.transform)
        print("Resolution:", ds.res)
        print("NoData:", ds.nodata)
        print("Valid cells:", values.size)

        if values.size:
            print("Minimum:", float(values.min()))
            print("Maximum:", float(values.max()))
            print("Mean:", float(values.mean()))
            print("NaN:", int(np.isnan(values).sum()))
            print("Infinite:", int(np.isinf(values).sum()))

    print(f"\n{'=' * 55}")
    print("ALIGNMENT CHECK")

    if len(datasets) == 3:
        reference = datasets["Exposure"]

        for name, ds in datasets.items():
            checks = {
                "Shape": (ds.shape == reference.shape),
                "CRS": (ds.crs == reference.crs),
                "Transform": (ds.transform == reference.transform),
                "Resolution": (ds.res == reference.res),
            }

            print(f"\n{name}:")
            for check, passed in checks.items():
                print(f"  {check}: {'PASS' if passed else 'FAIL'}")

        if all(
            ds.shape == reference.shape
            and ds.crs == reference.crs
            and ds.transform == reference.transform
            for ds in datasets.values()
        ):
            print("\nRESULT: All three rasters are grid-aligned.")
        else:
            print("\nRESULT: Alignment issues detected.")

finally:
    for ds in datasets.values():
        ds.close()
