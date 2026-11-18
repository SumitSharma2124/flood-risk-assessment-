
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parents[1]

FLOOD_FILE = (
    ROOT / "data/raw_data/flood_extent/kerala_flood_2018_s1.tif"
)
REFERENCE_FILE = ROOT / "data/processed/ksdma_100yr_90m.tif"
DISTRICT_FILE = ROOT / "data/processed/district_mask.tif"


def inspect(path, name):
    print(f"\n{'=' * 60}")
    print(name)
    print("File:", path)

    with rasterio.open(path) as ds:
        print("Dimensions:", ds.width, "x", ds.height)
        print("Bands:", ds.count)
        print("Data type:", ds.dtypes)
        print("CRS:", ds.crs)
        print("Resolution:", ds.res)
        print("Transform:", ds.transform)
        print("Bounds:", ds.bounds)
        print("NoData:", ds.nodata)
        print("Pixel interpretation:", ds.colorinterp)

        if ds.crs:
            print(
                "Bounds in project CRS:",
                transform_bounds(
                    ds.crs,
                    "EPSG:32643",
                    *ds.bounds,
                    densify_pts=21
                )
            )

        data = ds.read(1, masked=True)
        values = data.compressed()

        print("Valid pixels:", values.size)

        if values.size == 0:
            print("WARNING: No valid pixels")
            return

        print("Minimum:", float(values.min()))
        print("Maximum:", float(values.max()))

        if np.issubdtype(values.dtype, np.integer):
            unique, counts = np.unique(values, return_counts=True)

            if len(unique) <= 100:
                print("Unique values and counts:")
                for value, count in zip(unique, counts):
                    print(f"  {value}: {count}")
            else:
                print("Unique values: more than 100")
                print("Sample values:", unique[:30])
        else:
            print("Percentiles:")
            for p in [0, 1, 25, 50, 75, 99, 100]:
                print(f"  P{p}: {float(np.percentile(values, p))}")

            unique = np.unique(values)
            if len(unique) <= 100:
                print("Unique values:", unique.tolist())
            else:
                print("Unique values: more than 100")


inspect(FLOOD_FILE, "2018 SENTINEL-1 FLOOD EXTENT")
inspect(REFERENCE_FILE, "KSDMA REFERENCE GRID")
inspect(DISTRICT_FILE, "ALAPPUZHA DISTRICT MASK")

print(f"\n{'=' * 60}")
print("DIRECT GRID ALIGNMENT CHECK")

with rasterio.open(FLOOD_FILE) as flood, \
     rasterio.open(REFERENCE_FILE) as ref:

    print("Same CRS:", flood.crs == ref.crs)
    print("Same dimensions:", flood.shape == ref.shape)
    print("Same transform:", flood.transform == ref.transform)
    print("Same resolution:", flood.res == ref.res)
