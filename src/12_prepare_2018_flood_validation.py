
"""
Task 12: Prepare 2018 Sentinel-1 flood extent for validation.

Aggregates the binary 30m flood raster to the project's
90m Alappuzha grid using average resampling.

Run:
    python src/12_prepare_2018_flood_validation.py
"""

from pathlib import Path
import json
import logging

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data/raw_data/flood_extent/kerala_flood_2018_s1.tif"
REFERENCE = ROOT / "data/processed/ksdma_100yr_90m.tif"
DISTRICT = ROOT / "data/processed/district_mask.tif"
KSDMA_MASK = ROOT / "data/processed/ksdma_modelled_mask.tif"

OUT_DIR = ROOT / "data/processed"
TABLE_DIR = ROOT / "outputs/tables"

FRACTION_OUT = OUT_DIR / "12_2018_flood_fraction_90m.tif"
COVERAGE_OUT = OUT_DIR / "12_2018_flood_coverage_90m.tif"
SUMMARY_OUT = TABLE_DIR / "12_2018_flood_preparation_summary.json"

NODATA = -9999.0

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
log = logging.getLogger("task12")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Opening source flood raster...")

    with rasterio.open(SOURCE) as src, \
         rasterio.open(REFERENCE) as ref, \
         rasterio.open(DISTRICT) as district_ds, \
         rasterio.open(KSDMA_MASK) as mask_ds:

        if src.count != 1:
            raise ValueError("Expected a single-band flood raster.")

        if ref.crs is None or src.crs is None:
            raise ValueError("Source or reference CRS is missing.")

        profile = ref.profile.copy()
        height, width = ref.height, ref.width

        # Read source classification.
        source = src.read(1)

        source_valid = src.read_masks(1) > 0
        source_valid &= np.isin(source, [0, 1])

        if not np.any(source_valid):
            raise ValueError("No valid binary flood pixels found.")

        # Convert to float for average resampling.
        source_float = source.astype(np.float32)
        source_float[~source_valid] = NODATA

        # Output flood fraction.
        flood_fraction = np.full(
            (height, width),
            NODATA,
            dtype=np.float32
        )

        reproject(
            source=source_float,
            destination=flood_fraction,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=NODATA,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=NODATA,
            resampling=Resampling.average,
            init_dest_nodata=True,
        )

        # Reproject source coverage independently.
        source_coverage = source_valid.astype(np.float32)

        coverage_fraction = np.zeros(
            (height, width),
            dtype=np.float32
        )

        reproject(
            source=source_coverage,
            destination=coverage_fraction,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            src_nodata=None,
            dst_nodata=0.0,
            resampling=Resampling.average,
            init_dest_nodata=True,
        )

        # Read project masks.
        district = district_ds.read(1)
        district_valid = district_ds.read_masks(1) > 0
        district_cells = district_valid & (district > 0.5)

        ksdma_mask = mask_ds.read(1)
        ksdma_valid = mask_ds.read_masks(1) > 0
        ksdma_cells = ksdma_valid & (ksdma_mask > 0.5)

        # Only retain cells with at least 99% source coverage.
        coverage_valid = coverage_fraction >= 0.99

        flood_valid = (
            district_cells
            & coverage_valid
            & np.isfinite(flood_fraction)
            & (flood_fraction != NODATA)
        )

        if not np.any(flood_valid):
            raise ValueError(
                "No sufficiently covered Alappuzha cells found."
            )

        common_valid = flood_valid & ksdma_cells

        if not np.any(common_valid):
            raise ValueError(
                "No overlap between flood coverage and KSDMA modelled cells."
            )

        # Preserve fraction only on sufficiently covered district cells.
        flood_output = np.full(
            (height, width),
            NODATA,
            dtype=np.float32
        )
        flood_output[flood_valid] = np.clip(
            flood_fraction[flood_valid],
            0.0,
            1.0
        )

        coverage_output = coverage_fraction.astype(np.float32)

        # Write rasters.
        out_profile = profile.copy()
        out_profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=NODATA,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        with rasterio.open(FRACTION_OUT, "w", **out_profile) as dst:
            dst.write(flood_output, 1)
            dst.set_band_description(
                1, "2018 observed flood fraction"
            )

        coverage_profile = out_profile.copy()
        coverage_profile.update(nodata=NODATA)

        coverage_to_write = np.where(
            district_valid,
            coverage_output,
            NODATA
        ).astype(np.float32)

        with rasterio.open(COVERAGE_OUT, "w", **coverage_profile) as dst:
            dst.write(coverage_to_write, 1)
            dst.set_band_description(
                1, "Fraction of 90m cell covered by source raster"
            )

        # Threshold sensitivity summary.
        fractions = flood_output[flood_valid]
        common_fractions = flood_output[common_valid]

        thresholds = [0.10, 0.25, 0.50, 0.75]

        threshold_summary = {}

        for threshold in thresholds:
            threshold_summary[str(threshold)] = {
                "district_flood_cells": int(
                    np.count_nonzero(fractions >= threshold)
                ),
                "district_flood_share": float(
                    np.mean(fractions >= threshold)
                ),
                "common_domain_flood_cells": int(
                    np.count_nonzero(common_fractions >= threshold)
                ),
                "common_domain_flood_share": float(
                    np.mean(common_fractions >= threshold)
                ),
            }

        report = {
            "source": str(SOURCE),
            "source_crs": str(src.crs),
            "source_resolution": list(src.res),
            "source_valid_pixels": int(source_valid.sum()),
            "source_flooded_pixels": int(
                np.count_nonzero(source[source_valid] == 1)
            ),
            "target_crs": str(ref.crs),
            "target_resolution": list(ref.res),
            "target_shape": [height, width],
            "resampling": "average",
            "coverage_requirement": 0.99,
            "district_cells": int(district_cells.sum()),
            "district_cells_with_valid_flood_coverage": int(
                flood_valid.sum()
            ),
            "district_cells_without_sufficient_coverage": int(
                (district_cells & ~coverage_valid).sum()
            ),
            "common_ksdma_validation_cells": int(common_valid.sum()),
            "flood_fraction_statistics": {
                "minimum": float(fractions.min()),
                "maximum": float(fractions.max()),
                "mean": float(fractions.mean()),
                "median": float(np.median(fractions)),
            },
            "threshold_sensitivity": threshold_summary,
            "outputs": {
                "flood_fraction": str(FRACTION_OUT),
                "coverage": str(COVERAGE_OUT),
            },
            "limitations": [
                "Flood extent is an observed 2018 event, not a 100-year return-period map.",
                "Average resampling represents the fraction of source pixels classified as flooded.",
                "A 0.5 threshold is provisional; threshold sensitivity is retained.",
                "Only cells with at least 99% source coverage are used.",
            ],
        }

        with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

    log.info("Flood fraction raster: %s", FRACTION_OUT)
    log.info("Coverage raster: %s", COVERAGE_OUT)
    log.info("Summary: %s", SUMMARY_OUT)
    log.info("Valid district cells: %d", int(flood_valid.sum()))
    log.info("Common KSDMA validation cells: %d", int(common_valid.sum()))
    log.info("Task 12 preparation completed.")


if __name__ == "__main__":
    main()
