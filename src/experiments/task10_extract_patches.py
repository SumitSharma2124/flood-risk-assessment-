
from pathlib import Path
import csv
import json

import numpy as np
import rasterio


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

BLOCK_CSV = TABLES / "task9_block_assignments.csv"

FEATURE_STATS_FILE = TABLES / "10_unet_feature_normalisation.json"
TARGET_STATS_FILE = TABLES / "10_unet_target_normalisation.json"

OUTPUT_DIR = DATA / "task10_patches"
MANIFEST_FILE = TABLES / "10_unet_patch_manifest.csv"
SUMMARY_FILE = TABLES / "10_unet_patch_extraction_summary.json"

PATCH_SIZE = 16
BLOCK_SIZE = 5000

FEATURE_BANDS = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]

FEATURE_NAMES = [
    "elevation",
    "slope",
    "twi",
    "distance_waterway",
    "trees",
    "cropland",
    "builtup",
    "water",
    "wetland",
    "mangrove",
]

SPLIT_IDS = {
    "train": 1,
    "validation": 2,
    "test": 3,
}

ID_TO_SPLIT = {
    value: key for key, value in SPLIT_IDS.items()
}


# ============================================================
# RASTER HELPERS
# ============================================================

def read_raster(name):
    path = DATA / name

    if not path.exists():
        raise FileNotFoundError(f"Missing raster: {path}")

    with rasterio.open(path) as src:
        return (
            src.read(1),
            src.transform,
            src.crs,
            src.nodata,
        )


def validate_grid(reference_transform, reference_crs, name):
    path = DATA / name

    with rasterio.open(path) as src:
        if src.transform != reference_transform:
            raise ValueError(f"Grid transform mismatch: {name}")

        if src.crs != reference_crs:
            raise ValueError(f"CRS mismatch: {name}")


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing statistics file: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_block_assignments():
    assignments = {}

    with open(BLOCK_CSV, newline="", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            block = item["block_id"]
            split = item["split"].strip().lower()

            if split not in SPLIT_IDS:
                raise ValueError(
                    f"Invalid split '{split}' for block {block}"
                )

            assignments[block] = split

    if not assignments:
        raise RuntimeError("No block assignments found.")

    return assignments


# ============================================================
# BLOCK GRID
# ============================================================

def create_block_grid(transform, height, width):
    rows, cols = np.indices((height, width))

    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e

    bx = np.floor(x / BLOCK_SIZE).astype(np.int32)
    by = np.floor(y / BLOCK_SIZE).astype(np.int32)

    return np.char.add(
        np.char.add(bx.astype(str), "_"),
        by.astype(str),
    )


# ============================================================
# PATCH STORAGE
# ============================================================

def create_split_storage():
    return {
        name: {
            "X": [],
            "y": [],
            "mask": [],
            "row": [],
            "col": [],
            "block_id": [],
            "valid_height": [],
            "valid_width": [],
            "label_count": [],
        }
        for name in SPLIT_IDS
    }


def add_patch(
    storage,
    split_name,
    x_patch,
    y_patch,
    mask_patch,
    row,
    col,
    block,
    valid_height,
    valid_width,
    label_count,
):
    item = storage[split_name]

    item["X"].append(x_patch)
    item["y"].append(y_patch)
    item["mask"].append(mask_patch)

    item["row"].append(row)
    item["col"].append(col)
    item["block_id"].append(block)

    item["valid_height"].append(valid_height)
    item["valid_width"].append(valid_width)
    item["label_count"].append(label_count)


# ============================================================
# MAIN EXTRACTION
# ============================================================

def main():

    print("\nTASK 10 — U-NET PATCH EXTRACTION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Protect existing outputs
    # --------------------------------------------------------

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Patch output directory is not empty: {OUTPUT_DIR}"
        )

    for path in [MANIFEST_FILE, SUMMARY_FILE]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    # --------------------------------------------------------
    # 2. Load feature stack
    # --------------------------------------------------------

    feature_path = DATA / "features_stack.tif"

    with rasterio.open(feature_path) as src:
        features = src.read(FEATURE_BANDS).astype(np.float32)

        transform = src.transform
        crs = src.crs

        height = src.height
        width = src.width

        feature_nodata = src.nodata

    print("Feature stack:", features.shape)

    # Verify all other rasters use the same grid.
    required_rasters = [
        "task9_spatial_split.tif",
        "district_mask.tif",
        "features_valid_mask.tif",
        "ksdma_modelled_mask.tif",
        "ksdma_100yr_90m.tif",
    ]

    for name in required_rasters:
        validate_grid(transform, crs, name)

    # --------------------------------------------------------
    # 3. Load masks and target
    # --------------------------------------------------------

    split, _, _, _ = read_raster(
        "task9_spatial_split.tif"
    )

    district, _, _, _ = read_raster(
        "district_mask.tif"
    )

    feature_valid, _, _, _ = read_raster(
        "features_valid_mask.tif"
    )

    modelled, _, _, _ = read_raster(
        "ksdma_modelled_mask.tif"
    )

    target, _, _, target_nodata = read_raster(
        "ksdma_100yr_90m.tif"
    )

    # --------------------------------------------------------
    # 4. Build eligibility masks
    # --------------------------------------------------------

    feature_finite = np.all(
        np.isfinite(features),
        axis=0,
    )

    if feature_nodata is not None:
        feature_finite &= np.all(
            features != feature_nodata,
            axis=0,
        )

    context_valid = (
        (feature_valid > 0)
        & feature_finite
    )

    target_valid = (
        (modelled > 0)
        & np.isfinite(target)
    )

    if target_nodata is not None:
        target_valid &= target != target_nodata

    eligible = (
        (district == 1)
        & context_valid
        & target_valid
        & np.isin(split, [1, 2, 3])
    )

    print("Eligible supervised cells:", int(eligible.sum()))

    if not np.any(eligible):
        raise RuntimeError("No eligible supervised cells found.")

    # --------------------------------------------------------
    # 5. Load normalisation statistics
    # --------------------------------------------------------

    feature_stats = load_json(FEATURE_STATS_FILE)
    target_stats = load_json(TARGET_STATS_FILE)

    means = np.array(
        [feature_stats[name]["mean"] for name in FEATURE_NAMES],
        dtype=np.float32,
    )

    stds = np.array(
        [feature_stats[name]["std"] for name in FEATURE_NAMES],
        dtype=np.float32,
    )

    if not np.all(np.isfinite(means)):
        raise ValueError("Invalid feature means.")

    if not np.all(np.isfinite(stds)) or np.any(stds <= 0):
        raise ValueError("Invalid feature standard deviations.")

    target_mean = float(target_stats["mean"])
    target_std = float(target_stats["std"])

    if not np.isfinite(target_mean) or target_std <= 0:
        raise ValueError("Invalid target normalisation statistics.")

    # --------------------------------------------------------
    # 6. Normalise feature stack
    # --------------------------------------------------------

    features = (
        features - means[:, None, None]
    ) / stds[:, None, None]

    # Invalid feature values are replaced with zero after
    # normalisation. Zero therefore represents the training mean.
    features[:, ~context_valid] = 0.0

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Standardise target using training-only statistics.
    target_normalised = (
        target.astype(np.float32) - target_mean
    ) / target_std

    target_normalised = np.nan_to_num(
        target_normalised,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # --------------------------------------------------------
    # 7. Recover exact geographic block assignments
    # --------------------------------------------------------

    block_assignments = load_block_assignments()

    block_grid = create_block_grid(
        transform,
        height,
        width,
    )

    occupied_blocks = np.unique(block_grid[eligible])

    missing_blocks = [
        block for block in occupied_blocks
        if block not in block_assignments
    ]

    if missing_blocks:
        raise RuntimeError(
            f"Missing block assignments: {missing_blocks[:10]}"
        )

    # Confirm every eligible pixel agrees with the original
    # Task 9 block assignment.
    for block in occupied_blocks:
        block_mask = eligible & (block_grid == block)

        expected_split = SPLIT_IDS[
            block_assignments[block]
        ]

        actual_splits = np.unique(split[block_mask])

        if (
            len(actual_splits) != 1
            or int(actual_splits[0]) != expected_split
        ):
            raise RuntimeError(
                f"Split mismatch in geographic block {block}"
            )

    print("Verified geographic blocks:", len(occupied_blocks))

    # --------------------------------------------------------
    # 8. Prepare patch storage and coverage tracking
    # --------------------------------------------------------

    storage = create_split_storage()

    # Every eligible supervised cell must appear exactly once.
    coverage = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    patch_size = PATCH_SIZE

    # --------------------------------------------------------
    # 9. Extract globally aligned non-overlapping tiles
    # --------------------------------------------------------

    total_tiles = 0

    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):

            row_end = min(row + patch_size, height)
            col_end = min(col + patch_size, width)

            tile_height = row_end - row
            tile_width = col_end - col

            block_tile = block_grid[
                row:row_end,
                col:col_end,
            ]

            eligible_tile = eligible[
                row:row_end,
                col:col_end,
            ]

            # Only blocks containing supervised labels in this
            # tile need a separate patch.
            tile_blocks = np.unique(
                block_tile[eligible_tile]
            )

            for block in tile_blocks:

                block = str(block)
                split_name = block_assignments[block]
                expected_split = SPLIT_IDS[split_name]

                block_mask = block_tile == block

                label_mask = (
                    eligible_tile
                    & block_mask
                )

                # Ensure this patch contains labels from only
                # its assigned geographic split.
                patch_splits = np.unique(
                    split[
                        row:row_end,
                        col:col_end
                    ][label_mask]
                )

                if (
                    len(patch_splits) != 1
                    or int(patch_splits[0]) != expected_split
                ):
                    raise RuntimeError(
                        f"Mixed supervised splits in block {block}"
                    )

                label_count = int(label_mask.sum())

                if label_count == 0:
                    continue

                # ------------------------------------------------
                # Input patch
                # ------------------------------------------------

                x_patch = np.zeros(
                    (len(FEATURE_BANDS), patch_size, patch_size),
                    dtype=np.float32,
                )

                feature_tile = features[
                    :,
                    row:row_end,
                    col:col_end,
                ].copy()

                # Prevent feature context from crossing geographic
                # block boundaries.
                feature_tile[:, ~block_mask] = 0.0

                x_patch[
                    :,
                    :tile_height,
                    :tile_width,
                ] = feature_tile

                # ------------------------------------------------
                # Target patch
                # ------------------------------------------------

                y_patch = np.zeros(
                    (1, patch_size, patch_size),
                    dtype=np.float32,
                )

                target_tile = target_normalised[
                    row:row_end,
                    col:col_end,
                ]

                y_patch[
                    0,
                    :tile_height,
                    :tile_width,
                ][label_mask] = target_tile[label_mask]

                # ------------------------------------------------
                # Loss mask
                # ------------------------------------------------

                mask_patch = np.zeros(
                    (1, patch_size, patch_size),
                    dtype=np.uint8,
                )

                mask_patch[
                    0,
                    :tile_height,
                    :tile_width,
                ] = label_mask.astype(np.uint8)

                # ------------------------------------------------
                # Coverage tracking
                # ------------------------------------------------

                coverage_tile = coverage[
                    row:row_end,
                    col:col_end,
                ]

                if np.any(coverage_tile[label_mask] != 0):
                    raise RuntimeError(
                        "Duplicate supervised cell detected."
                    )

                coverage_tile[label_mask] += 1

                # ------------------------------------------------
                # Store patch
                # ------------------------------------------------

                add_patch(
                    storage=storage,
                    split_name=split_name,
                    x_patch=x_patch,
                    y_patch=y_patch,
                    mask_patch=mask_patch,
                    row=row,
                    col=col,
                    block=block,
                    valid_height=tile_height,
                    valid_width=tile_width,
                    label_count=label_count,
                )

                total_tiles += 1

    # --------------------------------------------------------
    # 10. Verify complete and unique supervised coverage
    # --------------------------------------------------------

    if not np.all(coverage[eligible] == 1):
        missing = int(np.sum(coverage[eligible] == 0))
        duplicate = int(np.sum(coverage[eligible] > 1))

        raise RuntimeError(
            f"Coverage failure: missing={missing}, "
            f"duplicate={duplicate}"
        )

    if np.any(coverage[~eligible] != 0):
        raise RuntimeError(
            "Non-eligible cells unexpectedly entered the loss mask."
        )

    print("\nCOVERAGE VERIFICATION PASSED")
    print("Eligible cells:", int(eligible.sum()))
    print("Covered cells:", int(coverage.sum()))
    print("Duplicate supervised cells: 0")

    # --------------------------------------------------------
    # 11. Prepare output directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    summary = {
        "patch_size": PATCH_SIZE,
        "block_size_metres": BLOCK_SIZE,
        "input_channels": len(FEATURE_BANDS),
        "feature_names": FEATURE_NAMES,
        "feature_bands": FEATURE_BANDS,
        "target": "ksdma_100yr_90m",
        "target_normalisation": {
            "mean": target_mean,
            "std": target_std,
        },
        "strategy": (
            "Globally aligned non-overlapping tiles, "
            "split into block-specific patches with "
            "block-masked feature context."
        ),
        "eligible_cells": int(eligible.sum()),
        "covered_cells": int(coverage.sum()),
        "duplicate_supervised_cells": 0,
        "splits": {},
    }

    # --------------------------------------------------------
    # 12. Save independent split datasets
    # --------------------------------------------------------

    for split_name in ["train", "validation", "test"]:

        item = storage[split_name]
        count = len(item["X"])

        if count == 0:
            raise RuntimeError(
                f"No patches generated for {split_name}"
            )

        output_path = OUTPUT_DIR / f"{split_name}.npz"

        arrays = {
            "X": np.stack(item["X"]),
            "y": np.stack(item["y"]),
            "mask": np.stack(item["mask"]),
            "row": np.asarray(item["row"], dtype=np.int32),
            "col": np.asarray(item["col"], dtype=np.int32),
            "block_id": np.asarray(item["block_id"], dtype=str),
            "valid_height": np.asarray(
                item["valid_height"], dtype=np.uint8
            ),
            "valid_width": np.asarray(
                item["valid_width"], dtype=np.uint8
            ),
            "label_count": np.asarray(
                item["label_count"], dtype=np.int32
            ),
        }

        np.savez_compressed(
            output_path,
            **arrays,
        )

        total_labels = int(arrays["label_count"].sum())

        summary["splits"][split_name] = {
            "patches": count,
            "supervised_cells": total_labels,
            "file": str(output_path.relative_to(ROOT)),
            "input_shape": list(arrays["X"].shape),
            "target_shape": list(arrays["y"].shape),
            "mask_shape": list(arrays["mask"].shape),
        }

        for index in range(count):
            manifest_rows.append({
                "split": split_name,
                "patch_index": index,
                "row": int(arrays["row"][index]),
                "col": int(arrays["col"][index]),
                "block_id": str(arrays["block_id"][index]),
                "valid_height": int(arrays["valid_height"][index]),
                "valid_width": int(arrays["valid_width"][index]),
                "label_count": int(arrays["label_count"][index]),
            })

        print(f"\n{split_name.upper()}")
        print("Patches:", count)
        print("Supervised cells:", total_labels)
        print("Saved:", output_path)

    # --------------------------------------------------------
    # 13. Save patch manifest and summary
    # --------------------------------------------------------

    with open(
        MANIFEST_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "split",
            "patch_index",
            "row",
            "col",
            "block_id",
            "valid_height",
            "valid_width",
            "label_count",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 65)
    print("TASK 10 PATCH EXTRACTION COMPLETED")
    print("=" * 65)

    print("Total patches:", total_tiles)
    print("Total supervised cells:", int(coverage.sum()))
    print("Patch directory:", OUTPUT_DIR)
    print("Manifest:", MANIFEST_FILE)
    print("Summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
