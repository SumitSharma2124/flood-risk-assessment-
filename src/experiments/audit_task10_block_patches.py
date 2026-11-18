
from pathlib import Path
import csv

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

BLOCK_CSV = ROOT / "outputs" / "tables" / "task9_block_assignments.csv"

BLOCK_SIZE = 5000
PATCH_SIZE = 16
MIN_LABELS = 64

SPLIT_IDS = {
    "train": 1,
    "validation": 2,
    "test": 3,
}


def read_raster(name):
    path = DATA / name

    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.profile.copy(), src.nodata


def load_block_assignments():
    if not BLOCK_CSV.exists():
        raise FileNotFoundError(f"Missing block assignment CSV: {BLOCK_CSV}")

    assignments = {}

    with open(BLOCK_CSV, newline="", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            assignments[item["block_id"]] = item["split"]

    return assignments


def main():
    print("TASK 10 — EXACT BLOCK PATCH AUDIT")
    print("=" * 65)

    # ---------------------------------------------------------
    # 1. Load rasters
    # ---------------------------------------------------------
    split, transform, profile, _ = read_raster(
        "task9_spatial_split.tif"
    )

    district, _, _, _ = read_raster("district_mask.tif")
    feature_valid, _, _, _ = read_raster("features_valid_mask.tif")
    target_mask, _, _, _ = read_raster("ksdma_modelled_mask.tif")

    target, _, _, target_nodata = read_raster(
        "ksdma_100yr_90m.tif"
    )

    # ---------------------------------------------------------
    # 2. Build boolean validity masks
    # ---------------------------------------------------------
    district_valid = district == 1
    features_valid = feature_valid > 0
    modelled_valid = target_mask > 0

    target_valid = np.isfinite(target)

    if target_nodata is not None:
        target_valid &= target != target_nodata

    eligible = (
        district_valid
        & features_valid
        & modelled_valid
        & target_valid
    )

    print("Raster shape:", split.shape)
    print("Eligible supervised cells:", int(eligible.sum()))

    # ---------------------------------------------------------
    # 3. Load exact Task 9 block assignments
    # ---------------------------------------------------------
    block_splits = load_block_assignments()

    print("Original geographic blocks:", len(block_splits))

    # ---------------------------------------------------------
    # 4. Reconstruct actual projected block IDs
    # ---------------------------------------------------------
    height, width = split.shape

    rows, cols = np.indices((height, width))

    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e

    bx = np.floor(x / BLOCK_SIZE).astype(np.int32)
    by = np.floor(y / BLOCK_SIZE).astype(np.int32)

    block_ids = np.char.add(
        np.char.add(bx.astype(str), "_"),
        by.astype(str)
    )

    # ---------------------------------------------------------
    # 5. Verify raster split against original block CSV
    # ---------------------------------------------------------
    print("\nVERIFYING BLOCK ASSIGNMENTS")

    expected_split = {
        "train": 1,
        "validation": 2,
        "test": 3,
    }

    for block, split_name in block_splits.items():
        block_mask = (block_ids == block) & eligible

        if not np.any(block_mask):
            continue

        expected_id = expected_split[split_name]
        actual_ids = np.unique(split[block_mask])

        if len(actual_ids) != 1 or actual_ids[0] != expected_id:
            raise RuntimeError(
                f"Split mismatch in block {block}: "
                f"expected {split_name}, got {actual_ids}"
            )

    print("All occupied blocks match the original CSV.")

    # ---------------------------------------------------------
    # 6. Extract non-overlapping patches block by block
    # ---------------------------------------------------------
    patch_counts = {
        name: 0 for name in SPLIT_IDS
    }

    label_occurrences = {
        name: 0 for name in SPLIT_IDS
    }

    unique_cells = {
        name: set() for name in SPLIT_IDS
    }

    for block, split_name in block_splits.items():

        block_mask = block_ids == block

        if not np.any(block_mask):
            continue

        block_rows, block_cols = np.where(block_mask)

        row_min = int(block_rows.min())
        row_max = int(block_rows.max()) + 1

        col_min = int(block_cols.min())
        col_max = int(block_cols.max()) + 1

        expected_id = SPLIT_IDS[split_name]

        for row in range(
            row_min,
            row_max - PATCH_SIZE + 1,
            PATCH_SIZE
        ):
            for col in range(
                col_min,
                col_max - PATCH_SIZE + 1,
                PATCH_SIZE
            ):

                row_end = row + PATCH_SIZE
                col_end = col + PATCH_SIZE

                patch_blocks = block_ids[
                    row:row_end,
                    col:col_end
                ]

                # Entire patch must belong to one geographic block.
                if not np.all(patch_blocks == block):
                    continue

                patch_split = split[
                    row:row_end,
                    col:col_end
                ]

                patch_eligible = eligible[
                    row:row_end,
                    col:col_end
                ]

                # Excluded pixels are allowed as feature context.
                # Supervised pixels must belong to this block's split.
                supervised_splits = np.unique(
                    patch_split[patch_eligible]
                )

                if len(supervised_splits) == 0:
                    continue

                if (
                    len(supervised_splits) != 1
                    or int(supervised_splits[0]) != expected_id
                ):
                    raise RuntimeError(
                        f"Mixed split labels in block {block}, "
                        f"patch row={row}, col={col}"
                    )

                valid_count = int(patch_eligible.sum())

                if valid_count < MIN_LABELS:
                    continue

                patch_counts[split_name] += 1
                label_occurrences[split_name] += valid_count

                valid_rows, valid_cols = np.where(patch_eligible)

                for dr, dc in zip(valid_rows, valid_cols):
                    unique_cells[split_name].add(
                        (row + int(dr), col + int(dc))
                    )

    # ---------------------------------------------------------
    # 7. Report results
    # ---------------------------------------------------------
    print("\nPATCH CONFIGURATION")
    print("Patch size:", f"{PATCH_SIZE} x {PATCH_SIZE}")
    print("Block size:", f"{BLOCK_SIZE} metres")
    print("Minimum eligible labels:", MIN_LABELS)
    print("Patch overlap: None")

    print("\nRESULTS")
    print("-" * 65)

    for name in ["train", "validation", "test"]:
        print(f"\n{name.upper()}")
        print("Candidate patches:", f"{patch_counts[name]:,}")
        print(
            "Label occurrences:",
            f"{label_occurrences[name]:,}"
        )
        print(
            "Unique supervised cells:",
            f"{len(unique_cells[name]):,}"
        )

        if label_occurrences[name] != len(unique_cells[name]):
            raise RuntimeError(
                f"Duplicate supervised cells detected in {name}"
            )

    print("\nAUDIT PASSED")
    print("No patches saved.")
    print("No source data modified.")


if __name__ == "__main__":
    main()
