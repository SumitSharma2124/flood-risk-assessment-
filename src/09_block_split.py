
from pathlib import Path
import sys
import csv

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg

BLOCK_SIZE = 5000
SEED = 42
N_TRIALS = 1000

SHARES = np.array([0.70, 0.15, 0.15])
NAMES = ["train", "validation", "test"]

SPLIT_RASTER = ROOT / "data" / "processed" / "task9_spatial_split.tif"
BLOCK_CSV = ROOT / "outputs" / "tables" / "task9_block_assignments.csv"


def objective(counts, total):
    actual = counts / total
    return float(np.sum((actual - SHARES) ** 2))


def assign_blocks(block_ids, block_counts):
    rng = np.random.default_rng(SEED)
    total = int(block_counts.sum())

    best_assignment = None
    best_score = float("inf")

    for _ in range(N_TRIALS):
        # Randomise equal-sized blocks while processing large blocks first.
        order = np.arange(len(block_ids))
        rng.shuffle(order)
        order = sorted(order, key=lambda i: block_counts[i], reverse=True)

        counts = np.zeros(3, dtype=np.int64)
        assignment = np.full(len(block_ids), -1, dtype=np.int8)

        for idx in order:
            candidates = []

            for split in range(3):
                proposed = counts.copy()
                proposed[split] += block_counts[idx]

                score = objective(proposed, total)
                candidates.append(score)

            minimum = min(candidates)
            choices = [
                i for i, score in enumerate(candidates)
                if np.isclose(score, minimum)
            ]

            selected = int(rng.choice(choices))
            assignment[idx] = selected
            counts[selected] += block_counts[idx]

        score = objective(counts, total)

        if score < best_score and np.all(counts > 0):
            best_score = score
            best_assignment = assignment.copy()

    if best_assignment is None:
        raise RuntimeError("Could not produce a valid three-way split.")

    return best_assignment


def main():
    with rasterio.open(cfg.output_path("district_mask")) as ds:
        district = ds.read(1) == 1
        profile = ds.profile.copy()
        transform = ds.transform
        height, width = ds.height, ds.width

    with rasterio.open(cfg.output_path("features_valid_mask")) as ds:
        features_valid = ds.read(1) > 0

    with rasterio.open(cfg.output_path("ksdma_modelled_mask")) as ds:
        modelled = ds.read(1) > 0

    with rasterio.open(cfg.output_path("ksdma_100yr_90m")) as ds:
        target = ds.read(1)
        target_nodata = ds.nodata

    eligible = district & features_valid & modelled
    eligible &= np.isfinite(target)

    if target_nodata is not None:
        eligible &= target != target_nodata

    rows, cols = np.where(eligible)

    if len(rows) == 0:
        raise RuntimeError("No eligible target cells found.")

    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e

    bx = np.floor(x / BLOCK_SIZE).astype(int)
    by = np.floor(y / BLOCK_SIZE).astype(int)

    block_ids = np.array([
        f"{a}_{b}" for a, b in zip(bx, by)
    ])

    unique_blocks, inverse, block_counts = np.unique(
        block_ids,
        return_inverse=True,
        return_counts=True
    )

    assignment = assign_blocks(unique_blocks, block_counts)

    # Raster values: 0 = excluded, 1 = train, 2 = validation, 3 = test
    split_raster = np.zeros((height, width), dtype=np.uint8)
    split_raster[rows, cols] = assignment[inverse] + 1

    # Prevent accidental overwriting.
    if SPLIT_RASTER.exists() or BLOCK_CSV.exists():
        raise FileExistsError(
            "A Task 9 split output already exists. "
            "Rename or move it before regenerating."
        )

    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="lzw"
    )

    SPLIT_RASTER.parent.mkdir(parents=True, exist_ok=True)
    BLOCK_CSV.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(SPLIT_RASTER, "w", **profile) as dst:
        dst.write(split_raster, 1)

    with open(BLOCK_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["block_id", "eligible_cells", "split"])

        for i, block in enumerate(unique_blocks):
            writer.writerow([
                block,
                int(block_counts[i]),
                NAMES[int(assignment[i])]
            ])

    print("\n=== TASK 9 SPATIAL SPLIT ===")
    print("Eligible cells:", len(rows))
    print("Occupied blocks:", len(unique_blocks))
    print("Block size:", BLOCK_SIZE, "metres")
    print("Random seed:", SEED)

    for i, name in enumerate(NAMES):
        cells = int(np.sum(assignment[inverse] == i))
        print(
            f"{name}: {cells:,} cells "
            f"({cells / len(rows) * 100:.2f}%)"
        )

    print("\nSplit raster:", SPLIT_RASTER)
    print("Block assignments:", BLOCK_CSV)


if __name__ == "__main__":
    main()
