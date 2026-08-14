"""Select one failure, one median, and one best PCDR qualitative case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    """Hash a selection input without importing the detector stack."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--existing-image-cache",
        type=Path,
        help=(
            "Optional prior combined NPZ used only to reuse already selected parent "
            "images when the third-party image payload is unavailable locally."
        ),
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for path in sorted(args.cache_dir.glob("Rollo*A.npz")):
        input_hashes[path.name] = sha256_file(path)
        with np.load(path, allow_pickle=False) as data:
            fold = str(data["fold"])
            for offset in range(len(data["indices"])):
                rows.append({
                    "fold": fold,
                    "index": int(data["indices"][offset]),
                    "filename": str(data["filenames"][offset]),
                    "mask": data["masks"][offset],
                    "hann_field": data["hann_fields"][offset],
                    "pcdr_field": data["pcdr_fields"][offset],
                    "hann_ap": float(data["hann_parent_pixel_ap"][offset]),
                    "pcdr_ap": float(data["pcdr_parent_pixel_ap"][offset]),
                    "delta": float(data["delta_parent_pixel_ap"][offset]),
                })
    if {row["fold"] for row in rows} != {
        "Rollo1A", "Rollo2A", "Rollo3A", "Rollo5A", "Rollo6A", "Rollo7A"
    }:
        raise ValueError("Expected all six candidate-evaluation fold caches")

    fold_order = {
        fold: order
        for order, fold in enumerate(
            ("Rollo1A", "Rollo2A", "Rollo3A", "Rollo5A", "Rollo6A", "Rollo7A")
        )
    }

    def stable_identity(row: dict[str, Any]) -> tuple[int, int, str]:
        return fold_order[row["fold"]], row["index"], row["filename"]

    r2_rows = [row for row in rows if row["fold"] == "Rollo2A"]
    failure = min(
        r2_rows, key=lambda row: (row["delta"], *stable_identity(row))
    )
    remaining = [row for row in rows if row is not failure]
    median_value = float(np.median([row["delta"] for row in rows]))
    median = min(
        remaining,
        key=lambda row: (
            abs(row["delta"] - median_value),
            *stable_identity(row),
        ),
    )
    remaining = [row for row in remaining if row is not median]
    best = min(
        remaining, key=lambda row: (-row["delta"], *stable_identity(row))
    )
    selected = [failure, median, best]
    labels = np.asarray(["Failure", "Median", "Best"])
    cached_images: dict[str, np.ndarray] = {}
    if args.existing_image_cache is not None:
        with np.load(args.existing_image_cache, allow_pickle=False) as cache:
            cached_images = {
                str(filename): image.copy()
                for filename, image in zip(
                    cache["filenames"], cache["images"], strict=True
                )
            }
    selected_images: list[np.ndarray] = []
    for row in selected:
        image_path = args.data_root / "images" / row["filename"]
        if image_path.is_file():
            selected_images.append(
                np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
            )
        elif row["filename"] in cached_images:
            selected_images.append(cached_images[row["filename"]])
        else:
            raise FileNotFoundError(
                f"Missing source image and cached selection: {row['filename']}"
            )
    images = np.stack(selected_images)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        selection_rule=np.asarray(
            "minimum per-parent PCDR-minus-Hann AP within Rollo2A; closest "
            "remaining case to the all-candidate anomaly-parent median; maximum "
            "remaining case; ties by fold order, parent index, then filename"
        ),
        labels=labels,
        folds=np.asarray([row["fold"] for row in selected]),
        indices=np.asarray([row["index"] for row in selected], dtype=np.int64),
        filenames=np.asarray([row["filename"] for row in selected]),
        hann_parent_pixel_ap=np.asarray([row["hann_ap"] for row in selected]),
        pcdr_parent_pixel_ap=np.asarray([row["pcdr_ap"] for row in selected]),
        delta_parent_pixel_ap=np.asarray([row["delta"] for row in selected]),
        images=images,
        masks=np.stack([row["mask"] for row in selected]).astype(np.uint8),
        hann_fields=np.stack([row["hann_field"] for row in selected]).astype(np.float32),
        pcdr_fields=np.stack([row["pcdr_field"] for row in selected]).astype(np.float32),
    )
    audit = {
        "schema_version": 1,
        "selection_rule": (
            "Select the most negative per-parent PCDR-minus-Hann Pixel-AP case "
            "within the retained negative roll Rollo2A, then the closest remaining "
            "case to the median across all anomalous parents in the six candidate-"
            "evaluation folds, then the best remaining case. Ties are resolved by "
            "candidate-fold order, parent index, and filename."
        ),
        "eligible_parents": len(rows),
        "median_delta": median_value,
        "selected": [
            {
                key: row[key]
                for key in ("fold", "index", "filename", "hann_ap", "pcdr_ap", "delta")
            }
            for row in selected
        ],
        "input_cache_sha256": input_hashes,
        "output_npz_sha256": sha256_file(args.output_npz),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
