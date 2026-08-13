"""Frozen Anomalib PatchCore gate for full-width RAW-FABRID roll holdout.

The 1792x1024 parent is scanned as a fixed, non-overlapping 7x4 grid. Tile
selection never uses defect labels. Training uses a fixed tile budget because
the official PatchCore implementation retains every training embedding on the
accelerator before coreset construction; evaluation always scans every tile.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from anomalib.data.dataclasses import ImageBatch, ImageItem
from anomalib.engine import Engine
from anomalib.models import Patchcore
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from probe_raw_fabrid_grouped import (
    annotation_masks,
    binary_metrics,
    load_rows,
    quantile_higher,
    split_p1,
    split_p2,
    top_fraction_mean,
)


@dataclass(frozen=True)
class TileRef:
    parent_index: int
    row: int
    column: int


class RawFabricTileDataset(Dataset[ImageItem]):
    """Read fixed RAW-FABRID tiles without materializing derivative images."""

    def __init__(
        self,
        data_root: Path,
        rows: list[dict[str, object]],
        refs: list[TileRef],
        tile_height: int,
        tile_width: int,
    ) -> None:
        self.data_root = data_root
        self.rows = rows
        self.refs = refs
        self.tile_height = tile_height
        self.tile_width = tile_width
        self._cached_index: int | None = None
        self._cached_image: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.refs)

    def _parent(self, index: int) -> np.ndarray:
        if self._cached_index != index:
            path = self.data_root / "images" / str(self.rows[index]["filename"])
            self._cached_image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            self._cached_index = index
        assert self._cached_image is not None
        return self._cached_image

    def __getitem__(self, position: int) -> ImageItem:
        ref = self.refs[position]
        parent = self._parent(ref.parent_index)
        y0 = ref.row * self.tile_height
        x0 = ref.column * self.tile_width
        tile = np.ascontiguousarray(parent[y0 : y0 + self.tile_height, x0 : x0 + self.tile_width])
        image = torch.from_numpy(tile).float().div_(255.0).unsqueeze(0).repeat(3, 1, 1)
        path = self.data_root / "images" / str(self.rows[ref.parent_index]["filename"])
        return ImageItem(image=image, gt_label=torch.tensor(0, dtype=torch.long), image_path=path)


def all_tile_refs(indices: list[int], tile_rows: int, tile_columns: int) -> list[TileRef]:
    return [
        TileRef(index, row, column)
        for index in sorted(indices)
        for row in range(tile_rows)
        for column in range(tile_columns)
    ]


def select_training_refs(
    indices: list[int],
    tile_rows: int,
    tile_columns: int,
    budget: int,
    seed: int,
) -> list[TileRef]:
    """Cover parents first, then fill the remaining fixed budget uniformly."""

    generator = np.random.default_rng(seed)
    parents = np.asarray(sorted(indices), dtype=np.int64)
    generator.shuffle(parents)
    selected: list[TileRef] = []
    selected_set: set[TileRef] = set()
    for index in parents[:budget]:
        position = int(generator.integers(tile_rows * tile_columns))
        ref = TileRef(int(index), position // tile_columns, position % tile_columns)
        selected.append(ref)
        selected_set.add(ref)
    if len(selected) < budget:
        remainder = [
            ref for ref in all_tile_refs(indices, tile_rows, tile_columns) if ref not in selected_set
        ]
        generator.shuffle(remainder)
        selected.extend(remainder[: budget - len(selected)])
    return selected


def make_loader(
    data_root: Path,
    rows: list[dict[str, object]],
    refs: list[TileRef],
    tile_height: int,
    tile_width: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = RawFabricTileDataset(data_root, rows, refs, tile_height, tile_width)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
        collate_fn=ImageBatch.collate,
    )


def stitch_predictions(
    predictions: list[ImageBatch],
    refs: list[TileRef],
    tile_rows: int,
    tile_columns: int,
    tile_height: int,
    tile_width: int,
    evaluation_stride: int,
) -> dict[int, np.ndarray]:
    tile_grid_height = tile_height // evaluation_stride
    tile_grid_width = tile_width // evaluation_stride
    flattened: list[np.ndarray] = []
    for batch in predictions:
        maps = batch.anomaly_map
        if maps is None:
            raise RuntimeError("Anomalib prediction did not return anomaly_map")
        if maps.ndim == 3:
            maps = maps.unsqueeze(1)
        if maps.ndim != 4:
            raise RuntimeError(f"Unexpected anomaly_map shape: {tuple(maps.shape)}")
        maps = F.interpolate(
            maps.detach().float(),
            size=(tile_grid_height, tile_grid_width),
            mode="bilinear",
            align_corners=False,
        )
        flattened.extend(maps[:, 0].cpu().numpy())
    if len(flattened) != len(refs):
        raise RuntimeError(f"Prediction count mismatch: {len(flattened)} != {len(refs)}")
    parent_maps: dict[int, np.ndarray] = {}
    for ref, tile_map in zip(refs, flattened, strict=True):
        if ref.parent_index not in parent_maps:
            parent_maps[ref.parent_index] = np.empty(
                (tile_rows * tile_grid_height, tile_columns * tile_grid_width), dtype=np.float32
            )
        y0 = ref.row * tile_grid_height
        x0 = ref.column * tile_grid_width
        parent_maps[ref.parent_index][y0 : y0 + tile_grid_height, x0 : x0 + tile_grid_width] = tile_map
    return parent_maps


def score_modes(raw: np.ndarray, kernel: int) -> dict[str, np.ndarray]:
    source = torch.from_numpy(raw).float()
    center = torch.median(source)
    mad = torch.median(torch.abs(source - center))
    robust = (source - center) / torch.clamp(1.4826 * mad, min=1e-6)
    local = F.avg_pool2d(
        robust[None, None], kernel_size=kernel, stride=1, padding=kernel // 2
    )[0, 0]
    return {
        "raw": raw,
        "robust_z": robust.numpy(),
        "robust_z_local_mean_max": torch.maximum(robust, local).numpy(),
    }


def evaluate(
    maps: dict[int, dict[str, np.ndarray]],
    split: dict[str, list[int]],
    annotations: dict[int, list[dict[str, object]]],
    union_masks: dict[int, np.ndarray],
    config: dict[str, object],
    position_counterfactual: bool = False,
) -> dict[str, object]:
    modes = list(config["scores"]["modes"])
    fraction = float(config["scores"]["image_top_fraction"])
    target_fpr = float(config["evaluation"]["normal_parent_fpr"])
    evaluation = split["evaluation_normals"] + split["anomalies"]
    labels = np.asarray(
        [0] * len(split["evaluation_normals"]) + [1] * len(split["anomalies"]), dtype=np.uint8
    )
    height, width = next(iter(next(iter(maps.values())).values())).shape
    result: dict[str, object] = {}
    for mode in modes:
        image_scores = {
            index: top_fraction_mean(current[mode], fraction) for index, current in maps.items()
        }
        max_scores = {index: float(current[mode].max()) for index, current in maps.items()}
        image_threshold = quantile_higher(
            [image_scores[index] for index in split["calibration_normals"]], target_fpr
        )
        pixel_threshold = quantile_higher(
            [max_scores[index] for index in split["calibration_normals"]], target_fpr
        )
        oracle_pixel_threshold = quantile_higher(
            [max_scores[index] for index in split["evaluation_normals"]], target_fpr
        )
        oracle_image_threshold = quantile_higher(
            [image_scores[index] for index in split["evaluation_normals"]], target_fpr
        )
        scores = np.asarray([image_scores[index] for index in evaluation], dtype=np.float64)
        pixel_labels: list[np.ndarray] = []
        pixel_scores: list[np.ndarray] = []
        records: list[dict[str, object]] = []
        for index in evaluation:
            pixel_labels.append(union_masks.get(index, np.zeros((height, width), dtype=bool)).reshape(-1))
            pixel_scores.append(maps[index][mode].reshape(-1))
        for index in split["anomalies"]:
            for annotation in annotations.get(index, []):
                inside = maps[index][mode][annotation["mask"]]
                instance_maximum = float(inside.max()) if inside.size else float("-inf")
                record: dict[str, object] = {
                    "image_index": index,
                    "annotation_id": annotation["annotation_id"],
                    "area_px": annotation["area_px"],
                    "aspect_ratio": annotation["aspect_ratio"],
                    "bbox_width_px": annotation["bbox_width_px"],
                    "bbox_height_px": annotation["bbox_height_px"],
                    "minor_axis_px": annotation["minor_axis_px"],
                    "major_axis_px": annotation["major_axis_px"],
                    "maximum_score": instance_maximum,
                    "detected": bool(instance_maximum > pixel_threshold),
                    "detected_target_oracle": bool(instance_maximum > oracle_pixel_threshold),
                    "source_calibrated_margin": instance_maximum - pixel_threshold,
                    "target_oracle_margin": instance_maximum - oracle_pixel_threshold,
                    "vs_target_normal_maximum_pairwise_auc": float(np.mean([
                        instance_maximum > max_scores[normal_index]
                        for normal_index in split["evaluation_normals"]
                    ])),
                }
                if position_counterfactual:
                    target_position_maxima = [
                        (
                            float(maps[normal_index][mode][annotation["mask"]].max())
                            if maps[normal_index][mode][annotation["mask"]].size
                            else float("-inf")
                        )
                        for normal_index in split["evaluation_normals"]
                    ]
                    source_position_maxima = [
                        (
                            float(maps[normal_index][mode][annotation["mask"]].max())
                            if maps[normal_index][mode][annotation["mask"]].size
                            else float("-inf")
                        )
                        for normal_index in split["calibration_normals"]
                    ]
                    target_position_auc = float(np.mean(
                        [instance_maximum > value for value in target_position_maxima]
                    ))
                    source_position_auc = float(np.mean(
                        [instance_maximum > value for value in source_position_maxima]
                    ))
                    record.update({
                        "vs_target_normal_same_position_pairwise_auc": target_position_auc,
                        "vs_source_normal_same_position_pairwise_auc": source_position_auc,
                        "target_same_position_minus_full_parent_auc": (
                            target_position_auc
                            - float(record["vs_target_normal_maximum_pairwise_auc"])
                        ),
                        "source_minus_target_same_position_auc": (
                            source_position_auc - target_position_auc
                        ),
                    })
                records.append(record)
        small_area = float(config["evaluation"]["small_instance_area_px"])
        elongated_aspect = float(config["evaluation"]["elongated_instance_aspect_ratio"])
        buckets = {
            "all": records,
            "small": [item for item in records if float(item["area_px"]) <= small_area],
            "not_small": [item for item in records if float(item["area_px"]) > small_area],
            "elongated": [item for item in records if float(item["aspect_ratio"]) >= elongated_aspect],
            "not_elongated": [item for item in records if float(item["aspect_ratio"]) < elongated_aspect],
        }
        instance_recall = {
            name: {
                "instances": len(items),
                "recall": float(np.mean([item["detected"] for item in items]))
                if items else float("nan"),
                "target_oracle_recall": float(np.mean([
                    item["detected_target_oracle"] for item in items
                ])) if items else float("nan"),
                "vs_target_normal_maximum_pairwise_auc": float(np.mean([
                    item["vs_target_normal_maximum_pairwise_auc"] for item in items
                ])) if items else float("nan"),
                "maximum_score_median": float(np.median([
                    item["maximum_score"] for item in items
                ])) if items else float("nan"),
            }
            for name, items in buckets.items()
        }
        if position_counterfactual:
            for name, items in buckets.items():
                for key in (
                    "vs_target_normal_same_position_pairwise_auc",
                    "vs_source_normal_same_position_pairwise_auc",
                    "target_same_position_minus_full_parent_auc",
                    "source_minus_target_same_position_auc",
                ):
                    instance_recall[name][key] = float(np.mean([
                        item[key] for item in items
                    ])) if items else float("nan")
        result[mode] = {
            **binary_metrics(labels, scores),
            "pixel_average_precision": float(
                average_precision_score(
                    np.concatenate(pixel_labels).astype(np.uint8), np.concatenate(pixel_scores)
                )
            ),
            "source_calibrated": {
                "image_threshold": image_threshold,
                "normal_parent_fpr": float(np.mean([
                    image_scores[index] > image_threshold for index in split["evaluation_normals"]
                ])),
                "defect_parent_recall_at_1pct_fpr": float(np.mean([
                    image_scores[index] > image_threshold for index in split["anomalies"]
                ])),
                "pixel_threshold": pixel_threshold,
            },
            "diagnostic_target_oracle": {
                "image_threshold": oracle_image_threshold,
                "pixel_threshold": oracle_pixel_threshold,
                "defect_parent_recall_at_1pct_fpr": float(np.mean([
                    image_scores[index] > oracle_image_threshold for index in split["anomalies"]
                ])),
            },
            "instance_recall": instance_recall,
            "instance_records": records,
            "parent_scores": {
                str(index): {
                    "image_score": image_scores[index],
                    "maximum_score": max_scores[index],
                }
                for index in sorted(maps)
            },
        }
    return result


def evaluate_spatial_extreme_controls(
    maps: dict[int, dict[str, np.ndarray]],
    split: dict[str, list[int]],
    annotations: dict[int, list[dict[str, object]]],
    union_masks: dict[int, np.ndarray],
    model_config: dict[str, object],
    control_config: dict[str, object],
) -> dict[str, object]:
    """Evaluate source-only fixed-position normalization with an independent threshold split."""

    mode = str(control_config["base_score_mode"])
    calibration = sorted(split["calibration_normals"])
    estimation_normals = calibration[0::2]
    threshold_normals = calibration[1::2]
    minimum = int(control_config["calibration_partition"]["minimum_parents_each"])
    if min(len(estimation_normals), len(threshold_normals)) < minimum:
        raise RuntimeError("Spatial control lacks the frozen minimum calibration subset size")
    estimation_stack = np.stack([maps[index][mode] for index in estimation_normals]).astype(
        np.float32, copy=False
    )
    scored_indices = sorted(
        set(threshold_normals + split["evaluation_normals"] + split["anomalies"])
    )
    evaluation = split["evaluation_normals"] + split["anomalies"]
    labels = np.asarray(
        [0] * len(split["evaluation_normals"]) + [1] * len(split["anomalies"]),
        dtype=np.uint8,
    )
    parent_fraction = float(control_config["parent_score"]["top_fraction"])
    target_fpr = float(control_config["parent_score"]["target_source_threshold_fpr"])
    small_area = float(model_config["evaluation"]["small_instance_area_px"])
    elongated_aspect = float(model_config["evaluation"]["elongated_instance_aspect_ratio"])
    results: dict[str, object] = {}

    for specification in control_config["controls"]:
        name = str(specification["name"])
        kind = str(specification["kind"])
        fit_start = time.perf_counter()
        if kind == "identity":
            center = None
            denominator = None
        elif kind == "median_mad":
            center = np.median(estimation_stack, axis=0).astype(np.float32)
            mad = np.median(np.abs(estimation_stack - center[None]), axis=0).astype(np.float32)
            denominator = np.maximum(
                float(specification["mad_multiplier"]) * mad,
                float(specification["denominator_floor"]),
            ).astype(np.float32)
        elif kind == "quantile_range":
            quantiles = np.quantile(
                estimation_stack,
                [float(specification["lower_quantile"]), float(specification["upper_quantile"])],
                axis=0,
                method=str(specification["quantile_method"]),
            )
            center = quantiles[0].astype(np.float32)
            denominator = np.maximum(
                quantiles[1] - quantiles[0], float(specification["denominator_floor"])
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown spatial control kind: {kind}")
        fit_seconds = time.perf_counter() - fit_start

        transform_start = time.perf_counter()
        controlled: dict[int, np.ndarray] = {}
        for index in scored_indices:
            source = maps[index][mode]
            controlled[index] = (
                source
                if kind == "identity"
                else ((source - center) / denominator).astype(np.float32)
            )
        transform_seconds = time.perf_counter() - transform_start
        image_scores = {
            index: top_fraction_mean(controlled[index], parent_fraction)
            for index in scored_indices
        }
        maximum_scores = {
            index: float(controlled[index].max()) for index in scored_indices
        }
        image_threshold = quantile_higher(
            [image_scores[index] for index in threshold_normals], target_fpr
        )
        pixel_threshold = quantile_higher(
            [maximum_scores[index] for index in threshold_normals], target_fpr
        )
        parent_scores = np.asarray(
            [image_scores[index] for index in evaluation], dtype=np.float64
        )
        pixel_labels: list[np.ndarray] = []
        pixel_scores: list[np.ndarray] = []
        for index in evaluation:
            pixel_labels.append(
                union_masks.get(index, np.zeros_like(controlled[index], dtype=bool)).reshape(-1)
            )
            pixel_scores.append(controlled[index].reshape(-1))

        records: list[dict[str, object]] = []
        for index in split["anomalies"]:
            for annotation in annotations.get(index, []):
                inside = controlled[index][annotation["mask"]]
                instance_maximum = float(inside.max()) if inside.size else float("-inf")
                records.append({
                    "image_index": index,
                    "annotation_id": annotation["annotation_id"],
                    "area_px": annotation["area_px"],
                    "aspect_ratio": annotation["aspect_ratio"],
                    "maximum_score": instance_maximum,
                    "detected": bool(instance_maximum > pixel_threshold),
                    "vs_target_normal_maximum_pairwise_auc": float(np.mean([
                        instance_maximum > maximum_scores[normal_index]
                        for normal_index in split["evaluation_normals"]
                    ])),
                })
        buckets = {
            "all": records,
            "small": [item for item in records if float(item["area_px"]) <= small_area],
            "elongated": [
                item for item in records if float(item["aspect_ratio"]) >= elongated_aspect
            ],
        }
        instance_metrics = {
            bucket: {
                "instances": len(items),
                "recall": float(np.mean([item["detected"] for item in items]))
                if items else float("nan"),
                "vs_target_normal_maximum_pairwise_auc": float(np.mean([
                    item["vs_target_normal_maximum_pairwise_auc"] for item in items
                ])) if items else float("nan"),
            }
            for bucket, items in buckets.items()
        }
        target_normal_image_fp = int(sum(
            image_scores[index] > image_threshold for index in split["evaluation_normals"]
        ))
        target_normal_pixel_fwer_fp = int(sum(
            maximum_scores[index] > pixel_threshold for index in split["evaluation_normals"]
        ))
        results[name] = {
            "kind": kind,
            "calibration": {
                "estimation_normal_indices": estimation_normals,
                "threshold_normal_indices": threshold_normals,
                "disjoint": not bool(set(estimation_normals).intersection(threshold_normals)),
                "image_threshold": image_threshold,
                "pixel_FWER_threshold": pixel_threshold,
            },
            "runtime": {
                "fit_seconds": fit_seconds,
                "transform_seconds": transform_seconds,
                "transform_microseconds_per_parent": (
                    1e6 * transform_seconds / len(scored_indices)
                ),
                "stored_statistic_bytes": int(
                    0 if center is None else center.nbytes + denominator.nbytes
                ),
            },
            "parent": {
                **binary_metrics(labels, parent_scores),
                "target_normal_false_positives": target_normal_image_fp,
                "defect_recall_at_source_threshold": float(np.mean([
                    image_scores[index] > image_threshold for index in split["anomalies"]
                ])),
            },
            "pixel": {
                "average_precision": float(average_precision_score(
                    np.concatenate(pixel_labels).astype(np.uint8),
                    np.concatenate(pixel_scores),
                )),
                "target_normal_parent_false_positives_at_FWER_threshold": (
                    target_normal_pixel_fwer_fp
                ),
            },
            "instance": instance_metrics,
            "instance_records": records,
        }
    return {
        "config": control_config,
        "base_score_mode": mode,
        "controls": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fold", required=True, help="Held-out roll, for example R1")
    parser.add_argument("--protocol", choices=("P1", "P2"), default="P2")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--random-backbone", action="store_true", help="API smoke only; never a result run")
    parser.add_argument(
        "--position-counterfactual",
        action="store_true",
        help="Record label-assisted same-position normal comparisons for failure diagnosis.",
    )
    parser.add_argument(
        "--spatial-control-config",
        type=Path,
        help="Evaluate frozen source-only spatial extreme controls from this config.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = args.data_root or Path(str(config["data"]["root"]))
    rows = load_rows(data_root, config["data"])
    by_roll: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_roll[str(row["roll_id"])].append(index)
    if args.fold not in by_roll:
        raise ValueError(f"Unknown fold {args.fold}; available: {sorted(by_roll)}")
    if args.protocol == "P1":
        p1_config = config["protocols"]["P1_same_roll_full_frame"]
        fold_number = sorted(by_roll).index(args.fold) + 1
        split = split_p1(
            by_roll[args.fold],
            rows,
            int(config["seed"]) + fold_number,
            float(p1_config["normal_train_fraction"]),
            float(p1_config["normal_calibration_fraction"]),
        )
    else:
        split = split_p2(args.fold, by_roll, rows)
    if args.smoke:
        calibration_smoke_count = 2 if args.spatial_control_config is not None else 1
        split = {
            "training_normals": split["training_normals"][:2],
            "calibration_normals": split["calibration_normals"][:calibration_smoke_count],
            "evaluation_normals": split["evaluation_normals"][:1],
            "anomalies": split["anomalies"][:1],
        }

    tiling = config["tiling"]
    model_config = config["model"]
    tile_rows = int(tiling["rows"])
    tile_columns = int(tiling["columns"])
    tile_height = int(tiling["tile_height"])
    tile_width = int(tiling["tile_width"])
    expected_shape = (int(config["data"]["image_height"]), int(config["data"]["image_width"]))
    if expected_shape != (tile_rows * tile_height, tile_columns * tile_width):
        raise ValueError("Tiling must exactly cover the parent without overlap or padding")

    budget = min(int(tiling["train_tile_budget"]), 8) if args.smoke else int(tiling["train_tile_budget"])
    train_refs = select_training_refs(
        split["training_normals"], tile_rows, tile_columns, budget, int(config["seed"])
    )
    scored_indices = sorted(
        set(split["calibration_normals"] + split["evaluation_normals"] + split["anomalies"])
    )
    predict_refs = all_tile_refs(scored_indices, tile_rows, tile_columns)
    workers = 0 if args.smoke else int(model_config["num_workers"])
    train_loader = make_loader(
        data_root, rows, train_refs, tile_height, tile_width,
        int(model_config["train_batch_size"]), workers,
    )
    predict_loader = make_loader(
        data_root, rows, predict_refs, tile_height, tile_width,
        int(model_config["predict_batch_size"]), workers,
    )

    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats()
    model = Patchcore(
        backbone=str(model_config["backbone"]),
        layers=tuple(model_config["layers"]),
        pre_trained=bool(model_config["pre_trained"]) and not args.random_backbone,
        coreset_sampling_ratio=float(model_config["coreset_sampling_ratio"]),
        num_neighbors=int(model_config["num_neighbors"]),
        precision=str(model_config["precision"]),
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    engine = Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
        default_root_dir=args.output.parent / f"lightning_{args.fold}",
    )
    train_start = time.perf_counter()
    engine.fit(model=model, train_dataloaders=train_loader)
    train_seconds = time.perf_counter() - train_start
    predict_start = time.perf_counter()
    predictions = engine.predict(model=model, dataloaders=predict_loader, return_predictions=True)
    predict_seconds = time.perf_counter() - predict_start
    assert isinstance(predictions, list)
    raw_maps = stitch_predictions(
        predictions,
        predict_refs,
        tile_rows,
        tile_columns,
        tile_height,
        tile_width,
        int(tiling["evaluation_stride"]),
    )
    maps = {
        index: score_modes(raw, int(config["scores"]["local_mean_kernel_tokens"]))
        for index, raw in raw_maps.items()
    }
    coco = json.loads((data_root / str(config["data"]["coco"])).read_text(encoding="utf-8"))
    annotations, union_masks = annotation_masks(
        coco,
        rows,
        int(config["data"]["image_height"]) // int(tiling["evaluation_stride"]),
        int(config["data"]["image_width"]) // int(tiling["evaluation_stride"]),
    )
    metrics = evaluate(
        maps,
        split,
        annotations,
        union_masks,
        config,
        position_counterfactual=args.position_counterfactual,
    )
    spatial_extreme_control = None
    if args.spatial_control_config is not None:
        spatial_control_config = json.loads(
            args.spatial_control_config.read_text(encoding="utf-8")
        )
        if args.smoke:
            spatial_control_config["calibration_partition"]["minimum_parents_each"] = 1
        spatial_extreme_control = evaluate_spatial_extreme_controls(
            maps,
            split,
            annotations,
            union_masks,
            config,
            spatial_control_config,
        )
    strongest = max(
        config["scores"]["modes"], key=lambda mode: float(metrics[mode]["average_precision"])
    )
    result = {
        "schema_version": 1,
        "config": config,
        "run": {
            "fold": args.fold,
            "protocol": args.protocol,
            "smoke": args.smoke,
            "random_backbone": args.random_backbone,
            "position_counterfactual": args.position_counterfactual,
            "numpy_seeded_for_sparse_random_projection": True,
            "data_root": str(data_root),
            "torch": torch.__version__,
            "anomalib": __import__("anomalib").__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds,
            "milliseconds_per_scored_parent": 1000.0 * predict_seconds / len(scored_indices),
            "peak_cuda_gib": (
                torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            ),
        },
        "counts": {
            **{key: len(value) for key, value in split.items()},
            "train_tiles": len(train_refs),
            "scored_parents": len(scored_indices),
            "prediction_tiles": len(predict_refs),
            "training_parents_covered": len({ref.parent_index for ref in train_refs}),
        },
        "split_indices": split,
        "metrics": metrics,
        "decision": {
            "strongest_mode_by_parent_average_precision": strongest,
            "scope": config["interpretation"],
        },
    }
    if spatial_extreme_control is not None:
        result["spatial_extreme_control"] = spatial_extreme_control
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run": result["run"], "counts": result["counts"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
