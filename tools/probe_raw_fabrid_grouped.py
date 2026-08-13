"""Frozen full-frame RAW-FABRID same-roll versus leave-one-roll-out K0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.models import convnext_tiny


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_rows(root: Path, config: dict[str, object]) -> list[dict[str, object]]:
    with (root / str(config["metadata"])).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["has_defect"] = str(row["has_defect"]).lower() == "true"
        row["total_area_px"] = float(row["total_area_px"])
        row["num_defects"] = int(row["num_defects"])
    return rows


def load_backbone(config: dict[str, object], device: torch.device) -> torch.nn.Module:
    model = convnext_tiny(weights=None)
    weights = torch.load(Path(str(config["weights"])), map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    backbone = model.features[: int(config["feature_stop_index"])].eval().to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone


def cache_identity(rows: list[dict[str, object]], config: dict[str, object]) -> str:
    payload = {
        "filenames": [row["filename"] for row in rows],
        "backbone": config,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@torch.inference_mode()
def prepare_feature_cache(
    rows: list[dict[str, object]],
    data_root: Path,
    project_root: Path,
    config: dict[str, object],
    device: torch.device,
    smoke: bool,
) -> tuple[np.ndarray, list[int]]:
    selected = list(range(len(rows)))
    if smoke:
        by_roll: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            by_roll[str(row["roll_id"])].append(index)
        selected = []
        for indices in by_roll.values():
            normals = [index for index in indices if not rows[index]["has_defect"]]
            defects = [index for index in indices if rows[index]["has_defect"]]
            selected.extend(normals[:6] + defects[:2])
        selected = sorted(selected)

    cache_path = resolve_path(project_root, str(config["cache"]))
    metadata_path = cache_path.with_suffix(".json")
    identity = cache_identity(rows, config)
    if not smoke and cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") == identity:
            return np.load(cache_path, mmap_mode="r"), selected

    backbone = load_backbone(config, device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    batch_size = 1 if smoke else int(config["batch_size"])
    destination: np.ndarray | None = None
    smoke_features: list[np.ndarray] = []
    for start in range(0, len(selected), batch_size):
        current = selected[start : start + batch_size]
        images: list[np.ndarray] = []
        for index in current:
            image_path = data_root / "images" / str(rows[index]["filename"])
            gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
            images.append(np.repeat(gray[..., None], 3, axis=2))
        batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
        batch = (batch.div_(255.0) - mean) / std
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
            features = backbone(batch)
        features = F.normalize(features.float(), dim=1).cpu().numpy().astype(np.float16)
        if smoke:
            smoke_features.extend(features)
            continue
        if destination is None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            destination = np.lib.format.open_memmap(
                cache_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(rows), *features.shape[1:]),
            )
        destination[current] = features
        if start == 0 or (start // batch_size + 1) % 50 == 0:
            print(f"feature-cache {min(start + batch_size, len(selected))}/{len(selected)}", flush=True)
    if smoke:
        return np.stack(smoke_features), selected
    assert destination is not None
    destination.flush()
    metadata_path.write_text(
        json.dumps(
            {
                "identity": identity,
                "shape": list(destination.shape),
                "dtype": str(destination.dtype),
                "filenames": [row["filename"] for row in rows],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    del destination
    return np.load(cache_path, mmap_mode="r"), selected


def split_p1(
    indices: list[int],
    rows: list[dict[str, object]],
    seed: int,
    train_fraction: float,
    calibration_fraction: float,
) -> dict[str, list[int]]:
    normals = sorted(index for index in indices if not rows[index]["has_defect"])
    defects = sorted(index for index in indices if rows[index]["has_defect"])
    generator = np.random.default_rng(seed)
    shuffled = np.asarray(normals, dtype=np.int64)
    generator.shuffle(shuffled)
    train_count = max(1, int(math.floor(len(normals) * train_fraction)))
    calibration_count = max(1, int(math.floor(len(normals) * calibration_fraction)))
    if train_count + calibration_count >= len(normals):
        calibration_count = max(1, len(normals) - train_count - 1)
    return {
        "training_normals": sorted(shuffled[:train_count].tolist()),
        "calibration_normals": sorted(shuffled[train_count : train_count + calibration_count].tolist()),
        "evaluation_normals": sorted(shuffled[train_count + calibration_count :].tolist()),
        "anomalies": defects,
    }


def split_p2(target_roll: str, by_roll: dict[str, list[int]], rows: list[dict[str, object]]) -> dict[str, list[int]]:
    training: list[int] = []
    calibration: list[int] = []
    for roll, indices in sorted(by_roll.items()):
        if roll == target_roll:
            continue
        normals = sorted(index for index in indices if not rows[index]["has_defect"])
        calibration.extend(normals[::4])
        calibration_set = set(normals[::4])
        training.extend(index for index in normals if index not in calibration_set)
    target = by_roll[target_roll]
    return {
        "training_normals": sorted(training),
        "calibration_normals": sorted(calibration),
        "evaluation_normals": sorted(index for index in target if not rows[index]["has_defect"]),
        "anomalies": sorted(index for index in target if rows[index]["has_defect"]),
    }


def sample_bank(
    features: np.ndarray,
    training_indices: list[int],
    tokens_per_parent: int,
    maximum_tokens: int,
    seed: int,
    device: torch.device,
    cache_indices: dict[int, int],
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples: list[torch.Tensor] = []
    for index in training_indices:
        feature = torch.from_numpy(np.asarray(features[cache_indices[index]], dtype=np.float32))
        tokens = feature.permute(1, 2, 0).reshape(-1, feature.shape[0])
        selection = torch.randperm(tokens.shape[0], generator=generator)[:tokens_per_parent]
        samples.append(tokens[selection])
    bank = torch.cat(samples, dim=0)
    if bank.shape[0] > maximum_tokens:
        selection = torch.randperm(bank.shape[0], generator=generator)[:maximum_tokens]
        bank = bank[selection]
    return F.normalize(bank, dim=1).to(device)


@torch.inference_mode()
def score_map(
    feature_array: np.ndarray,
    bank: torch.Tensor,
    query_chunk: int,
    device: torch.device,
    local_kernel: int,
) -> dict[str, np.ndarray]:
    feature = torch.from_numpy(np.asarray(feature_array, dtype=np.float32))
    channels, height, width = feature.shape
    queries = F.normalize(feature.permute(1, 2, 0).reshape(-1, channels), dim=1).to(device)
    best = torch.full((queries.shape[0],), -1.0, device=device)
    for start in range(0, queries.shape[0], query_chunk):
        current = queries[start : start + query_chunk]
        best[start : start + query_chunk] = (current @ bank.T).max(dim=1).values
    base = (1.0 - best).reshape(height, width)
    source = base[None, None]
    local = F.avg_pool2d(source, local_kernel, stride=1, padding=local_kernel // 2)[0, 0]
    center = torch.median(base)
    mad = torch.median(torch.abs(base - center))
    robust_z = (base - center) / torch.clamp(1.4826 * mad, min=1e-6)
    robust_source = robust_z[None, None]
    robust_local = F.avg_pool2d(
        robust_source, local_kernel, stride=1, padding=local_kernel // 2
    )[0, 0]
    return {
        "base": base.cpu().numpy(),
        "local_mean_max": torch.maximum(base, local).cpu().numpy(),
        "robust_z": robust_z.cpu().numpy(),
        "robust_z_local_mean_max": torch.maximum(robust_z, robust_local).cpu().numpy(),
    }


def top_fraction_mean(score: np.ndarray, fraction: float) -> float:
    flat = score.reshape(-1)
    count = max(1, int(math.ceil(flat.size * fraction)))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def quantile_higher(values: list[float], fpr: float) -> float:
    return float(np.quantile(np.asarray(values), 1.0 - fpr, method="higher"))


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def pool_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    source_height, source_width = mask.shape
    if source_height % height or source_width % width:
        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(resized) > 0
    stride_y, stride_x = source_height // height, source_width // width
    return mask.reshape(height, stride_y, width, stride_x).max(axis=(1, 3)).astype(bool)


def annotation_masks(
    coco: dict[str, object],
    rows: list[dict[str, object]],
    feature_height: int,
    feature_width: int,
) -> tuple[dict[int, list[dict[str, object]]], dict[int, np.ndarray]]:
    filename_to_index = {str(row["filename"]): index for index, row in enumerate(rows)}
    image_id_to_index = {
        int(image["id"]): filename_to_index[str(image["file_name"])]
        for image in coco["images"]
    }
    records: dict[int, list[dict[str, object]]] = defaultdict(list)
    unions: dict[int, np.ndarray] = {}
    for annotation in coco["annotations"]:
        image_index = image_id_to_index[int(annotation["image_id"])]
        image = Image.new("1", (1792, 1024), 0)
        draw = ImageDraw.Draw(image)
        segmentation = annotation.get("segmentation", [])
        if not isinstance(segmentation, list):
            raise TypeError("RAW-FABRID K0 expects polygon COCO segmentations")
        for polygon in segmentation:
            points = list(zip(polygon[0::2], polygon[1::2]))
            if len(points) >= 3:
                draw.polygon(points, fill=1)
        grid_mask = pool_mask(np.asarray(image, dtype=np.uint8), feature_height, feature_width)
        x, y, box_width, box_height = [float(value) for value in annotation["bbox"]]
        aspect = max(box_width / max(box_height, 1e-8), box_height / max(box_width, 1e-8))
        records[image_index].append(
            {
                "annotation_id": int(annotation["id"]),
                "area_px": float(annotation["area"]),
                "aspect_ratio": float(aspect),
                "bbox_width_px": box_width,
                "bbox_height_px": box_height,
                "minor_axis_px": min(box_width, box_height),
                "major_axis_px": max(box_width, box_height),
                "mask": grid_mask,
            }
        )
        unions[image_index] = np.logical_or(unions.get(image_index, False), grid_mask)
    return records, unions


def evaluate_protocol(
    name: str,
    splits: dict[str, dict[str, list[int]]],
    rows: list[dict[str, object]],
    features: np.ndarray,
    cache_indices: dict[int, int],
    annotations: dict[int, list[dict[str, object]]],
    union_masks: dict[int, np.ndarray],
    config: dict[str, object],
    device: torch.device,
    seed: int,
    matched_normals: dict[str, list[int]],
) -> dict[str, object]:
    modes = list(config["scores"]["modes"])
    top_fraction = float(config["scores"]["image_top_fraction"])
    target_fpr = float(config["evaluation"]["normal_parent_fpr"])
    all_image_labels: dict[str, list[int]] = {mode: [] for mode in modes}
    all_image_scores: dict[str, list[float]] = {mode: [] for mode in modes}
    all_pixel_labels: list[np.ndarray] = []
    all_pixel_scores: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    instance_records: dict[str, list[dict[str, object]]] = {mode: [] for mode in modes}
    folds: dict[str, object] = {}
    for fold_number, (roll, split) in enumerate(sorted(splits.items()), start=1):
        bank = sample_bank(
            features,
            split["training_normals"],
            int(config["memory_bank"]["tokens_per_training_parent"]),
            int(config["memory_bank"]["maximum_tokens"]),
            seed + 1000 * fold_number + (0 if name.startswith("P1") else 100),
            device,
            cache_indices,
        )
        scored_indices = sorted(set(
            split["calibration_normals"]
            + split["evaluation_normals"]
            + split["anomalies"]
            + matched_normals[roll]
        ))
        maps: dict[str, dict[int, np.ndarray]] = {mode: {} for mode in modes}
        image_scores: dict[str, dict[int, float]] = {mode: {} for mode in modes}
        maximum_scores: dict[str, dict[int, float]] = {mode: {} for mode in modes}
        for offset, index in enumerate(scored_indices, start=1):
            current = score_map(
                features[cache_indices[index]],
                bank,
                int(config["memory_bank"]["query_chunk"]),
                device,
                int(config["scores"]["local_mean_kernel_tokens"]),
            )
            for mode in modes:
                maps[mode][index] = current[mode]
                image_scores[mode][index] = top_fraction_mean(current[mode], top_fraction)
                maximum_scores[mode][index] = float(current[mode].max())
        fold_result: dict[str, object] = {
            "counts": {key: len(value) for key, value in split.items()},
            "modes": {},
        }
        evaluation = split["evaluation_normals"] + split["anomalies"]
        labels = np.asarray(
            [0] * len(split["evaluation_normals"]) + [1] * len(split["anomalies"]), dtype=np.uint8
        )
        matched_evaluation = matched_normals[roll] + split["anomalies"]
        matched_labels = np.asarray(
            [0] * len(matched_normals[roll]) + [1] * len(split["anomalies"]), dtype=np.uint8
        )
        for mode in modes:
            image_threshold = quantile_higher(
                [image_scores[mode][index] for index in split["calibration_normals"]], target_fpr
            )
            pixel_threshold = quantile_higher(
                [maximum_scores[mode][index] for index in split["calibration_normals"]], target_fpr
            )
            scores = np.asarray([image_scores[mode][index] for index in evaluation], dtype=np.float64)
            matched_scores = np.asarray(
                [image_scores[mode][index] for index in matched_evaluation], dtype=np.float64
            )
            normal_fpr = float(np.mean([
                image_scores[mode][index] > image_threshold for index in split["evaluation_normals"]
            ]))
            parent_recall = float(np.mean([
                image_scores[mode][index] > image_threshold for index in split["anomalies"]
            ]))
            oracle_image_threshold = quantile_higher(
                [image_scores[mode][index] for index in split["evaluation_normals"]], target_fpr
            )
            oracle_parent_recall = float(np.mean([
                image_scores[mode][index] > oracle_image_threshold for index in split["anomalies"]
            ]))
            oracle_pixel_threshold = quantile_higher(
                [maximum_scores[mode][index] for index in split["evaluation_normals"]], target_fpr
            )
            fold_result["modes"][mode] = {
                **binary_metrics(labels, scores),
                "matched_auroc": float(roc_auc_score(matched_labels, matched_scores)),
                "matched_average_precision": float(average_precision_score(matched_labels, matched_scores)),
                "normal_parent_fpr": normal_fpr,
                "defect_parent_recall_at_calibrated_1pct_fpr": parent_recall,
                "image_threshold": image_threshold,
                "pixel_threshold": pixel_threshold,
                "diagnostic_oracle_target_image_threshold": oracle_image_threshold,
                "diagnostic_oracle_target_parent_recall_at_1pct_fpr": oracle_parent_recall,
                "diagnostic_oracle_target_pixel_threshold": oracle_pixel_threshold,
            }
            all_image_labels[mode].extend(labels.tolist())
            all_image_scores[mode].extend(scores.tolist())
            for index in split["anomalies"]:
                for annotation in annotations.get(index, []):
                    inside = maps[mode][index][annotation["mask"]]
                    instance_records[mode].append({
                        "roll": roll,
                        "image_index": index,
                        "annotation_id": annotation["annotation_id"],
                        "area_px": annotation["area_px"],
                        "aspect_ratio": annotation["aspect_ratio"],
                        "detected": bool(inside.size and float(inside.max()) > pixel_threshold),
                    })
        height, width = next(iter(maps[modes[0]].values())).shape
        for index in evaluation:
            label = union_masks.get(index, np.zeros((height, width), dtype=bool))
            all_pixel_labels.append(label.reshape(-1))
            for mode in modes:
                all_pixel_scores[mode].append(maps[mode][index].reshape(-1).astype(np.float32))
        folds[roll] = fold_result
        print(f"{name} {roll} scored={len(scored_indices)}", flush=True)

    pooled: dict[str, object] = {}
    pixel_labels = np.concatenate(all_pixel_labels).astype(np.uint8)
    small_area = float(config["evaluation"]["small_instance_area_px"])
    elongated_aspect = float(config["evaluation"]["elongated_instance_aspect_ratio"])
    for mode in modes:
        records = instance_records[mode]
        buckets = {
            "all": records,
            "small": [item for item in records if float(item["area_px"]) <= small_area],
            "not_small": [item for item in records if float(item["area_px"]) > small_area],
            "elongated": [item for item in records if float(item["aspect_ratio"]) >= elongated_aspect],
            "not_elongated": [item for item in records if float(item["aspect_ratio"]) < elongated_aspect],
        }
        pooled[mode] = {
            "image": binary_metrics(
                np.asarray(all_image_labels[mode], dtype=np.uint8),
                np.asarray(all_image_scores[mode], dtype=np.float64),
            ),
            "pixel": binary_metrics(pixel_labels, np.concatenate(all_pixel_scores[mode])),
            "instance_recall": {
                bucket: {
                    "instances": len(items),
                    "recall": float(np.mean([item["detected"] for item in items])) if items else float("nan"),
                }
                for bucket, items in buckets.items()
            },
        }
    return {"folds": folds, "pooled": pooled, "instance_records": instance_records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    project_root = args.config.resolve().parent.parent
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = Path(str(config["data"]["root"]))
    rows = load_rows(data_root, config["data"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, selected = prepare_feature_cache(
        rows, data_root, project_root, config["backbone"], device, args.smoke
    )
    cache_indices = {original: cache for cache, original in enumerate(selected)} if args.smoke else {
        index: index for index in selected
    }
    by_roll: dict[str, list[int]] = defaultdict(list)
    for index in selected:
        by_roll[str(rows[index]["roll_id"])].append(index)
    p1_config = config["protocols"]["P1_same_roll_full_frame"]
    p1_splits = {
        roll: split_p1(
            indices,
            rows,
            int(config["seed"]) + fold,
            float(p1_config["normal_train_fraction"]),
            float(p1_config["normal_calibration_fraction"]),
        )
        for fold, (roll, indices) in enumerate(sorted(by_roll.items()), start=1)
    }
    if args.smoke:
        for split in p1_splits.values():
            split["training_normals"] = split["training_normals"][:3]
            split["calibration_normals"] = split["calibration_normals"][:1]
            split["evaluation_normals"] = split["evaluation_normals"][:2]
            split["anomalies"] = split["anomalies"][:1]
    p2_splits = {roll: split_p2(roll, by_roll, rows) for roll in sorted(by_roll)}
    if args.smoke:
        for split in p2_splits.values():
            split["training_normals"] = split["training_normals"][:8]
            split["calibration_normals"] = split["calibration_normals"][:4]
            split["evaluation_normals"] = split["evaluation_normals"][:2]
            split["anomalies"] = split["anomalies"][:1]

    coco = json.loads((data_root / str(config["data"]["coco"])).read_text(encoding="utf-8"))
    feature_height, feature_width = int(features.shape[-2]), int(features.shape[-1])
    annotations, unions = annotation_masks(coco, rows, feature_height, feature_width)
    matched_normals = {roll: split["evaluation_normals"] for roll, split in p1_splits.items()}
    p1 = evaluate_protocol(
        "P1_same_roll_full_frame", p1_splits, rows, features, cache_indices, annotations, unions,
        config, device, int(config["seed"]), matched_normals,
    )
    p2 = evaluate_protocol(
        "P2_leave_one_roll_out_full_frame", p2_splits, rows, features, cache_indices, annotations, unions,
        config, device, int(config["seed"]), matched_normals,
    )
    modes = list(config["scores"]["modes"])
    comparisons: dict[str, object] = {}
    for mode in modes:
        per_roll: dict[str, object] = {}
        drops: list[float] = []
        for roll in sorted(by_roll):
            p1_fold = p1["folds"][roll]["modes"][mode]
            p2_fold = p2["folds"][roll]["modes"][mode]
            drop = float(
                p1_fold["defect_parent_recall_at_calibrated_1pct_fpr"]
                - p2_fold["defect_parent_recall_at_calibrated_1pct_fpr"]
            )
            drops.append(drop)
            per_roll[roll] = {
                "matched_image_auroc_change_P2_minus_P1": float(
                    p2_fold["matched_auroc"] - p1_fold["matched_auroc"]
                ),
                "parent_recall_drop_P1_minus_P2": drop,
                "normal_fpr_change_P2_minus_P1": float(
                    p2_fold["normal_parent_fpr"] - p1_fold["normal_parent_fpr"]
                ),
            }
        threshold = float(config["stop_rules"]["substantive_matched_parent_recall_drop"])
        minimum_rolls = int(config["stop_rules"]["minimum_rolls_same_direction"])
        failing_rolls = sum(drop >= threshold for drop in drops)
        comparisons[mode] = {
            "per_roll": per_roll,
            "rolls_with_substantive_parent_recall_drop": failing_rolls,
            "median_parent_recall_drop": float(np.median(drops)),
            "problem_gate_passed": bool(failing_rolls >= minimum_rolls),
        }
    strongest_mode = max(
        modes,
        key=lambda mode: float(p2["pooled"][mode]["image"]["average_precision"]),
    )
    verdict = (
        "PROBLEM_SURVIVES_FIRST_BASELINE_ONLY"
        if comparisons[strongest_mode]["problem_gate_passed"]
        else "PROBLEM_GATE_NOT_MET_ON_FIRST_BASELINE"
    )
    result = {
        "schema_version": 1,
        "config": config,
        "runtime": {
            "torch": torch.__version__,
            "device": str(device),
            "feature_shape": [int(value) for value in features.shape[1:]],
            "smoke": args.smoke,
        },
        "dataset": {
            "selected_parents": len(selected),
            "rolls": {roll: len(indices) for roll, indices in sorted(by_roll.items())},
        },
        "protocols": {
            "P1_same_roll_full_frame": p1,
            "P2_leave_one_roll_out_full_frame": p2,
        },
        "comparison": comparisons,
        "decision": {
            "verdict": verdict,
            "strongest_P2_mode_by_pooled_image_AP": strongest_mode,
            "scope": config["stop_rules"]["interpretation"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
    print(json.dumps(comparisons[strongest_mode], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
