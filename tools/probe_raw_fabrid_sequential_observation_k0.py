"""Frozen K0 for label-independent sequential crop-context acquisition."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from probe_raw_fabrid_anomalib_patchcore import evaluate, score_modes
from probe_raw_fabrid_grouped import annotation_masks, load_rows, split_p2
from probe_raw_fabrid_physical_field_k0 import (
    compact_metrics,
    json_safe,
    make_crop_refs,
    make_loader,
    score_patchcore,
    sha256_file,
)


VIEWS = ("base_grid", "x_shifted_grid", "y_shifted_grid")


def relative_coordinate_ids(
    shape: tuple[int, int], crop_tokens: int, shift_tokens: int
) -> dict[str, np.ndarray]:
    yy, xx = np.indices(shape)
    offsets = {
        "base_grid": (0, 0),
        "x_shifted_grid": (0, shift_tokens),
        "y_shifted_grid": (shift_tokens, 0),
    }
    return {
        view: (
            ((yy - y_offset) % crop_tokens) * crop_tokens
            + ((xx - x_offset) % crop_tokens)
        ).astype(np.int32)
        for view, (y_offset, x_offset) in offsets.items()
    }


def fit_coordinate_calibration(
    accumulators: dict[int, dict[str, np.ndarray]],
    fit_indices: list[int],
    relative_ids: dict[str, np.ndarray],
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.empty(bins, dtype=np.float32)
    scales = np.empty(bins, dtype=np.float32)
    counts = np.empty(bins, dtype=np.int64)
    for relative_id in range(bins):
        chunks: list[np.ndarray] = []
        for view in VIEWS:
            ids = relative_ids[view]
            for index in fit_indices:
                values = accumulators[index][f"view_{view}"][ids == relative_id]
                finite = values[np.isfinite(values)]
                if finite.size:
                    chunks.append(finite)
        if not chunks:
            raise RuntimeError(f"No calibration values for relative bin {relative_id}")
        samples = np.concatenate(chunks)
        center = float(np.median(samples))
        mad = float(np.median(np.abs(samples - center)))
        centers[relative_id] = center
        scales[relative_id] = max(1.4826 * mad, 1e-6)
        counts[relative_id] = samples.size
    return centers, scales, counts


def fuse_views(
    accumulator: dict[str, np.ndarray],
    views: tuple[str, ...],
    relative_ids: dict[str, np.ndarray],
    centers: np.ndarray,
    scales: np.ndarray,
    optional_mask: np.ndarray | None = None,
) -> np.ndarray:
    shape = accumulator["count"].shape
    weighted_sum = np.zeros(shape, dtype=np.float64)
    weight_sum = np.zeros(shape, dtype=np.float64)
    for view in views:
        values = accumulator[f"view_{view}"]
        valid = np.isfinite(values)
        if view == "x_shifted_grid" and optional_mask is not None:
            valid &= optional_mask
        ids = relative_ids[view]
        weights = 1.0 / np.square(scales[ids[valid]].astype(np.float64))
        weighted_sum[valid] += (
            values[valid].astype(np.float64) - centers[ids[valid]]
        ) * weights
        weight_sum[valid] += weights
    if np.any(weight_sum <= 0):
        raise RuntimeError("Selected observation set does not cover the complete parent")
    return (weighted_sum / weight_sum).astype(np.float32)


def optional_crop_slots(
    height: int, width: int, edge: int, shift: int, stride: int
) -> list[tuple[slice, slice]]:
    slots: list[tuple[slice, slice]] = []
    for y0 in range(0, height - edge + 1, edge):
        for x0 in range(shift, width - edge + 1, edge):
            y = y0 // stride
            x = x0 // stride
            size = edge // stride
            slots.append((slice(y, y + size), slice(x, x + size)))
    return slots


def top_fraction_mean(values: np.ndarray, fraction: float) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    count = max(1, int(math.ceil(flat.size * fraction)))
    threshold = flat.size - count
    return float(np.partition(flat, threshold)[threshold:].mean())


def slot_statistics(
    field: np.ndarray,
    slots: list[tuple[slice, slice]],
    fraction: float,
) -> np.ndarray:
    return np.asarray(
        [top_fraction_mean(field[y_slice, x_slice], fraction) for y_slice, x_slice in slots],
        dtype=np.float64,
    )


def mask_from_selection(
    shape: tuple[int, int],
    slots: list[tuple[slice, slice]],
    selected: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for keep, (y_slice, x_slice) in zip(selected, slots, strict=True):
        if bool(keep):
            mask[y_slice, x_slice] = True
    return mask


def summarize_counts(counts: list[int], optional_slots: int) -> dict[str, Any]:
    values = np.asarray(counts, dtype=np.int64)
    return {
        "parents": int(values.size),
        "optional_crop_count_mean": float(values.mean()) if values.size else None,
        "optional_crop_count_median": float(np.median(values)) if values.size else None,
        "optional_crop_count_minimum": int(values.min()) if values.size else None,
        "optional_crop_count_maximum": int(values.max()) if values.size else None,
        "optional_crop_fraction_mean": (
            float(values.mean() / optional_slots) if values.size else None
        ),
    }


def max_metric_difference(left: dict[str, Any], right: dict[str, Any]) -> float:
    keys = sorted(set(left).intersection(right))
    differences = [
        abs(float(left[key]) - float(right[key]))
        for key in keys
        if isinstance(left[key], (int, float)) and isinstance(right[key], (int, float))
    ]
    return max(differences, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fold")
    parser.add_argument("--full-reference", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    probe = json.loads(args.config.read_text(encoding="utf-8"))
    base_path = Path(str(probe["base_config"]))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    data_root = args.data_root or Path(str(base["data"]["root"]))
    rows = load_rows(data_root, base["data"])
    by_roll: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_roll[str(row["roll_id"])].append(index)
    fold = str(args.fold or probe["fold"])
    split = split_p2(fold, by_roll, rows)
    if args.smoke:
        split = {
            "training_normals": split["training_normals"][:2],
            "calibration_normals": split["calibration_normals"][:3],
            "evaluation_normals": split["evaluation_normals"][:2],
            "anomalies": split["anomalies"][:2],
        }

    geometry = probe["crop_observations"]
    edge = int(geometry["physical_crop_edge"])
    shift = int(geometry["phase_shift_pixels"])
    stride = int(geometry["output_stride_pixels"])
    height = int(base["data"]["image_height"])
    width = int(base["data"]["image_width"])
    parent_shape = (height // stride, width // stride)
    crop_tokens = edge // stride
    shift_tokens = shift // stride
    if any(value <= 0 for value in (edge, shift, stride)):
        raise ValueError("Crop geometry values must be positive")
    if edge % stride or shift % stride or height % stride or width % stride:
        raise ValueError("Crop geometry must align to the output stride")

    scored_indices = sorted(set(
        split["calibration_normals"] + split["evaluation_normals"] + split["anomalies"]
    ))
    refs = make_crop_refs(scored_indices, height, width, edge, shift, list(VIEWS))
    crops_per_parent = len(refs) // len(scored_indices)
    if crops_per_parent != int(geometry["full_crops_per_parent"]):
        raise RuntimeError("Full crop count does not match the frozen configuration")

    model_config = base["model"]
    loader = make_loader(
        data_root,
        rows,
        refs,
        edge,
        "patchcore",
        edge,
        int(model_config.get("predict_batch_size", 8)),
        0 if args.smoke else int(model_config["num_workers"]),
    )
    torch.manual_seed(int(probe["seed"]))
    np.random.seed(int(probe["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(probe["seed"]))
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False

    checkpoint = args.checkpoint or Path(str(probe["checkpoint"]))
    accumulators, predict_seconds, model_facts = score_patchcore(
        base,
        checkpoint,
        loader,
        refs,
        device,
        crop_tokens,
        stride,
        parent_shape,
    )

    calibration_started = time.perf_counter()
    relative_ids = relative_coordinate_ids(parent_shape, crop_tokens, shift_tokens)
    centers, scales, calibration_counts = fit_coordinate_calibration(
        accumulators,
        list(split["calibration_normals"]),
        relative_ids,
        crop_tokens * crop_tokens,
    )
    stage_views = tuple(str(view) for view in geometry["stage_one_views"])
    if stage_views != ("base_grid", "y_shifted_grid"):
        raise ValueError("This frozen probe implements base+y as stage one")
    stage_raw = {
        index: fuse_views(
            accumulators[index], stage_views, relative_ids, centers, scales
        )
        for index in scored_indices
    }
    selector = probe["selector"]
    selector_mode = str(selector["score_mode"])
    kernel = int(base["scores"]["local_mean_kernel_tokens"])
    stage_selector_fields = {
        index: score_modes(raw, kernel)[selector_mode]
        for index, raw in stage_raw.items()
    }
    slots = optional_crop_slots(height, width, edge, shift, stride)
    expected_optional = crops_per_parent - int(geometry["stage_one_crops_per_parent"])
    if len(slots) != expected_optional:
        raise RuntimeError("Optional crop geometry does not match the frozen crop counts")
    fraction = float(selector["candidate_region_top_fraction"])
    calibration_statistics = np.stack([
        slot_statistics(stage_selector_fields[index], slots, fraction)
        for index in split["calibration_normals"]
    ])
    thresholds = np.quantile(
        calibration_statistics,
        float(selector["source_normal_quantile"]),
        axis=0,
        method=str(selector["quantile_method"]),
    )

    selected_by_parent: dict[int, np.ndarray] = {}
    uniform_by_parent: dict[int, np.ndarray] = {}
    statistics_by_parent: dict[int, np.ndarray] = {}
    for index in scored_indices:
        statistics = slot_statistics(stage_selector_fields[index], slots, fraction)
        selected = statistics > thresholds
        count = int(selected.sum())
        generator = np.random.default_rng(int(probe["seed"]) + int(index))
        uniform = np.zeros(len(slots), dtype=bool)
        if count:
            uniform[generator.permutation(len(slots))[:count]] = True
        statistics_by_parent[index] = statistics
        selected_by_parent[index] = selected
        uniform_by_parent[index] = uniform

    full_raw: dict[int, np.ndarray] = {}
    adaptive_raw: dict[int, np.ndarray] = {}
    uniform_raw: dict[int, np.ndarray] = {}
    selected_masks: dict[int, np.ndarray] = {}
    for index in scored_indices:
        selected_mask = mask_from_selection(parent_shape, slots, selected_by_parent[index])
        uniform_mask = mask_from_selection(parent_shape, slots, uniform_by_parent[index])
        selected_masks[index] = selected_mask
        full_raw[index] = fuse_views(
            accumulators[index], VIEWS, relative_ids, centers, scales
        )
        adaptive_raw[index] = fuse_views(
            accumulators[index], VIEWS, relative_ids, centers, scales, selected_mask
        )
        uniform_raw[index] = fuse_views(
            accumulators[index], VIEWS, relative_ids, centers, scales, uniform_mask
        )
    calibration_seconds = time.perf_counter() - calibration_started

    raw_variants = {
        "base_y_pcaf": stage_raw,
        "matched_uniform_sparse_pcaf": uniform_raw,
        "sequential_pcaf": adaptive_raw,
        "full_pcaf": full_raw,
    }
    maps = {
        variant: {index: score_modes(raw, kernel) for index, raw in values.items()}
        for variant, values in raw_variants.items()
    }
    coco = json.loads((data_root / str(base["data"]["coco"])).read_text(encoding="utf-8"))
    annotations, union_masks = annotation_masks(coco, rows, *parent_shape)
    metrics = {
        variant: evaluate(parent_maps, split, annotations, union_masks, base)
        for variant, parent_maps in maps.items()
    }
    mode = str(probe["primary_score_mode"])
    compact = {
        variant: compact_metrics(current, mode, len(split["evaluation_normals"]))
        for variant, current in metrics.items()
    }

    groups = {
        "calibration_normals": list(split["calibration_normals"]),
        "evaluation_normals": list(split["evaluation_normals"]),
        "anomalies": list(split["anomalies"]),
    }
    selection_summary = {
        name: summarize_counts(
            [int(selected_by_parent[index].sum()) for index in indices], len(slots)
        )
        for name, indices in groups.items()
    }
    evaluation_indices = groups["evaluation_normals"] + groups["anomalies"]
    evaluation_optional_mean = float(np.mean([
        selected_by_parent[index].sum() for index in evaluation_indices
    ]))
    stage_count = int(geometry["stage_one_crops_per_parent"])
    evaluation_crop_mean = stage_count + evaluation_optional_mean
    budget_reduction = 1.0 - evaluation_crop_mean / crops_per_parent

    coverage_rows = []
    for index in split["anomalies"]:
        mask = union_masks.get(index)
        if mask is None or not np.any(mask):
            continue
        optional_available = np.isfinite(
            accumulators[index]["view_x_shifted_grid"]
        )
        eligible = mask & optional_available
        coverage_rows.append({
            "parent_index": int(index),
            "defect_optional_eligible_cells": int(eligible.sum()),
            "selected_defect_optional_cells": int((eligible & selected_masks[index]).sum()),
        })
    eligible_total = sum(row["defect_optional_eligible_cells"] for row in coverage_rows)
    selected_total = sum(row["selected_defect_optional_cells"] for row in coverage_rows)

    replay_difference = None
    if not args.smoke:
        reference_path = args.full_reference or Path(str(probe["frozen_full_reference"]))
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        replay_difference = max_metric_difference(
            compact["full_pcaf"],
            reference["compact_primary_metrics"]["context_bias_weighted_mean"],
        )

    gate = probe["evidence_gate"]
    candidate = compact[str(gate["candidate"])]
    stage = compact[str(gate["stage_one_reference"])]
    matched = compact[str(gate["matched_budget_reference"])]
    full = compact[str(gate["full_reference"])]
    checks = {
        "full_reference_exact_replay": args.smoke or (
            replay_difference is not None and replay_difference <= 1e-8
        ),
        "budget_reduction": budget_reduction
        >= float(gate["minimum_evaluation_budget_reduction_vs_full"]),
        "pixel_ap_retained_vs_full": candidate["pixel_average_precision"]
        >= full["pixel_average_precision"] - float(gate["maximum_pixel_ap_loss_vs_full"]),
        "pixel_ap_gain_over_stage_one": candidate["pixel_average_precision"]
        >= stage["pixel_average_precision"]
        + float(gate["minimum_pixel_ap_gain_over_stage_one"]),
        "pixel_ap_gain_over_matched_budget": candidate["pixel_average_precision"]
        >= matched["pixel_average_precision"]
        + float(gate["minimum_pixel_ap_gain_over_matched_budget"]),
        "parent_ap_retained": candidate["parent_average_precision"]
        >= full["parent_average_precision"] - float(gate["maximum_parent_ap_loss_vs_full"]),
        "small_auc_retained": candidate["instance_auc_small"]
        >= full["instance_auc_small"]
        - float(gate["maximum_small_instance_auc_loss_vs_full"]),
        "elongated_auc_retained": candidate["instance_auc_elongated"]
        >= full["instance_auc_elongated"]
        - float(gate["maximum_elongated_instance_auc_loss_vs_full"]),
        "normal_false_positives_retained": candidate["target_normal_false_positives"]
        <= full["target_normal_false_positives"]
        + int(gate["maximum_target_normal_false_positive_increment_vs_full"]),
    }

    result = {
        "schema_version": 1,
        "config": probe,
        "run": {
            "fold": fold,
            "smoke": args.smoke,
            "data_root": str(data_root),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "predict_seconds_for_full_observation_replay": predict_seconds,
            "calibration_selection_fusion_seconds": calibration_seconds,
            "peak_cuda_gib": (
                torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            ),
            "script_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(args.config),
            **model_facts,
        },
        "counts": {
            **{key: len(value) for key, value in split.items()},
            "scored_parents": len(scored_indices),
            "full_crops_per_parent": crops_per_parent,
            "stage_one_crops_per_parent": stage_count,
            "optional_crop_slots": len(slots),
            "evaluation_online_crops_per_parent_mean": evaluation_crop_mean,
            "evaluation_online_crop_ratio_to_full": evaluation_crop_mean / crops_per_parent,
            "evaluation_online_budget_reduction_vs_full": budget_reduction,
        },
        "coordinate_calibration": {
            "relative_bins": int(centers.size),
            "samples_per_bin_minimum": int(calibration_counts.min()),
            "samples_per_bin_median": float(np.median(calibration_counts)),
            "robust_scale_minimum": float(scales.min()),
            "robust_scale_median": float(np.median(scales)),
        },
        "selector": {
            "slot_threshold_minimum": float(thresholds.min()),
            "slot_threshold_median": float(np.median(thresholds)),
            "slot_threshold_maximum": float(thresholds.max()),
            "selection_summary": selection_summary,
            "anomaly_defect_cell_optional_view_coverage": (
                float(selected_total / eligible_total) if eligible_total else None
            ),
            "anomaly_defect_cell_coverage_counts": {
                "eligible": int(eligible_total),
                "selected": int(selected_total),
            },
        },
        "compact_primary_metrics": compact,
        "full_reference_max_absolute_compact_difference": replay_difference,
        "fixed_candidate_gate": {
            "checks": checks,
            "passed": all(checks.values()),
            "scope": gate["formal_expansion"],
        },
        "metrics": metrics,
    }
    safe = json_safe(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "run": safe["run"],
        "counts": safe["counts"],
        "selector": safe["selector"],
        "compact_primary_metrics": safe["compact_primary_metrics"],
        "fixed_candidate_gate": safe["fixed_candidate_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
