"""Audit crop-coordinate calibration with physical-location fixed effects.

The frozen PCAF experiment estimates one robust response center per crop-relative
coordinate by pooling source-normal observations.  Pooling is useful, but it can
mix a detector's crop-coordinate response with parent-coordinate texture or
illumination.  This audit fits the alternative model

    s(v, p) = alpha_p + b[r(v, p)] + epsilon(v, p)

on source-normal parents.  Eliminating ``alpha_p`` by within-location centering
makes ``b`` depend only on differences between observations of the same physical
location.  The scan graph has several disconnected coordinate components, so the
script reports that non-identifiability explicitly and uses pooled normal centers
only to anchor one unavoidable constant per component.

This file is an additive audit.  It does not modify the frozen seven-fold runner
or its recorded JSON results.
"""

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
    OBSERVATION_VIEWS,
    compact_metrics,
    finalize_fusions,
    json_safe,
    make_crop_refs,
    make_loader,
    score_patchcore,
    sha256_file,
)


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


def fit_stacks(
    accumulators: dict[int, dict[str, np.ndarray]], fit_indices: list[int]
) -> dict[str, np.ndarray]:
    if not fit_indices:
        raise ValueError("At least one source-normal parent is required")
    return {
        view: np.stack(
            [accumulators[index][f"view_{view}"] for index in fit_indices], axis=0
        ).astype(np.float32, copy=False)
        for view in OBSERVATION_VIEWS
    }


def pooled_coordinate_calibration(
    stacks: dict[str, np.ndarray],
    ids: dict[str, np.ndarray],
    relative_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.empty(relative_bins, dtype=np.float64)
    scales = np.empty(relative_bins, dtype=np.float64)
    counts = np.empty(relative_bins, dtype=np.int64)
    for relative_id in range(relative_bins):
        chunks: list[np.ndarray] = []
        for view in OBSERVATION_VIEWS:
            values = stacks[view][:, ids[view] == relative_id].reshape(-1)
            chunks.append(values[np.isfinite(values)])
        samples = np.concatenate(chunks)
        if not samples.size:
            raise RuntimeError(f"No observations for relative coordinate {relative_id}")
        center = float(np.median(samples))
        mad = float(np.median(np.abs(samples - center)))
        centers[relative_id] = center
        scales[relative_id] = max(1.4826 * mad, 1e-6)
        counts[relative_id] = samples.size
    return centers, scales, counts


def coordinate_components(
    crop_tokens: int, shift_tokens: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    divisor = math.gcd(crop_tokens, shift_tokens)
    component_count = divisor * divisor
    component_side = crop_tokens // divisor
    component_size = component_side * component_side
    components = np.empty(crop_tokens * crop_tokens, dtype=np.int32)
    local_indices = np.empty_like(components)
    for relative_id in range(crop_tokens * crop_tokens):
        row, column = divmod(relative_id, crop_tokens)
        components[relative_id] = (row % divisor) * divisor + column % divisor
        local_indices[relative_id] = (row // divisor) * component_side + column // divisor
    return components, local_indices, component_count, component_size


def fit_physical_fixed_effects(
    stacks: dict[str, np.ndarray],
    ids: dict[str, np.ndarray],
    crop_tokens: int,
    shift_tokens: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit exact least-squares location and crop-coordinate fixed effects.

    Physical-location intercepts are analytically eliminated by subtracting the
    within-location observation mean.  A minimum-norm solution fixes the mean of
    each disconnected coordinate component at zero.
    """

    components, local_indices, component_count, component_size = coordinate_components(
        crop_tokens, shift_tokens
    )
    hessians = np.zeros((component_count, component_size, component_size), dtype=np.float64)
    gradients = np.zeros((component_count, component_size), dtype=np.float64)
    parent_count, height, width = next(iter(stacks.values())).shape
    multi_observation_locations = 0
    centered_observations = 0

    for y in range(height):
        for x in range(width):
            views = [
                view for view in OBSERVATION_VIEWS if np.isfinite(stacks[view][0, y, x])
            ]
            if len(views) < 2:
                continue
            values = np.stack([stacks[view][:, y, x] for view in views], axis=1).astype(
                np.float64, copy=False
            )
            if not np.isfinite(values).all():
                raise RuntimeError("Crop coverage changed across calibration parents")
            relative = np.asarray([ids[view][y, x] for view in views], dtype=np.int64)
            current_components = components[relative]
            if np.unique(current_components).size != 1:
                raise RuntimeError("One physical location crossed disconnected coordinate components")
            component = int(current_components[0])
            local = local_indices[relative]
            count = len(views)
            centering = np.eye(count, dtype=np.float64) - np.full(
                (count, count), 1.0 / count, dtype=np.float64
            )
            hessians[component][np.ix_(local, local)] += parent_count * centering
            gradients[component][local] += (values - values.mean(axis=1, keepdims=True)).sum(axis=0)
            multi_observation_locations += parent_count
            centered_observations += parent_count * count

    zero_gauge = np.empty(crop_tokens * crop_tokens, dtype=np.float64)
    ranks: list[int] = []
    condition_numbers: list[float] = []
    for component in range(component_count):
        solution, _, rank, singular = np.linalg.lstsq(
            hessians[component], gradients[component], rcond=1e-10
        )
        members = np.flatnonzero(components == component)
        zero_gauge[members] = solution[local_indices[members]]
        ranks.append(int(rank))
        positive = singular[singular > singular.max(initial=0.0) * 1e-10]
        condition_numbers.append(
            float(positive.max() / positive.min()) if positive.size else float("inf")
        )

    # Estimate the residual observation scale after alpha_p and b_r are removed.
    residual_sum_sq = np.zeros(crop_tokens * crop_tokens, dtype=np.float64)
    residual_counts = np.zeros(crop_tokens * crop_tokens, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            views = [
                view for view in OBSERVATION_VIEWS if np.isfinite(stacks[view][0, y, x])
            ]
            if len(views) < 2:
                continue
            relative = np.asarray([ids[view][y, x] for view in views], dtype=np.int64)
            values = np.stack([stacks[view][:, y, x] for view in views], axis=1).astype(
                np.float64, copy=False
            )
            adjusted = values - zero_gauge[relative][None, :]
            residual = adjusted - adjusted.mean(axis=1, keepdims=True)
            for position, relative_id in enumerate(relative):
                residual_sum_sq[relative_id] += np.square(residual[:, position]).sum()
                residual_counts[relative_id] += parent_count
    residual_scales = np.sqrt(
        residual_sum_sq / np.maximum(residual_counts, 1)
    )
    positive_scales = residual_scales[residual_counts > 0]
    floor = max(float(np.quantile(positive_scales, 0.01)) * 0.1, 1e-6)
    residual_scales = np.maximum(residual_scales, floor)

    expected_rank = component_size - 1
    facts = {
        "fit_normal_parents": parent_count,
        "multi_observation_physical_locations": multi_observation_locations,
        "centered_observations": centered_observations,
        "coordinate_bins": crop_tokens * crop_tokens,
        "coordinate_graph_components": component_count,
        "component_size": component_size,
        "unidentifiable_component_constants": component_count,
        "rank_per_component": ranks,
        "expected_rank_per_component": expected_rank,
        "all_components_have_expected_rank": all(rank == expected_rank for rank in ranks),
        "condition_number_identifiable_subspace_minimum": float(np.min(condition_numbers)),
        "condition_number_identifiable_subspace_median": float(np.median(condition_numbers)),
        "condition_number_identifiable_subspace_maximum": float(np.max(condition_numbers)),
        "gauge": "minimum-norm zero mean within each disconnected coordinate component",
        "residual_scale_floor": floor,
    }
    return zero_gauge, residual_scales, facts


def anchor_components(
    zero_gauge: np.ndarray,
    pooled_centers: np.ndarray,
    crop_tokens: int,
    shift_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    components, _, component_count, _ = coordinate_components(crop_tokens, shift_tokens)
    anchors = np.empty(component_count, dtype=np.float64)
    anchored = zero_gauge.copy()
    for component in range(component_count):
        selected = components == component
        anchors[component] = float(np.median(pooled_centers[selected] - zero_gauge[selected]))
        anchored[selected] += anchors[component]
    return anchored, anchors


def fuse_calibrated(
    accumulators: dict[int, dict[str, np.ndarray]],
    expected_indices: list[int],
    ids: dict[str, np.ndarray],
    centers: np.ndarray,
    scales: np.ndarray | None,
) -> dict[int, np.ndarray]:
    fields: dict[int, np.ndarray] = {}
    shape = ids["base_grid"].shape
    for index in expected_indices:
        weighted_sum = np.zeros(shape, dtype=np.float64)
        weight_sum = np.zeros(shape, dtype=np.float64)
        for view in OBSERVATION_VIEWS:
            values = accumulators[index][f"view_{view}"]
            valid = np.isfinite(values)
            relative = ids[view]
            if scales is None:
                weights = np.ones(int(valid.sum()), dtype=np.float64)
            else:
                weights = 1.0 / np.square(scales[relative[valid]])
            weighted_sum[valid] += (
                values[valid].astype(np.float64) - centers[relative[valid]]
            ) * weights
            weight_sum[valid] += weights
        if np.any(weight_sum == 0):
            raise RuntimeError(f"Uncovered parent cells for index {index}")
        fields[index] = (weighted_sum / weight_sum).astype(np.float32)
    return fields


def overlap_difference_diagnostics(
    stacks: dict[str, np.ndarray],
    ids: dict[str, np.ndarray],
    pooled_centers: np.ndarray,
    fixed_effect_centers: np.ndarray,
) -> dict[str, Any]:
    groups: dict[str, list[np.ndarray]] = {
        "raw": [], "pooled_corrected": [], "fixed_effect_corrected": []
    }
    views = list(OBSERVATION_VIEWS)
    for left_position in range(len(views)):
        for right_position in range(left_position + 1, len(views)):
            left = views[left_position]
            right = views[right_position]
            valid = np.isfinite(stacks[left][0]) & np.isfinite(stacks[right][0])
            raw = (stacks[left][:, valid] - stacks[right][:, valid]).reshape(-1).astype(
                np.float64, copy=False
            )
            left_ids = ids[left][valid]
            right_ids = ids[right][valid]
            pooled_delta = pooled_centers[left_ids] - pooled_centers[right_ids]
            fixed_delta = fixed_effect_centers[left_ids] - fixed_effect_centers[right_ids]
            spatial_repeats = stacks[left].shape[0]
            groups["raw"].append(raw)
            groups["pooled_corrected"].append(
                raw - np.tile(pooled_delta, spatial_repeats)
            )
            groups["fixed_effect_corrected"].append(
                raw - np.tile(fixed_delta, spatial_repeats)
            )

    result: dict[str, Any] = {}
    for name, chunks in groups.items():
        values = np.concatenate(chunks)
        absolute = np.abs(values)
        result[name] = {
            "paired_differences": int(values.size),
            "signed_mean": float(values.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(values)))),
            "absolute_mean": float(absolute.mean()),
            "absolute_median": float(np.median(absolute)),
            "absolute_q90": float(np.quantile(absolute, 0.9)),
        }
    return result


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, spearman: bool = False) -> float:
    if spearman:
        left = rankdata(left)
        right = rankdata(right)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return float(np.dot(left_centered, right_centered) / denominator) if denominator else 0.0


def pairwise_stability(maps: dict[str, np.ndarray]) -> dict[str, Any]:
    names = sorted(maps)
    pearson: list[float] = []
    spearman: list[float] = []
    pairs: dict[str, dict[str, float]] = {}
    for left_position, left_name in enumerate(names):
        for right_name in names[left_position + 1 :]:
            left = maps[left_name].reshape(-1).astype(np.float64)
            right = maps[right_name].reshape(-1).astype(np.float64)
            current_pearson = correlation(left, right)
            current_spearman = correlation(left, right, spearman=True)
            pearson.append(current_pearson)
            spearman.append(current_spearman)
            pairs[f"{left_name}__{right_name}"] = {
                "pearson": current_pearson, "spearman": current_spearman
            }
    if not pearson:
        return {
            "rolls": names,
            "pairs": pairs,
            "pair_count": 0,
            "pearson_minimum": None,
            "pearson_median": None,
            "pearson_maximum": None,
            "spearman_minimum": None,
            "spearman_median": None,
            "spearman_maximum": None,
        }
    return {
        "rolls": names,
        "pairs": pairs,
        "pair_count": len(pearson),
        "pearson_minimum": float(np.min(pearson)),
        "pearson_median": float(np.median(pearson)),
        "pearson_maximum": float(np.max(pearson)),
        "spearman_minimum": float(np.min(spearman)),
        "spearman_median": float(np.median(spearman)),
        "spearman_maximum": float(np.max(spearman)),
    }


def spatial_summary(values: np.ndarray, crop_tokens: int, boundary_band: int = 4) -> dict[str, Any]:
    field = values.reshape(crop_tokens, crop_tokens)
    yy, xx = np.indices(field.shape)
    boundary = (
        (yy < boundary_band) | (yy >= crop_tokens - boundary_band)
        | (xx < boundary_band) | (xx >= crop_tokens - boundary_band)
    )
    interior = ~boundary
    return {
        "boundary_band_tokens": boundary_band,
        "boundary_mean": float(field[boundary].mean()),
        "boundary_median": float(np.median(field[boundary])),
        "interior_mean": float(field[interior].mean()),
        "interior_median": float(np.median(field[interior])),
        "boundary_minus_interior_mean": float(field[boundary].mean() - field[interior].mean()),
        "boundary_minus_interior_median": float(
            np.median(field[boundary]) - np.median(field[interior])
        ),
    }


def operating_point(metrics: dict[str, Any], mode: str, normal_count: int) -> dict[str, Any]:
    current = metrics[mode]
    source = current["source_calibrated"]
    return {
        "image_threshold": float(source["image_threshold"]),
        "pixel_threshold": float(source["pixel_threshold"]),
        "target_normal_fpr": float(source["normal_parent_fpr"]),
        "target_normal_false_positives": int(round(source["normal_parent_fpr"] * normal_count)),
        "defect_parent_recall": float(source["defect_parent_recall_at_1pct_fpr"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("calibration_output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", default="Rollo4A")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    probe = json.loads(args.config.read_text(encoding="utf-8"))
    base_path = Path(str(probe["base_configs"]["patchcore"]))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    data_root = args.data_root or Path(str(base["data"]["root"]))
    rows = load_rows(data_root, base["data"])
    by_roll: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_roll[str(row["roll_id"])].append(index)
    split = split_p2(str(args.fold), by_roll, rows)
    if args.smoke:
        split = {
            "training_normals": split["training_normals"][:2],
            "calibration_normals": split["calibration_normals"][:8],
            "evaluation_normals": split["evaluation_normals"][:2],
            "anomalies": split["anomalies"][:2],
        }

    geometry = probe["crop_observations"]
    edge = int(geometry["physical_crop_edge"])
    shift = int(geometry["phase_shift_pixels"])
    stride = int(geometry["output_stride_pixels"])
    height = int(base["data"]["image_height"])
    width = int(base["data"]["image_width"])
    crop_tokens = edge // stride
    shift_tokens = shift // stride
    parent_shape = (height // stride, width // stride)
    scored_indices = sorted(set(
        split["calibration_normals"] + split["evaluation_normals"] + split["anomalies"]
    ))
    refs = make_crop_refs(
        scored_indices, height, width, edge, shift,
        [str(view) for view in geometry["views"]],
    )
    model_config = base["model"]
    loader = make_loader(
        data_root, rows, refs, edge, "patchcore",
        int(base["tiling"].get("model_edge_size", edge)),
        int(model_config.get("predict_batch_size", 8)),
        0 if args.smoke else int(model_config["num_workers"]),
    )
    torch.manual_seed(int(probe["seed"]))
    np.random.seed(int(probe["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(probe["seed"]))
        torch.backends.cuda.matmul.allow_tf32 = False

    accumulators, predict_seconds, model_facts = score_patchcore(
        base, args.checkpoint, loader, refs, device, edge // stride, stride, parent_shape
    )
    calibration_started = time.perf_counter()
    ids = relative_coordinate_ids(parent_shape, crop_tokens, shift_tokens)
    stacks = fit_stacks(accumulators, split["calibration_normals"])
    relative_bins = crop_tokens * crop_tokens
    pooled_centers, pooled_scales, sample_counts = pooled_coordinate_calibration(
        stacks, ids, relative_bins
    )
    zero_gauge, residual_scales, fixed_effect_facts = fit_physical_fixed_effects(
        stacks, ids, crop_tokens, shift_tokens
    )
    anchored_centers, component_anchors = anchor_components(
        zero_gauge, pooled_centers, crop_tokens, shift_tokens
    )

    raw_variants, _, frozen_context_facts = finalize_fusions(
        accumulators,
        scored_indices,
        ["mean", "context_bias_equal_mean", "context_bias_weighted_mean"],
        split["calibration_normals"],
        crop_tokens,
        shift_tokens,
    )
    audited_fields = {
        "fixed_effect_equal_mean": fuse_calibrated(
            accumulators, scored_indices, ids, anchored_centers, None
        ),
        "fixed_effect_weighted_mean": fuse_calibrated(
            accumulators, scored_indices, ids, anchored_centers, pooled_scales
        ),
        "fixed_effect_zero_gauge_equal_mean": fuse_calibrated(
            accumulators, scored_indices, ids, zero_gauge, None
        ),
        "fixed_effect_residual_weighted_mean": fuse_calibrated(
            accumulators, scored_indices, ids, anchored_centers, residual_scales
        ),
    }
    pooled_replay = fuse_calibrated(
        accumulators, scored_indices, ids, pooled_centers, pooled_scales
    )
    replay_max_abs = max(
        float(np.max(np.abs(
            pooled_replay[index] - raw_variants["context_bias_weighted_mean"][index]
        )))
        for index in scored_indices
    )
    if replay_max_abs > 1e-5:
        raise RuntimeError(f"Pooled-calibration replay drifted by {replay_max_abs}")

    source_roll_pooled: dict[str, np.ndarray] = {}
    source_roll_fixed: dict[str, np.ndarray] = {}
    source_roll_anchored: dict[str, np.ndarray] = {}
    source_roll_counts: dict[str, int] = {}
    for roll in sorted(by_roll):
        if roll == str(args.fold):
            continue
        current = [
            index for index in split["calibration_normals"] if str(rows[index]["roll_id"]) == roll
        ]
        if not current:
            continue
        current_stacks = fit_stacks(accumulators, current)
        current_pooled, _, _ = pooled_coordinate_calibration(
            current_stacks, ids, relative_bins
        )
        current_fixed, _, _ = fit_physical_fixed_effects(
            current_stacks, ids, crop_tokens, shift_tokens
        )
        current_anchored, _ = anchor_components(
            current_fixed, current_pooled, crop_tokens, shift_tokens
        )
        source_roll_pooled[roll] = current_pooled
        source_roll_fixed[roll] = current_fixed
        source_roll_anchored[roll] = current_anchored
        source_roll_counts[roll] = len(current)

    difference_diagnostics = overlap_difference_diagnostics(
        stacks, ids, pooled_centers, anchored_centers
    )
    calibration_seconds = time.perf_counter() - calibration_started

    all_raw_fields = {**raw_variants, **audited_fields}
    maps = {
        variant: {
            index: score_modes(raw, int(base["scores"]["local_mean_kernel_tokens"]))
            for index, raw in fields.items()
        }
        for variant, fields in all_raw_fields.items()
    }
    coco = json.loads((data_root / str(base["data"]["coco"])).read_text(encoding="utf-8"))
    annotations, union_masks = annotation_masks(coco, rows, *parent_shape)
    evaluation_started = time.perf_counter()
    metrics = {
        variant: evaluate(parent_maps, split, annotations, union_masks, base)
        for variant, parent_maps in maps.items()
    }
    evaluation_seconds = time.perf_counter() - evaluation_started
    mode = str(probe["primary_score_mode"])
    compact = {
        variant: compact_metrics(current, mode, len(split["evaluation_normals"]))
        for variant, current in metrics.items()
    }
    operating_points = {
        variant: operating_point(current, mode, len(split["evaluation_normals"]))
        for variant, current in metrics.items()
    }

    components, _, component_count, _ = coordinate_components(crop_tokens, shift_tokens)
    within_component_center_correlations = []
    for component in range(component_count):
        selected = components == component
        within_component_center_correlations.append(
            correlation(pooled_centers[selected], anchored_centers[selected])
        )
    result = {
        "schema_version": 1,
        "purpose": "physical-location fixed-effect audit of crop-coordinate calibration",
        "run": {
            "fold": str(args.fold),
            "smoke": args.smoke,
            "data_root": str(data_root),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "predict_seconds": predict_seconds,
            "calibration_seconds": calibration_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": time.perf_counter() - started,
            "script_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(args.config),
            **model_facts,
        },
        "counts": {
            **{key: len(value) for key, value in split.items()},
            "scored_parents": len(scored_indices),
            "crops_per_parent": len(refs) // len(scored_indices),
            "source_roll_calibration_parents": source_roll_counts,
        },
        "identifiability": fixed_effect_facts,
        "calibration": {
            "pooled_replay_max_absolute_difference": replay_max_abs,
            "samples_per_coordinate_minimum": int(sample_counts.min()),
            "samples_per_coordinate_median": float(np.median(sample_counts)),
            "pooled_center_spatial_summary": spatial_summary(pooled_centers, crop_tokens),
            "pooled_scale_spatial_summary": spatial_summary(pooled_scales, crop_tokens),
            "fixed_effect_zero_gauge_spatial_summary": spatial_summary(zero_gauge, crop_tokens),
            "anchored_fixed_effect_spatial_summary": spatial_summary(
                anchored_centers, crop_tokens
            ),
            "pooled_vs_anchored_pearson": correlation(pooled_centers, anchored_centers),
            "pooled_vs_anchored_spearman": correlation(
                pooled_centers, anchored_centers, spearman=True
            ),
            "within_component_pooled_vs_fixed_pearson_minimum": float(
                np.min(within_component_center_correlations)
            ),
            "within_component_pooled_vs_fixed_pearson_median": float(
                np.median(within_component_center_correlations)
            ),
            "within_component_pooled_vs_fixed_pearson_maximum": float(
                np.max(within_component_center_correlations)
            ),
            "overlap_pair_differences": difference_diagnostics,
            "source_roll_stability": {
                "pooled_centers": pairwise_stability(source_roll_pooled),
                "fixed_effect_zero_gauge": pairwise_stability(source_roll_fixed),
                "fixed_effect_anchored": pairwise_stability(source_roll_anchored),
            },
            "frozen_pooled_calibration": frozen_context_facts,
        },
        "compact_primary_metrics": compact,
        "source_calibrated_operating_points": operating_points,
        "paired_fixed_effect_minus_mean": {
            variant: {
                metric: float(value - compact["mean"][metric])
                for metric, value in values.items()
                if isinstance(value, (int, float))
            }
            for variant, values in compact.items()
            if variant.startswith("fixed_effect")
        },
        "interpretation_guard": (
            "The paired fixed effect removes physical-location content only within the "
            "identifiable crop-coordinate graph. One constant per disconnected component "
            "is not identifiable from the frozen scan and is anchored by pooled source-normal centers."
        ),
    }
    safe_result = json_safe(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(safe_result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    roll_names = np.asarray(sorted(source_roll_pooled))
    np.savez_compressed(
        args.calibration_output,
        crop_tokens=np.asarray(crop_tokens, dtype=np.int64),
        shift_tokens=np.asarray(shift_tokens, dtype=np.int64),
        pooled_centers=pooled_centers.reshape(crop_tokens, crop_tokens).astype(np.float32),
        pooled_scales=pooled_scales.reshape(crop_tokens, crop_tokens).astype(np.float32),
        fixed_effect_zero_gauge=zero_gauge.reshape(crop_tokens, crop_tokens).astype(np.float32),
        fixed_effect_anchored=anchored_centers.reshape(crop_tokens, crop_tokens).astype(np.float32),
        residual_scales=residual_scales.reshape(crop_tokens, crop_tokens).astype(np.float32),
        component_anchors=component_anchors.astype(np.float32),
        source_roll_names=roll_names,
        source_roll_pooled=np.stack([source_roll_pooled[name] for name in roll_names]).reshape(
            len(roll_names), crop_tokens, crop_tokens
        ).astype(np.float32),
        source_roll_fixed=np.stack([source_roll_fixed[name] for name in roll_names]).reshape(
            len(roll_names), crop_tokens, crop_tokens
        ).astype(np.float32),
        source_roll_anchored=np.stack([source_roll_anchored[name] for name in roll_names]).reshape(
            len(roll_names), crop_tokens, crop_tokens
        ).astype(np.float32),
    )
    print(json.dumps({
        "run": safe_result["run"],
        "identifiability": safe_result["identifiability"],
        "overlap_pair_differences": safe_result["calibration"]["overlap_pair_differences"],
        "compact_primary_metrics": safe_result["compact_primary_metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
