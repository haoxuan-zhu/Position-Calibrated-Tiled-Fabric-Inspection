"""Evaluate fixed center-weighted fusion against frozen RAW-FABRID PCAF.

The detector, crop geometry, split, score normalization, and source-calibrated
operating-point logic are imported from the frozen PCAF experiment.  This probe
adds only deterministic reconstruction rules whose weights depend on a token's
coordinate inside its crop.  It deliberately does not edit the hash-locked
PCAF script or use target labels to select a window.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

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


GEOMETRIC_VARIANTS = ("gaussian", "hann", "hard_center")
REFERENCE_VARIANTS = ("mean", "context_bias_weighted_mean")
COMPACT_TOLERANCE = 1e-8


def gaussian_importance_map(edge: int, sigma_scale: float = 1.0 / 8.0) -> np.ndarray:
    """Reproduce nnU-Net's impulse-filtered Gaussian importance map on CPU."""
    impulse = np.zeros((edge, edge), dtype=np.float64)
    impulse[(edge // 2, edge // 2)] = 1.0
    weights = gaussian_filter(
        impulse,
        sigma=(edge * sigma_scale, edge * sigma_scale),
        order=0,
        mode="constant",
        cval=0,
    )
    weights /= weights.max()
    zero = weights == 0
    if np.any(zero):
        weights[zero] = weights[~zero].min()
    return weights


def hann_importance_map(edge: int) -> np.ndarray:
    """Return an exact separable symmetric Hann window with a unit maximum."""
    one_dimensional = np.hanning(edge).astype(np.float64)
    weights = np.outer(one_dimensional, one_dimensional)
    return weights / weights.max()


def hard_center_importance_map(edge: int) -> np.ndarray:
    """Rank crop tokens by their Chebyshev distance from the crop boundary."""
    yy, xx = np.indices((edge, edge))
    return np.minimum.reduce((yy, xx, edge - 1 - yy, edge - 1 - xx)).astype(
        np.float64
    )


def relative_coordinate_ids(
    shape: tuple[int, int], crop_tokens: int, shift_tokens: int
) -> dict[str, np.ndarray]:
    """Map every parent token to its crop-relative coordinate for each grid."""
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


def geometric_fusion_fields(
    accumulators: dict[int, dict[str, np.ndarray]],
    expected_indices: list[int],
    crop_tokens: int,
    shift_tokens: int,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    """Fuse registered observations using three score-independent windows."""
    shape = accumulators[expected_indices[0]]["count"].shape
    relative_ids = relative_coordinate_ids(shape, crop_tokens, shift_tokens)
    windows = {
        "gaussian": gaussian_importance_map(crop_tokens),
        "hann": hann_importance_map(crop_tokens),
        "hard_center": hard_center_importance_map(crop_tokens),
    }
    flattened = {name: window.reshape(-1) for name, window in windows.items()}
    fields: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in GEOMETRIC_VARIANTS
    }
    hann_fallback_cells = 0

    for index in expected_indices:
        accumulator = accumulators[index]
        weighted_sums = {
            "gaussian": np.zeros(shape, dtype=np.float64),
            "hann": np.zeros(shape, dtype=np.float64),
        }
        weight_sums = {
            "gaussian": np.zeros(shape, dtype=np.float64),
            "hann": np.zeros(shape, dtype=np.float64),
        }
        hard_best = np.full(shape, -np.inf, dtype=np.float64)
        hard_sum = np.zeros(shape, dtype=np.float64)
        hard_count = np.zeros(shape, dtype=np.uint8)

        for view in OBSERVATION_VIEWS:
            values = accumulator[f"view_{view}"]
            valid = np.isfinite(values)
            ids = relative_ids[view]
            for name in ("gaussian", "hann"):
                weights = flattened[name][ids[valid]]
                weighted_sums[name][valid] += values[valid].astype(np.float64) * weights
                weight_sums[name][valid] += weights

            center_rank = flattened["hard_center"][ids[valid]]
            current_best = hard_best[valid]
            better = center_rank > current_best
            tied = center_rank == current_best
            selected_sum = hard_sum[valid]
            selected_count = hard_count[valid]
            selected_sum[better] = values[valid][better]
            selected_count[better] = 1
            selected_sum[tied] += values[valid][tied]
            selected_count[tied] += 1
            hard_sum[valid] = selected_sum
            hard_count[valid] = selected_count
            hard_best[valid] = np.maximum(current_best, center_rank)

        equal_mean = accumulator["sum"] / accumulator["count"].astype(np.float64)
        for name in ("gaussian", "hann"):
            output = np.empty(shape, dtype=np.float64)
            positive = weight_sums[name] > 0
            output[positive] = weighted_sums[name][positive] / weight_sums[name][positive]
            output[~positive] = equal_mean[~positive]
            if name == "hann":
                hann_fallback_cells += int(np.sum(~positive))
            fields[name][index] = output.astype(np.float32)
        if np.any(hard_count == 0):
            raise RuntimeError(f"Hard-center fusion left parent {index} uncovered")
        fields["hard_center"][index] = (hard_sum / hard_count).astype(np.float32)

    return fields, {
        "uses_scores_to_define_weights": False,
        "uses_labels": False,
        "crop_token_edge": crop_tokens,
        "gaussian_sigma_scale": 1.0 / 8.0,
        "gaussian_minimum_weight": float(windows["gaussian"].min()),
        "hann_zero_weight_fallback": "equal mean of available observations",
        "hann_fallback_parent_cells": hann_fallback_cells,
        "hard_center_tie_rule": "equal mean among observations tied for maximum boundary distance",
    }


def verify_frozen_reference(
    compact: dict[str, dict[str, float | int]], reference_path: Path
) -> dict[str, Any]:
    """Fail if rescoring no longer reproduces the frozen mean and PCAF metrics."""
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = reference["compact_primary_metrics"]
    errors: dict[str, dict[str, float]] = {}
    maximum = 0.0
    for variant in REFERENCE_VARIANTS:
        errors[variant] = {}
        for metric, value in compact[variant].items():
            difference = abs(float(value) - float(expected[variant][metric]))
            errors[variant][metric] = difference
            maximum = max(maximum, difference)
    if maximum > COMPACT_TOLERANCE:
        raise RuntimeError(
            f"Frozen reference mismatch: maximum compact-metric error {maximum:.3g}"
        )
    return {
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "absolute_tolerance": COMPACT_TOLERANCE,
        "maximum_absolute_error": maximum,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    base_path = Path(str(config["base_config"]))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    data_root = args.data_root or Path(str(base["data"]["root"]))
    rows = load_rows(data_root, base["data"])
    by_roll: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_roll[str(row["roll_id"])].append(index)
    fold = str(args.fold)
    split = split_p2(fold, by_roll, rows)
    if args.smoke:
        split = {
            "training_normals": split["training_normals"][:2],
            "calibration_normals": split["calibration_normals"][:2],
            "evaluation_normals": split["evaluation_normals"][:2],
            "anomalies": split["anomalies"][:2],
        }

    geometry = config["crop_observations"]
    edge = int(geometry["physical_crop_edge"])
    shift = int(geometry["phase_shift_pixels"])
    stride = int(geometry["output_stride_pixels"])
    height = int(base["data"]["image_height"])
    width = int(base["data"]["image_width"])
    if any(value % stride for value in (edge, shift, height, width)):
        raise ValueError("Crop, shift, and parent geometry must align to output stride")
    scored_indices = sorted(set(
        split["calibration_normals"] + split["evaluation_normals"] + split["anomalies"]
    ))
    refs = make_crop_refs(
        scored_indices,
        height,
        width,
        edge,
        shift,
        [str(view) for view in geometry["views"]],
    )
    crops_per_parent = len(refs) // len(scored_indices)
    if crops_per_parent != int(geometry["crops_per_parent"]):
        raise RuntimeError(f"Expected 73 crops per parent, obtained {crops_per_parent}")

    model_config = base["model"]
    workers = 0 if args.smoke else int(model_config["num_workers"])
    loader = make_loader(
        data_root,
        rows,
        refs,
        edge,
        "patchcore",
        int(base["tiling"].get("model_edge_size", edge)),
        int(model_config.get("predict_batch_size", 8)),
        workers,
    )
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False

    checkpoint = args.checkpoint or Path(
        str(config["checkpoint_pattern"]).format(fold=fold)
    )
    parent_shape = (height // stride, width // stride)
    output_edge = edge // stride
    accumulators, predict_seconds, model_facts = score_patchcore(
        base, checkpoint, loader, refs, device, output_edge, stride, parent_shape
    )

    fusion_started = time.perf_counter()
    raw_variants, _, context_facts = finalize_fusions(
        accumulators,
        scored_indices,
        ["single_base", "mean", "context_bias_weighted_mean"],
        list(split["calibration_normals"]),
        output_edge,
        shift // stride,
    )
    geometry_fields, geometry_facts = geometric_fusion_fields(
        accumulators, scored_indices, output_edge, shift // stride
    )
    raw_variants.update(geometry_fields)
    fusion_seconds = time.perf_counter() - fusion_started

    maps = {
        variant: {
            index: score_modes(raw, int(base["scores"]["local_mean_kernel_tokens"]))
            for index, raw in parent_maps.items()
        }
        for variant, parent_maps in raw_variants.items()
    }
    coco = json.loads((data_root / str(base["data"]["coco"])).read_text(encoding="utf-8"))
    annotations, union_masks = annotation_masks(coco, rows, *parent_shape)
    evaluation_started = time.perf_counter()
    metrics = {
        variant: evaluate(parent_maps, split, annotations, union_masks, base)
        for variant, parent_maps in maps.items()
    }
    evaluation_seconds = time.perf_counter() - evaluation_started
    mode = str(config["primary_score_mode"])
    compact = {
        variant: compact_metrics(current, mode, len(split["evaluation_normals"]))
        for variant, current in metrics.items()
    }
    reference_path = args.reference or Path(
        str(config["reference_result_pattern"]).format(fold=fold)
    )
    replay_check = None if args.smoke else verify_frozen_reference(compact, reference_path)

    result = {
        "schema_version": 1,
        "config": config,
        "run": {
            "fold": fold,
            "smoke": args.smoke,
            "data_root": str(data_root),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "predict_seconds": predict_seconds,
            "fusion_seconds": fusion_seconds,
            "evaluation_seconds": evaluation_seconds,
            "milliseconds_per_parent": 1000.0 * predict_seconds / len(scored_indices),
            "fusion_milliseconds_per_parent": 1000.0 * fusion_seconds / len(scored_indices),
            "peak_cuda_gib": (
                torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            ),
            "script_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(args.config),
            "checkpoint_sha256": sha256_file(checkpoint),
            **model_facts,
        },
        "counts": {
            **{key: len(value) for key, value in split.items()},
            "scored_parents": len(scored_indices),
            "crops_per_parent": crops_per_parent,
            "total_crops": len(refs),
        },
        "split_indices": split,
        "geometric_fusion": geometry_facts,
        "context_response_calibration": context_facts,
        "frozen_replay_check": replay_check,
        "compact_primary_metrics": compact,
        "metrics": metrics,
    }
    safe_result = json_safe(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(safe_result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "run": safe_result["run"],
        "counts": safe_result["counts"],
        "geometric_fusion": safe_result["geometric_fusion"],
        "frozen_replay_check": safe_result["frozen_replay_check"],
        "compact_primary_metrics": safe_result["compact_primary_metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
