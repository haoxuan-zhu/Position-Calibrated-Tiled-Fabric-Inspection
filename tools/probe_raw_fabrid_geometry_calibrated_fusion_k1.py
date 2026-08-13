"""Probe a product of fixed geometric and source-normal reliability weights.

K0 showed that deterministic center windows can be stronger than either equal
mean or PCAF.  This follow-up treats Gaussian position and source-normal robust
variance as complementary reliability factors.  Rollo4A is the declared
development gate; the other six rolls must not be run unless that gate passes.
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

from probe_raw_fabrid_anomalib_patchcore import evaluate, score_modes
from probe_raw_fabrid_geometric_fusion_baselines_k0 import (
    gaussian_importance_map,
    geometric_fusion_fields,
    relative_coordinate_ids,
    verify_frozen_reference,
)
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


CANDIDATE = "gaussian_bias_scale_weighted_mean"
HYBRID_VARIANTS = (
    "gaussian_bias_equal_mean",
    "gaussian_scale_weighted_mean",
    CANDIDATE,
)


def fit_coordinate_calibration(
    accumulators: dict[int, dict[str, np.ndarray]],
    fit_indices: list[int],
    relative_ids: dict[str, np.ndarray],
    relative_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the same pooled coordinate median and MAD used by frozen PCAF."""
    fit_stacks = {
        view: np.stack(
            [accumulators[index][f"view_{view}"] for index in fit_indices], axis=0
        ).astype(np.float32, copy=False)
        for view in OBSERVATION_VIEWS
    }
    centers = np.empty(relative_bins, dtype=np.float32)
    scales = np.empty(relative_bins, dtype=np.float32)
    sample_counts = np.empty(relative_bins, dtype=np.int64)
    for relative_id in range(relative_bins):
        chunks = []
        for view in OBSERVATION_VIEWS:
            values = fit_stacks[view][
                :, relative_ids[view] == relative_id
            ].reshape(-1)
            chunks.append(values[np.isfinite(values)])
        samples = np.concatenate(chunks)
        if not samples.size:
            raise RuntimeError(f"No normal samples for coordinate {relative_id}")
        center = float(np.median(samples))
        mad = float(np.median(np.abs(samples - center)))
        centers[relative_id] = center
        scales[relative_id] = max(1.4826 * mad, 1e-6)
        sample_counts[relative_id] = samples.size
    return centers, scales, sample_counts


def geometry_calibrated_fields(
    accumulators: dict[int, dict[str, np.ndarray]],
    fit_indices: list[int],
    expected_indices: list[int],
    crop_tokens: int,
    shift_tokens: int,
    pcaf_reference: dict[int, np.ndarray],
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    """Build three fixed ablations and verify the unit-geometry PCAF replica."""
    shape = accumulators[expected_indices[0]]["count"].shape
    coordinate_ids = relative_coordinate_ids(shape, crop_tokens, shift_tokens)
    centers, scales, sample_counts = fit_coordinate_calibration(
        accumulators, fit_indices, coordinate_ids, crop_tokens * crop_tokens
    )
    gaussian = gaussian_importance_map(crop_tokens).reshape(-1)
    fields = {name: {} for name in HYBRID_VARIANTS}
    pcaf_replica_max_error = 0.0

    for index in expected_indices:
        numerators = {
            "gaussian_bias_equal_mean": np.zeros(shape, dtype=np.float64),
            "gaussian_scale_weighted_mean": np.zeros(shape, dtype=np.float64),
            CANDIDATE: np.zeros(shape, dtype=np.float64),
        }
        denominators = {
            name: np.zeros(shape, dtype=np.float64) for name in HYBRID_VARIANTS
        }
        pcaf_numerator = np.zeros(shape, dtype=np.float64)
        pcaf_denominator = np.zeros(shape, dtype=np.float64)
        for view in OBSERVATION_VIEWS:
            values = accumulators[index][f"view_{view}"]
            valid = np.isfinite(values)
            ids = coordinate_ids[view][valid]
            raw = values[valid].astype(np.float64)
            bias_corrected = raw - centers[ids].astype(np.float64)
            geometry_weight = gaussian[ids]
            inverse_variance = 1.0 / scales[ids].astype(np.float64) ** 2
            product_weight = geometry_weight * inverse_variance

            numerators["gaussian_bias_equal_mean"][valid] += (
                geometry_weight * bias_corrected
            )
            denominators["gaussian_bias_equal_mean"][valid] += geometry_weight
            numerators["gaussian_scale_weighted_mean"][valid] += product_weight * raw
            denominators["gaussian_scale_weighted_mean"][valid] += product_weight
            numerators[CANDIDATE][valid] += product_weight * bias_corrected
            denominators[CANDIDATE][valid] += product_weight
            pcaf_numerator[valid] += inverse_variance * bias_corrected
            pcaf_denominator[valid] += inverse_variance

        for name in HYBRID_VARIANTS:
            if np.any(denominators[name] <= 0):
                raise RuntimeError(f"{name} left parent {index} uncovered")
            fields[name][index] = (
                numerators[name] / denominators[name]
            ).astype(np.float32)
        pcaf_replica = (pcaf_numerator / pcaf_denominator).astype(np.float32)
        pcaf_replica_max_error = max(
            pcaf_replica_max_error,
            float(np.max(np.abs(pcaf_replica - pcaf_reference[index]))),
        )

    if pcaf_replica_max_error > 1e-6:
        raise RuntimeError(
            f"Independent PCAF reconstruction mismatch: {pcaf_replica_max_error:.3g}"
        )
    return fields, {
        "uses_target_labels_or_scores_to_define_weights": False,
        "gaussian_sigma_scale": 1.0 / 8.0,
        "relative_coordinate_bins": crop_tokens * crop_tokens,
        "samples_per_relative_bin_minimum": int(sample_counts.min()),
        "samples_per_relative_bin_median": float(np.median(sample_counts)),
        "robust_scale_minimum": float(scales.min()),
        "robust_scale_median": float(np.median(scales)),
        "pcaf_replica_maximum_map_error": pcaf_replica_max_error,
    }


def development_gate(
    compact: dict[str, dict[str, float | int]], config: dict[str, Any], fold: str
) -> dict[str, Any] | None:
    gate = config["development_gate"]
    if fold != str(gate["fold"]):
        return None
    candidate = compact[CANDIDATE]
    references = [compact["gaussian"], compact["context_bias_weighted_mean"]]
    best_pixel = max(float(item["pixel_average_precision"]) for item in references)
    best_auc = max(float(item["instance_auc_all"]) for item in references)
    best_recall = max(float(item["source_parent_recall"]) for item in references)
    pcaf_fp = int(compact["context_bias_weighted_mean"]["target_normal_false_positives"])
    checks = {
        "pixel_ap_gain": (
            float(candidate["pixel_average_precision"]) - best_pixel
            >= float(gate["minimum_pixel_ap_gain_over_both_gaussian_and_pcaf"])
        ),
        "instance_auc_retained": (
            float(candidate["instance_auc_all"])
            >= best_auc - float(gate["maximum_instance_auc_loss_from_better_reference"])
        ),
        "parent_recall_retained": (
            float(candidate["source_parent_recall"])
            >= best_recall - float(gate["maximum_parent_recall_loss_from_better_reference"])
        ),
        "target_normal_fp_retained": (
            int(candidate["target_normal_false_positives"])
            <= pcaf_fp
            + int(gate["maximum_target_normal_false_positive_increase_over_pcaf"])
        ),
    }
    return {
        "candidate": CANDIDATE,
        "checks": checks,
        "pixel_ap_gain_over_better_reference": (
            float(candidate["pixel_average_precision"]) - best_pixel
        ),
        "instance_auc_delta_from_better_reference": (
            float(candidate["instance_auc_all"]) - best_auc
        ),
        "parent_recall_delta_from_better_reference": (
            float(candidate["source_parent_recall"]) - best_recall
        ),
        "target_normal_fp_delta_from_pcaf": (
            int(candidate["target_normal_false_positives"]) - pcaf_fp
        ),
        "passed": all(checks.values()),
        "scope": gate["interpretation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    base = json.loads(Path(str(config["base_config"])).read_text(encoding="utf-8"))
    data_root = args.data_root or Path(str(base["data"]["root"]))
    rows = load_rows(data_root, base["data"])
    by_roll: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_roll[str(row["roll_id"])].append(index)
    fold = str(args.fold)
    split = split_p2(fold, by_roll, rows)

    geometry = config["crop_observations"]
    edge = int(geometry["physical_crop_edge"])
    shift = int(geometry["phase_shift_pixels"])
    stride = int(geometry["output_stride_pixels"])
    height = int(base["data"]["image_height"])
    width = int(base["data"]["image_width"])
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
    loader = make_loader(
        data_root,
        rows,
        refs,
        edge,
        "patchcore",
        int(base["tiling"].get("model_edge_size", edge)),
        int(model_config.get("predict_batch_size", 8)),
        int(model_config["num_workers"]),
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
    geometric, geometric_facts = geometric_fusion_fields(
        accumulators, scored_indices, output_edge, shift // stride
    )
    raw_variants["gaussian"] = geometric["gaussian"]
    calibrated, calibrated_facts = geometry_calibrated_fields(
        accumulators,
        list(split["calibration_normals"]),
        scored_indices,
        output_edge,
        shift // stride,
        raw_variants["context_bias_weighted_mean"],
    )
    raw_variants.update(calibrated)
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
    replay_check = verify_frozen_reference(compact, reference_path)
    gate = development_gate(compact, config, fold)

    result = {
        "schema_version": 1,
        "config": config,
        "run": {
            "fold": fold,
            "data_root": str(data_root),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "predict_seconds": predict_seconds,
            "fusion_seconds": fusion_seconds,
            "evaluation_seconds": evaluation_seconds,
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
        "geometric_fusion": geometric_facts,
        "geometry_calibration": calibrated_facts,
        "context_response_calibration": context_facts,
        "frozen_replay_check": replay_check,
        "development_gate": gate,
        "compact_primary_metrics": compact,
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
        "geometry_calibration": safe["geometry_calibration"],
        "frozen_replay_check": safe["frozen_replay_check"],
        "development_gate": safe["development_gate"],
        "compact_primary_metrics": safe["compact_primary_metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
