"""Measure actual two-stage online latency for the frozen sequential PCAF policy.

The source-normal coordinate calibration and selector thresholds are fitted
offline.  Timed online runs compare a full 73-crop replay with the actual
49-crop first stage followed by scoring only the selected x-shifted crops.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from probe_raw_fabrid_anomalib_patchcore import evaluate, score_modes
from probe_raw_fabrid_grouped import annotation_masks, load_rows, split_p2
from probe_raw_fabrid_physical_field_k0 import (
    CropRef,
    add_crop,
    compact_metrics,
    json_safe,
    make_crop_refs,
    make_loader,
    sha256_file,
)
from probe_raw_fabrid_sequential_observation_k0 import (
    VIEWS,
    fit_coordinate_calibration,
    fuse_views,
    mask_from_selection,
    max_metric_difference,
    optional_crop_slots,
    relative_coordinate_ids,
    slot_statistics,
)
from probe_raw_patchcore_memory_quantization import embed, load_model


@torch.inference_mode()
def score_loaded_model(
    model: Any,
    loader: DataLoader,
    refs: list[CropRef],
    device: torch.device,
    output_edge: int,
    stride: int,
    parent_shape: tuple[int, int],
) -> tuple[dict[int, dict[str, np.ndarray]], float]:
    """Score refs with an already loaded PatchCore model and memory bank."""
    accumulators: dict[int, dict[str, np.ndarray]] = {}
    offset = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in loader:
        images, parents, ys, xs, views = batch
        images = images.to(device, non_blocking=True)
        if model.pre_processor is not None:
            images = model.pre_processor(images)
        embedding, grid_shape, output_size = embed(model, images)
        patch_scores, _ = model.model.nearest_neighbors(embedding, n_neighbors=1)
        patch_scores = patch_scores.reshape(images.shape[0], 1, *grid_shape)
        maps = model.model.anomaly_map_generator(patch_scores, output_size)
        maps = F.interpolate(
            maps.float(),
            size=(output_edge, output_edge),
            mode="bilinear",
            align_corners=False,
        )[:, 0].cpu().numpy().astype(np.float32, copy=False)
        for position, crop_map in enumerate(maps):
            add_crop(
                accumulators,
                int(parents[position]),
                int(ys[position]),
                int(xs[position]),
                str(views[position]),
                crop_map,
                stride,
                parent_shape,
            )
        offset += images.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if offset != len(refs):
        raise RuntimeError(f"Prediction count mismatch: {offset} != {len(refs)}")
    return accumulators, elapsed


def build_loader(
    data_root: Path,
    rows: list[dict[str, object]],
    refs: list[CropRef],
    edge: int,
    batch_size: int,
    workers: int,
) -> DataLoader:
    return make_loader(
        data_root,
        rows,
        refs,
        edge,
        "patchcore",
        edge,
        batch_size,
        workers,
    )


def x_shifted_positions(
    height: int, width: int, edge: int, shift: int
) -> list[tuple[int, int]]:
    return [
        (y0, x0)
        for y0 in range(0, height - edge + 1, edge)
        for x0 in range(shift, width - edge + 1, edge)
    ]


def median_record(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(statistics.median(record[key] for record in records))
        for key in records[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fold")
    parser.add_argument("--formal-reference", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    probe = json.loads(args.config.read_text(encoding="utf-8"))
    base_path = Path(str(probe["base_config"]))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    data_root = args.data_root or Path(str(base["data"]["root"]))
    checkpoint = args.checkpoint or Path(str(probe["checkpoint"]))
    fold = str(args.fold or probe["fold"])

    rows = load_rows(data_root, base["data"])
    by_roll: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_roll[str(row["roll_id"])].append(index)
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
    batch_size = int(base["model"].get("predict_batch_size", 8))
    workers = 0 if args.smoke else int(base["model"]["num_workers"])
    kernel = int(base["scores"]["local_mean_kernel_tokens"])
    mode = str(probe["primary_score_mode"])
    calibration_indices = list(split["calibration_normals"])
    evaluation_indices = list(split["evaluation_normals"] + split["anomalies"])

    torch.manual_seed(int(probe["seed"]))
    np.random.seed(int(probe["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(probe["seed"]))
        torch.backends.cuda.matmul.allow_tf32 = False

    model, bank = load_model(base, checkpoint, device)
    relative_ids = relative_coordinate_ids(parent_shape, crop_tokens, shift_tokens)
    slots = optional_crop_slots(height, width, edge, shift, stride)
    positions = x_shifted_positions(height, width, edge, shift)
    if len(slots) != len(positions):
        raise RuntimeError("Optional slot and crop-position orders disagree")

    # Offline source-normal fitting.  It is recorded but excluded from online latency.
    calibration_refs = make_crop_refs(
        calibration_indices, height, width, edge, shift, list(VIEWS)
    )
    calibration_loader = build_loader(
        data_root, rows, calibration_refs, edge, batch_size, workers
    )
    calibration_started = time.perf_counter()
    calibration_accumulators, calibration_detector_seconds = score_loaded_model(
        model,
        calibration_loader,
        calibration_refs,
        device,
        crop_tokens,
        stride,
        parent_shape,
    )
    centers, scales, calibration_counts = fit_coordinate_calibration(
        calibration_accumulators,
        calibration_indices,
        relative_ids,
        crop_tokens * crop_tokens,
    )
    stage_views = tuple(str(view) for view in geometry["stage_one_views"])
    calibration_stage_raw = {
        index: fuse_views(
            calibration_accumulators[index], stage_views, relative_ids, centers, scales
        )
        for index in calibration_indices
    }
    selector = probe["selector"]
    selector_mode = str(selector["score_mode"])
    fraction = float(selector["candidate_region_top_fraction"])
    calibration_stage_fields = {
        index: score_modes(raw, kernel)[selector_mode]
        for index, raw in calibration_stage_raw.items()
    }
    calibration_statistics = np.stack([
        slot_statistics(calibration_stage_fields[index], slots, fraction)
        for index in calibration_indices
    ])
    thresholds = np.quantile(
        calibration_statistics,
        float(selector["source_normal_quantile"]),
        axis=0,
        method=str(selector["quantile_method"]),
    )
    calibration_selected = {
        index: calibration_statistics[position] > thresholds
        for position, index in enumerate(calibration_indices)
    }
    calibration_full_raw = {
        index: fuse_views(
            calibration_accumulators[index], VIEWS, relative_ids, centers, scales
        )
        for index in calibration_indices
    }
    calibration_sequential_raw = {
        index: fuse_views(
            calibration_accumulators[index],
            VIEWS,
            relative_ids,
            centers,
            scales,
            mask_from_selection(parent_shape, slots, calibration_selected[index]),
        )
        for index in calibration_indices
    }
    calibration_seconds = time.perf_counter() - calibration_started

    full_refs = make_crop_refs(
        evaluation_indices, height, width, edge, shift, list(VIEWS)
    )
    stage_refs = make_crop_refs(
        evaluation_indices, height, width, edge, shift, list(stage_views)
    )

    # Untimed prepass warms the model, data workers, and filesystem cache.
    warmup_refs = make_crop_refs(
        evaluation_indices, height, width, edge, shift, ["base_grid"]
    )
    warmup_loader = build_loader(
        data_root, rows, warmup_refs, edge, batch_size, workers
    )
    _, warmup_seconds = score_loaded_model(
        model,
        warmup_loader,
        warmup_refs,
        device,
        crop_tokens,
        stride,
        parent_shape,
    )

    def run_full() -> tuple[dict[int, dict[str, np.ndarray]], dict[str, float]]:
        started = time.perf_counter()
        loader = build_loader(data_root, rows, full_refs, edge, batch_size, workers)
        accumulators, detector_seconds = score_loaded_model(
            model, loader, full_refs, device, crop_tokens, stride, parent_shape
        )
        fusion_started = time.perf_counter()
        raw = {
            index: fuse_views(
                accumulators[index], VIEWS, relative_ids, centers, scales
            )
            for index in evaluation_indices
        }
        parent_maps = {index: score_modes(value, kernel) for index, value in raw.items()}
        fusion_seconds = time.perf_counter() - fusion_started
        total_seconds = time.perf_counter() - started
        return parent_maps, {
            "total_seconds": total_seconds,
            "detector_seconds": detector_seconds,
            "selection_seconds": 0.0,
            "fusion_and_scoring_seconds": fusion_seconds,
        }

    def run_sequential() -> tuple[
        dict[int, dict[str, np.ndarray]], dict[int, np.ndarray], dict[str, float]
    ]:
        started = time.perf_counter()
        stage_loader = build_loader(data_root, rows, stage_refs, edge, batch_size, workers)
        stage_accumulators, stage_detector_seconds = score_loaded_model(
            model, stage_loader, stage_refs, device, crop_tokens, stride, parent_shape
        )

        selection_started = time.perf_counter()
        stage_raw = {
            index: fuse_views(
                stage_accumulators[index], stage_views, relative_ids, centers, scales
            )
            for index in evaluation_indices
        }
        stage_fields = {
            index: score_modes(raw, kernel)[selector_mode]
            for index, raw in stage_raw.items()
        }
        selected = {
            index: slot_statistics(stage_fields[index], slots, fraction) > thresholds
            for index in evaluation_indices
        }
        optional_refs = [
            CropRef(index, positions[slot][0], positions[slot][1], "x_shifted_grid")
            for index in evaluation_indices
            for slot in np.flatnonzero(selected[index])
        ]
        selection_seconds = time.perf_counter() - selection_started

        optional_detector_seconds = 0.0
        optional_accumulators: dict[int, dict[str, np.ndarray]] = {}
        if optional_refs:
            optional_loader = build_loader(
                data_root, rows, optional_refs, edge, batch_size, workers
            )
            optional_accumulators, optional_detector_seconds = score_loaded_model(
                model,
                optional_loader,
                optional_refs,
                device,
                crop_tokens,
                stride,
                parent_shape,
            )

        fusion_started = time.perf_counter()
        for index, optional in optional_accumulators.items():
            stage_accumulators[index]["view_x_shifted_grid"] = optional[
                "view_x_shifted_grid"
            ]
        raw = {
            index: fuse_views(
                stage_accumulators[index],
                VIEWS,
                relative_ids,
                centers,
                scales,
                mask_from_selection(parent_shape, slots, selected[index]),
            )
            for index in evaluation_indices
        }
        parent_maps = {index: score_modes(value, kernel) for index, value in raw.items()}
        fusion_seconds = time.perf_counter() - fusion_started
        total_seconds = time.perf_counter() - started
        return parent_maps, selected, {
            "total_seconds": total_seconds,
            "detector_seconds": stage_detector_seconds + optional_detector_seconds,
            "stage_one_detector_seconds": stage_detector_seconds,
            "optional_detector_seconds": optional_detector_seconds,
            "selection_seconds": selection_seconds,
            "fusion_and_scoring_seconds": fusion_seconds,
        }

    full_records: list[dict[str, float]] = []
    sequential_records: list[dict[str, float]] = []
    full_maps: dict[int, dict[str, np.ndarray]] = {}
    sequential_maps: dict[int, dict[str, np.ndarray]] = {}
    selected_by_parent: dict[int, np.ndarray] = {}
    order: list[str] = []
    for repeat in range(args.repeats):
        variants = ("full", "sequential") if repeat % 2 == 0 else ("sequential", "full")
        for variant in variants:
            order.append(variant)
            if variant == "full":
                full_maps, record = run_full()
                full_records.append(record)
            else:
                sequential_maps, selected_by_parent, record = run_sequential()
                sequential_records.append(record)

    coco = json.loads((data_root / str(base["data"]["coco"])).read_text(encoding="utf-8"))
    annotations, union_masks = annotation_masks(coco, rows, *parent_shape)
    calibration_full_maps = {
        index: score_modes(raw, kernel) for index, raw in calibration_full_raw.items()
    }
    calibration_sequential_maps = {
        index: score_modes(raw, kernel)
        for index, raw in calibration_sequential_raw.items()
    }
    full_metrics = evaluate(
        {**calibration_full_maps, **full_maps}, split, annotations, union_masks, base
    )
    sequential_metrics = evaluate(
        {**calibration_sequential_maps, **sequential_maps},
        split,
        annotations,
        union_masks,
        base,
    )
    compact = {
        "full_pcaf": compact_metrics(full_metrics, mode, len(split["evaluation_normals"])),
        "sequential_pcaf": compact_metrics(
            sequential_metrics, mode, len(split["evaluation_normals"])
        ),
    }

    replay_difference = None
    reference_hash = None
    if not args.smoke:
        reference_path = args.formal_reference or Path(
            f"runs/raw_fabrid_sequential_observation_k0/all_folds/{fold}.json"
        )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_hash = sha256_file(reference_path)
        replay_difference = max(
            max_metric_difference(
                compact[variant], reference["compact_primary_metrics"][variant]
            )
            for variant in compact
        )

    selected_counts = np.asarray(
        [int(selected_by_parent[index].sum()) for index in evaluation_indices],
        dtype=np.int64,
    )
    online_crops = int(geometry["stage_one_crops_per_parent"]) + selected_counts
    full_median = median_record(full_records)
    sequential_median = median_record(sequential_records)
    speedup = full_median["total_seconds"] / sequential_median["total_seconds"]
    latency_reduction = 1.0 - sequential_median["total_seconds"] / full_median["total_seconds"]

    result = {
        "schema_version": 1,
        "config": probe,
        "run": {
            "fold": fold,
            "smoke": args.smoke,
            "repeats": args.repeats,
            "timing_order": order,
            "data_root": str(data_root),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "batch_size": batch_size,
            "workers": workers,
            "checkpoint_sha256": sha256_file(checkpoint),
            "memory_vectors": int(bank.shape[0]),
            "memory_dimensions": int(bank.shape[1]),
            "script_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(args.config),
            "formal_reference_sha256": reference_hash,
        },
        "offline_source_normal_fitting": {
            "parents": len(calibration_indices),
            "crops": len(calibration_refs),
            "total_seconds": calibration_seconds,
            "detector_seconds": calibration_detector_seconds,
            "coordinate_bins": int(centers.size),
            "samples_per_bin_minimum": int(calibration_counts.min()),
        },
        "untimed_warmup": {
            "parents": len(evaluation_indices),
            "crops": len(warmup_refs),
            "seconds": warmup_seconds,
        },
        "online_crop_budget": {
            "parents": len(evaluation_indices),
            "full_crops_per_parent": int(geometry["full_crops_per_parent"]),
            "stage_one_crops_per_parent": int(geometry["stage_one_crops_per_parent"]),
            "selected_optional_crops_total": int(selected_counts.sum()),
            "selected_optional_crops_per_parent_mean": float(selected_counts.mean()),
            "sequential_crops_per_parent_mean": float(online_crops.mean()),
            "crop_reduction_vs_full": float(
                1.0 - online_crops.mean() / int(geometry["full_crops_per_parent"])
            ),
        },
        "online_timing_seconds": {
            "full_repeats": full_records,
            "sequential_repeats": sequential_records,
            "full_median": full_median,
            "sequential_median": sequential_median,
            "end_to_end_speedup": speedup,
            "end_to_end_latency_reduction": latency_reduction,
        },
        "compact_primary_metrics": compact,
        "formal_result_max_absolute_compact_difference": replay_difference,
        "validation": {
            "exact_formal_metric_replay": args.smoke
            or (replay_difference is not None and replay_difference <= 1e-8),
            "same_selected_count_as_formal": args.smoke
            or abs(
                float(online_crops.mean())
                - float(reference["counts"]["evaluation_online_crops_per_parent_mean"])
            )
            <= 1e-12,
        },
    }
    safe = json_safe(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "run": safe["run"],
        "online_crop_budget": safe["online_crop_budget"],
        "online_timing_seconds": safe["online_timing_seconds"],
        "compact_primary_metrics": safe["compact_primary_metrics"],
        "formal_result_max_absolute_compact_difference": safe[
            "formal_result_max_absolute_compact_difference"
        ],
        "validation": safe["validation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
