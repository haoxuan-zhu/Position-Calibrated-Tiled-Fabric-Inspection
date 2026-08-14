"""Audit whether PatchCore map rendering creates the PCDR coordinate effect.

One frozen forward pass produces native patch distances.  Three deterministic
renderings of those same distances are then evaluated with identical crops,
normal-only calibration, fusion rules, masks, and metrics.  This isolates the
raw patch field, resize round-trip, and standard Anomalib smoothing paths
without retraining or tripling detector inference.
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

from patchcore_field_variants import ANOMALIB_DEFAULT, FIELD_VARIANTS, derive_patchcore_fields
from probe_raw_fabrid_anomalib_patchcore import evaluate, score_modes
from probe_raw_fabrid_dual_readout_fusion_k2 import (
    ALARM,
    CANDIDATE,
    LOCALIZATION,
    compose_dual_readout_metrics,
    verify_dual_readout_metrics,
)
from probe_raw_fabrid_geometric_fusion_baselines_k0 import (
    geometric_fusion_fields,
    relative_coordinate_ids,
)
from probe_raw_fabrid_geometry_calibrated_fusion_k1 import (
    fit_coordinate_calibration,
    geometry_calibrated_fields,
)
from probe_raw_fabrid_grouped import annotation_masks, load_rows, split_p2
from probe_raw_fabrid_physical_field_k0 import (
    add_crop,
    compact_metrics,
    finalize_fusions,
    json_safe,
    make_crop_refs,
    make_loader,
    sha256_file,
)
from probe_raw_patchcore_memory_quantization import embed, load_model as load_patchcore


REFERENCE_VARIANTS = (
    "mean",
    "gaussian",
    "hann",
    ALARM,
    LOCALIZATION,
    CANDIDATE,
)
REFERENCE_TOLERANCE = 1e-8


@torch.inference_mode()
def score_field_variants(
    base: dict[str, Any],
    checkpoint: Path,
    loader: torch.utils.data.DataLoader,
    refs: list[Any],
    device: torch.device,
    output_edge: int,
    stride: int,
    parent_shape: tuple[int, int],
) -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], float, dict[str, Any]]:
    """Score every crop once and register all three controlled fields."""
    model, bank = load_patchcore(base, checkpoint, device)
    accumulators: dict[str, dict[int, dict[str, np.ndarray]]] = {
        name: {} for name in FIELD_VARIANTS
    }
    processed = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in loader:
        images, parents, ys, xs, views = batch
        images = images.to(device, non_blocking=True)
        if model.pre_processor is not None:
            images = model.pre_processor(images)
        embedding, grid_shape, image_size = embed(model, images)
        patch_scores, _ = model.model.nearest_neighbors(embedding, n_neighbors=1)
        patch_scores = patch_scores.reshape(images.shape[0], 1, *grid_shape)
        field_maps = derive_patchcore_fields(
            patch_scores,
            image_size=image_size,
            output_size=(output_edge, output_edge),
            anomaly_map_generator=model.model.anomaly_map_generator,
        )
        for name, maps in field_maps.items():
            arrays = maps[:, 0].cpu().numpy().astype(np.float32, copy=False)
            for position, crop_map in enumerate(arrays):
                add_crop(
                    accumulators[name],
                    int(parents[position]),
                    int(ys[position]),
                    int(xs[position]),
                    str(views[position]),
                    crop_map,
                    stride,
                    parent_shape,
                )
        processed += images.shape[0]
        if processed == images.shape[0] or processed % 2000 < images.shape[0]:
            print(f"field-control crops {processed}/{len(refs)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return accumulators, time.perf_counter() - started, {
        "checkpoint_sha256": sha256_file(checkpoint),
        "memory_vectors": int(bank.shape[0]),
        "memory_dimensions": int(bank.shape[1]),
        "single_patchcore_forward_per_crop": True,
    }


def evaluate_field(
    accumulators: dict[int, dict[str, np.ndarray]],
    scored_indices: list[int],
    calibration_indices: list[int],
    split: dict[str, list[int]],
    annotations: dict[int, Any],
    union_masks: dict[int, np.ndarray],
    base: dict[str, Any],
    output_edge: int,
    shift_tokens: int,
) -> dict[str, Any]:
    """Apply the frozen Hann, PCAF, and PCDR logic to one score field."""
    variants, _, context_facts = finalize_fusions(
        accumulators,
        scored_indices,
        ["mean", ALARM],
        calibration_indices,
        output_edge,
        shift_tokens,
    )
    geometric, geometric_facts = geometric_fusion_fields(
        accumulators, scored_indices, output_edge, shift_tokens
    )
    variants.update(geometric)
    calibrated, calibrated_facts = geometry_calibrated_fields(
        accumulators,
        calibration_indices,
        scored_indices,
        output_edge,
        shift_tokens,
        variants[ALARM],
    )
    variants[LOCALIZATION] = calibrated[LOCALIZATION]

    relative_ids = relative_coordinate_ids(
        accumulators[scored_indices[0]]["count"].shape,
        output_edge,
        shift_tokens,
    )
    centers, dispersions, sample_counts = fit_coordinate_calibration(
        accumulators,
        calibration_indices,
        relative_ids,
        output_edge * output_edge,
    )

    scored_maps = {
        name: {
            index: score_modes(raw, int(base["scores"]["local_mean_kernel_tokens"]))
            for index, raw in parent_maps.items()
        }
        for name, parent_maps in variants.items()
    }
    metrics = {
        name: evaluate(parent_maps, split, annotations, union_masks, base)
        for name, parent_maps in scored_maps.items()
    }
    metrics[CANDIDATE] = compose_dual_readout_metrics(
        metrics[LOCALIZATION], metrics[ALARM]
    )
    identity = verify_dual_readout_metrics(
        metrics[CANDIDATE], metrics[LOCALIZATION], metrics[ALARM]
    )
    mode = "robust_z"
    compact = {
        name: compact_metrics(current, mode, len(split["evaluation_normals"]))
        for name, current in metrics.items()
    }
    return {
        "compact_primary_metrics": {
            name: compact[name] for name in REFERENCE_VARIANTS
        },
        "pcdr_minus_hann": {
            "pixel_average_precision": float(
                compact[CANDIDATE]["pixel_average_precision"]
                - compact["hann"]["pixel_average_precision"]
            ),
            "instance_auc_all": float(
                compact[CANDIDATE]["instance_auc_all"]
                - compact["hann"]["instance_auc_all"]
            ),
        },
        "coordinate_calibration": {
            "center_map": centers.reshape(output_edge, output_edge).tolist(),
            "dispersion_map": dispersions.reshape(output_edge, output_edge).tolist(),
            "samples_per_bin_minimum": int(sample_counts.min()),
            "samples_per_bin_median": float(np.median(sample_counts)),
        },
        "dual_readout_identity": identity,
        "geometric_fusion": geometric_facts,
        "geometry_calibration": calibrated_facts,
        "context_response_calibration": context_facts,
    }


def verify_standard_replay(
    field_result: dict[str, Any], reference_path: Path
) -> dict[str, Any]:
    """Require the standard branch to reproduce the frozen PCDR record."""
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = reference["compact_primary_metrics"]
    observed = field_result["compact_primary_metrics"]
    maximum = 0.0
    errors: dict[str, dict[str, float]] = {}
    for variant in REFERENCE_VARIANTS:
        errors[variant] = {}
        for metric, value in observed[variant].items():
            difference = abs(float(value) - float(expected[variant][metric]))
            errors[variant][metric] = difference
            maximum = max(maximum, difference)
    if maximum > REFERENCE_TOLERANCE:
        raise RuntimeError(
            "standard Anomalib branch did not reproduce the frozen PCDR record: "
            f"maximum error {maximum:.3g}"
        )
    return {
        "reference": str(reference_path),
        "maximum_absolute_metric_error": maximum,
        "tolerance": REFERENCE_TOLERANCE,
        "passed": True,
        "per_metric_errors": errors,
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
    base = json.loads(Path(config["base_config"]).read_text(encoding="utf-8"))
    pcdr = json.loads(Path(config["pcdr_config"]).read_text(encoding="utf-8"))
    fold = str(args.fold)
    if fold not in config["folds"]:
        raise ValueError(f"Unknown fold {fold}; expected {config['folds']}")
    data_root = args.data_root or Path(base["data"]["root"])
    rows = load_rows(data_root, base["data"])
    by_roll: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_roll[str(row["roll_id"])].append(index)
    split = split_p2(fold, by_roll, rows)

    geometry = pcdr["crop_observations"]
    edge = int(geometry["physical_crop_edge"])
    shift = int(geometry["phase_shift_pixels"])
    stride = int(geometry["output_stride_pixels"])
    height = int(base["data"]["image_height"])
    width = int(base["data"]["image_width"])
    scored_indices = sorted(
        set(
            split["calibration_normals"]
            + split["evaluation_normals"]
            + split["anomalies"]
        )
    )
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
        raise RuntimeError(
            f"Expected {geometry['crops_per_parent']} crops per parent, got "
            f"{crops_per_parent}"
        )

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
    accumulators, predict_seconds, model_facts = score_field_variants(
        base,
        checkpoint,
        loader,
        refs,
        device,
        output_edge,
        stride,
        parent_shape,
    )

    coco = json.loads(
        (data_root / str(base["data"]["coco"])).read_text(encoding="utf-8")
    )
    annotations, union_masks = annotation_masks(coco, rows, *parent_shape)
    evaluation_started = time.perf_counter()
    field_results = {
        name: evaluate_field(
            current,
            scored_indices,
            list(split["calibration_normals"]),
            split,
            annotations,
            union_masks,
            base,
            output_edge,
            shift // stride,
        )
        for name, current in accumulators.items()
    }
    evaluation_seconds = time.perf_counter() - evaluation_started
    reference_path = args.reference or Path(
        str(config["reference_result_pattern"]).format(fold=fold)
    )
    replay = verify_standard_replay(field_results[ANOMALIB_DEFAULT], reference_path)

    result = {
        "schema_version": 1,
        "purpose": "PatchCore patch-score rendering control for PCDR",
        "config": config,
        "run": {
            "fold": fold,
            "data_root": str(data_root),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "predict_seconds": predict_seconds,
            "evaluation_seconds": evaluation_seconds,
            "script_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(args.config),
            "field_helper_sha256": sha256_file(
                Path(__file__).with_name("patchcore_field_variants.py")
            ),
            **model_facts,
        },
        "counts": {
            **{key: len(value) for key, value in split.items()},
            "scored_parents": len(scored_indices),
            "crops_per_parent": crops_per_parent,
            "total_crops": len(refs),
        },
        "split_indices": split,
        "standard_replay": replay,
        "field_results": field_results,
    }
    safe = json_safe(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run": safe["run"],
                "counts": safe["counts"],
                "standard_replay": safe["standard_replay"],
                "pcdr_minus_hann": {
                    name: field_results[name]["pcdr_minus_hann"]
                    for name in FIELD_VARIANTS
                },
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

