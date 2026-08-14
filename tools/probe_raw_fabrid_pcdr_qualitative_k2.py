"""Cache per-parent Hann and PCDR fields for deterministic case selection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from probe_raw_fabrid_anomalib_patchcore import evaluate, score_modes
from probe_raw_fabrid_dual_readout_fusion_k2 import ALARM, CANDIDATE, LOCALIZATION
from probe_raw_fabrid_geometric_fusion_baselines_k0 import geometric_fusion_fields
from probe_raw_fabrid_geometry_calibrated_fusion_k1 import geometry_calibrated_fields
from probe_raw_fabrid_grouped import annotation_masks, load_rows, split_p2
from probe_raw_fabrid_physical_field_k0 import (
    compact_metrics,
    finalize_fusions,
    json_safe,
    make_crop_refs,
    make_loader,
    score_patchcore,
    sha256_file,
)


SPATIAL_METRICS = (
    "pixel_average_precision",
    "instance_auc_all",
    "instance_auc_small",
    "instance_auc_elongated",
)
REPLAY_TOLERANCE = 1e-8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    fold = str(args.fold)
    confirmation = list(config["confirmation"]["folds"])
    if fold not in confirmation:
        raise ValueError(f"Qualitative cache is restricted to {confirmation}")
    base = json.loads(Path(config["base_config"]).read_text(encoding="utf-8"))
    data_root = args.data_root or Path(base["data"]["root"])
    rows = load_rows(data_root, base["data"])
    by_roll: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_roll[str(row["roll_id"])].append(index)
    split = split_p2(fold, by_roll, rows)

    geometry = config["crop_observations"]
    edge = int(geometry["physical_crop_edge"])
    shift = int(geometry["phase_shift_pixels"])
    stride = int(geometry["output_stride_pixels"])
    height = int(base["data"]["image_height"])
    width = int(base["data"]["image_width"])
    parent_shape = (height // stride, width // stride)
    output_edge = edge // stride
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
    accumulators, predict_seconds, model_facts = score_patchcore(
        base,
        checkpoint,
        loader,
        refs,
        device,
        output_edge,
        stride,
        parent_shape,
    )
    pcaf, _, context_facts = finalize_fusions(
        accumulators,
        scored_indices,
        [ALARM],
        list(split["calibration_normals"]),
        output_edge,
        shift // stride,
    )
    geometric, geometric_facts = geometric_fusion_fields(
        accumulators, scored_indices, output_edge, shift // stride
    )
    calibrated, calibrated_facts = geometry_calibrated_fields(
        accumulators,
        list(split["calibration_normals"]),
        scored_indices,
        output_edge,
        shift // stride,
        pcaf[ALARM],
    )
    fields = {
        "hann": geometric["hann"],
        LOCALIZATION: calibrated[LOCALIZATION],
    }
    maps = {
        name: {
            index: score_modes(raw, int(base["scores"]["local_mean_kernel_tokens"]))
            for index, raw in parent_fields.items()
        }
        for name, parent_fields in fields.items()
    }
    coco = json.loads(
        (data_root / str(base["data"]["coco"])).read_text(encoding="utf-8")
    )
    annotations, union_masks = annotation_masks(coco, rows, *parent_shape)
    metrics = {
        name: evaluate(parent_maps, split, annotations, union_masks, base)
        for name, parent_maps in maps.items()
    }
    mode = str(config["confirmation"]["primary_score_mode"])
    compact = {
        name: compact_metrics(current, mode, len(split["evaluation_normals"]))
        for name, current in metrics.items()
    }
    reference_path = args.reference or Path(
        str(config["reference_result_pattern"]).format(fold=fold)
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = reference["compact_primary_metrics"]
    replay_errors: dict[str, float] = {}
    for metric in SPATIAL_METRICS:
        replay_errors[f"hann/{metric}"] = abs(
            float(compact["hann"][metric]) - float(expected["hann"][metric])
        )
        replay_errors[f"pcdr/{metric}"] = abs(
            float(compact[LOCALIZATION][metric]) - float(expected[CANDIDATE][metric])
        )
    maximum_error = max(replay_errors.values())
    if maximum_error > REPLAY_TOLERANCE:
        raise RuntimeError(
            f"Qualitative-cache replay drift for {fold}: {maximum_error:.3g}"
        )

    anomaly_indices = np.asarray(split["anomalies"], dtype=np.int64)
    hann_fields = np.stack(
        [maps["hann"][int(index)][mode] for index in anomaly_indices]
    ).astype(np.float32)
    pcdr_fields = np.stack(
        [maps[LOCALIZATION][int(index)][mode] for index in anomaly_indices]
    ).astype(np.float32)
    masks = np.stack([union_masks[int(index)] for index in anomaly_indices]).astype(np.uint8)
    hann_ap = np.asarray(
        [
            average_precision_score(mask.reshape(-1), field.reshape(-1))
            for mask, field in zip(masks, hann_fields, strict=True)
        ],
        dtype=np.float64,
    )
    pcdr_ap = np.asarray(
        [
            average_precision_score(mask.reshape(-1), field.reshape(-1))
            for mask, field in zip(masks, pcdr_fields, strict=True)
        ],
        dtype=np.float64,
    )
    filenames = np.asarray(
        [str(rows[int(index)]["filename"]) for index in anomaly_indices]
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        fold=np.asarray(fold),
        indices=anomaly_indices,
        filenames=filenames,
        masks=masks,
        hann_fields=hann_fields,
        pcdr_fields=pcdr_fields,
        hann_parent_pixel_ap=hann_ap,
        pcdr_parent_pixel_ap=pcdr_ap,
        delta_parent_pixel_ap=pcdr_ap - hann_ap,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "Deterministic PCDR qualitative-case cache",
        "run": {
            "fold": fold,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "predict_seconds": predict_seconds,
            "script_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(args.config),
            "checkpoint_sha256": sha256_file(checkpoint),
            "cache_sha256": sha256_file(args.output_npz),
            **model_facts,
        },
        "counts": {
            "anomalous_parents": int(len(anomaly_indices)),
            "crops_per_parent": len(refs) // len(scored_indices),
        },
        "replay": {
            "reference": str(reference_path),
            "maximum_absolute_spatial_metric_error": maximum_error,
            "tolerance": REPLAY_TOLERANCE,
            "per_metric_errors": replay_errors,
            "passed": True,
        },
        "per_parent_pixel_ap_delta": {
            str(index): float(value)
            for index, value in zip(anomaly_indices, pcdr_ap - hann_ap, strict=True)
        },
        "geometric_fusion": geometric_facts,
        "geometry_calibration": calibrated_facts,
        "context_response_calibration": context_facts,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "fold": fold,
        "anomalous_parents": len(anomaly_indices),
        "delta_min": float((pcdr_ap - hann_ap).min()),
        "delta_median": float(np.median(pcdr_ap - hann_ap)),
        "delta_max": float((pcdr_ap - hann_ap).max()),
        "replay_max_error": maximum_error,
    }, indent=2))


if __name__ == "__main__":
    main()
