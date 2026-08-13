"""Evaluate a frozen dual-readout tiled-fabric inspection rule.

The localization readout is the Rollo4A-developed Gaussian-weighted,
crop-coordinate-bias-corrected field.  Parent alarming remains the frozen PCAF
readout.  The two outputs reuse identical detector crops but estimate different
quantities: a spatial ranking and a parent-level alarm.  Rollo4A is explicitly
development data; only the other six rolls constitute confirmation.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from probe_raw_fabrid_anomalib_patchcore import evaluate, score_modes
from probe_raw_fabrid_geometric_fusion_baselines_k0 import (
    geometric_fusion_fields,
    verify_frozen_reference,
)
from probe_raw_fabrid_geometry_calibrated_fusion_k1 import (
    geometry_calibrated_fields,
)
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


CANDIDATE = "dual_gaussian_bias_localization_pcaf_alarm"
LOCALIZATION = "gaussian_bias_equal_mean"
ALARM = "context_bias_weighted_mean"
REFERENCES = ("gaussian", "hann", ALARM)
PARENT_SCALAR_KEYS = ("auroc", "average_precision")
SOURCE_IMAGE_KEYS = (
    "image_threshold",
    "normal_parent_fpr",
    "defect_parent_recall_at_1pct_fpr",
)
ORACLE_IMAGE_KEYS = (
    "image_threshold",
    "defect_parent_recall_at_1pct_fpr",
)


def compose_dual_readout_metrics(
    localization_metrics: dict[str, Any],
    alarm_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Combine spatial metrics with an independently specified parent alarm.

    Pixel AP, pixel thresholds, instance records, and maximum scores remain
    localization-derived.  Parent AUROC/AP, image thresholds, image scores,
    normal FPR, and defect-parent recall come only from the alarm readout.
    """
    if set(localization_metrics) != set(alarm_metrics):
        raise ValueError("Localization and alarm score modes differ")
    combined: dict[str, Any] = {}
    for mode in localization_metrics:
        localization = localization_metrics[mode]
        alarm = alarm_metrics[mode]
        current = copy.deepcopy(localization)
        for key in PARENT_SCALAR_KEYS:
            current[key] = alarm[key]
        for key in SOURCE_IMAGE_KEYS:
            current["source_calibrated"][key] = alarm["source_calibrated"][key]
        for key in ORACLE_IMAGE_KEYS:
            current["diagnostic_target_oracle"][key] = (
                alarm["diagnostic_target_oracle"][key]
            )
        if set(localization["parent_scores"]) != set(alarm["parent_scores"]):
            raise ValueError(f"Parent-score identities differ for mode {mode}")
        current["parent_scores"] = {
            index: {
                "image_score": alarm["parent_scores"][index]["image_score"],
                "maximum_score": localization["parent_scores"][index][
                    "maximum_score"
                ],
            }
            for index in sorted(localization["parent_scores"], key=int)
        }
        current["dual_readout_contract"] = {
            "localization": LOCALIZATION,
            "parent_alarm": ALARM,
            "parent_metrics_copied_from_alarm": list(PARENT_SCALAR_KEYS),
            "source_image_fields_copied_from_alarm": list(SOURCE_IMAGE_KEYS),
            "oracle_image_fields_copied_from_alarm": list(ORACLE_IMAGE_KEYS),
            "pixel_and_instance_fields_copied_from_localization": True,
        }
        combined[mode] = current
    return combined


def verify_dual_readout_metrics(
    candidate: dict[str, Any],
    localization: dict[str, Any],
    alarm: dict[str, Any],
) -> dict[str, Any]:
    """Fail unless each metric is sourced from its declared readout exactly."""
    maximum_parent_error = 0.0
    maximum_localization_error = 0.0
    parent_score_error = 0.0
    maximum_score_error = 0.0
    for mode in candidate:
        current = candidate[mode]
        local = localization[mode]
        alarm_mode = alarm[mode]
        for key in PARENT_SCALAR_KEYS:
            maximum_parent_error = max(
                maximum_parent_error,
                abs(float(current[key]) - float(alarm_mode[key])),
            )
        for key in SOURCE_IMAGE_KEYS:
            maximum_parent_error = max(
                maximum_parent_error,
                abs(
                    float(current["source_calibrated"][key])
                    - float(alarm_mode["source_calibrated"][key])
                ),
            )
        for key in ORACLE_IMAGE_KEYS:
            maximum_parent_error = max(
                maximum_parent_error,
                abs(
                    float(current["diagnostic_target_oracle"][key])
                    - float(alarm_mode["diagnostic_target_oracle"][key])
                ),
            )
        for key in ("pixel_average_precision",):
            maximum_localization_error = max(
                maximum_localization_error,
                abs(float(current[key]) - float(local[key])),
            )
        maximum_localization_error = max(
            maximum_localization_error,
            abs(
                float(current["source_calibrated"]["pixel_threshold"])
                - float(local["source_calibrated"]["pixel_threshold"])
            ),
            abs(
                float(current["diagnostic_target_oracle"]["pixel_threshold"])
                - float(local["diagnostic_target_oracle"]["pixel_threshold"])
            ),
        )
        if current["instance_recall"] != local["instance_recall"]:
            raise RuntimeError(f"Instance summary drift in mode {mode}")
        if current["instance_records"] != local["instance_records"]:
            raise RuntimeError(f"Instance-record drift in mode {mode}")
        for index, scores in current["parent_scores"].items():
            parent_score_error = max(
                parent_score_error,
                abs(
                    float(scores["image_score"])
                    - float(alarm_mode["parent_scores"][index]["image_score"])
                ),
            )
            maximum_score_error = max(
                maximum_score_error,
                abs(
                    float(scores["maximum_score"])
                    - float(local["parent_scores"][index]["maximum_score"])
                ),
            )
    errors = {
        "maximum_parent_metric_error_from_pcaf": maximum_parent_error,
        "maximum_localization_metric_error_from_gaussian_bias": (
            maximum_localization_error
        ),
        "maximum_parent_image_score_error_from_pcaf": parent_score_error,
        "maximum_parent_maximum_score_error_from_gaussian_bias": (
            maximum_score_error
        ),
    }
    if any(value != 0.0 for value in errors.values()):
        raise RuntimeError(f"Dual-readout identity check failed: {errors}")
    return {
        **errors,
        "exact_pcaf_parent_readout": True,
        "exact_gaussian_bias_localization_readout": True,
    }


def development_reproduction(
    compact: dict[str, dict[str, float | int]],
    composition: dict[str, Any],
    config: dict[str, Any],
    fold: str,
) -> dict[str, Any] | None:
    """Reproduce the disclosed Rollo4A development fact; never select a model."""
    specification = config["development_record"]
    if fold != str(specification["fold"]):
        return None
    candidate = compact[CANDIDATE]
    best_pixel = max(
        float(compact[name]["pixel_average_precision"]) for name in REFERENCES
    )
    best_auc = max(float(compact[name]["instance_auc_all"]) for name in REFERENCES)
    checks = {
        "pixel_ap_reproduced": (
            float(candidate["pixel_average_precision"]) - best_pixel
            >= float(
                specification[
                    "minimum_reproduction_pixel_ap_gain_over_best_of_gaussian_hann_pcaf"
                ]
            )
        ),
        "instance_auc_reproduced": (
            float(candidate["instance_auc_all"])
            >= best_auc
            - float(
                specification[
                    "maximum_reproduction_instance_auc_loss_from_best_of_gaussian_hann_pcaf"
                ]
            )
        ),
        "pcaf_parent_identity": bool(composition["exact_pcaf_parent_readout"]),
    }
    return {
        "status": specification["status"],
        "checks": checks,
        "passed": all(checks.values()),
        "pixel_ap_gain_over_best_reference": (
            float(candidate["pixel_average_precision"]) - best_pixel
        ),
        "instance_auc_delta_from_best_reference": (
            float(candidate["instance_auc_all"]) - best_auc
        ),
        "scope": specification["interpretation"],
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
        split["calibration_normals"]
        + split["evaluation_normals"]
        + split["anomalies"]
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
        raise RuntimeError(
            f"Expected {geometry['crops_per_parent']} crops per parent, obtained "
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
    accumulators, predict_seconds, model_facts = score_patchcore(
        base, checkpoint, loader, refs, device, output_edge, stride, parent_shape
    )

    fusion_started = time.perf_counter()
    raw_variants, _, context_facts = finalize_fusions(
        accumulators,
        scored_indices,
        ["mean", ALARM],
        list(split["calibration_normals"]),
        output_edge,
        shift // stride,
    )
    geometric, geometric_facts = geometric_fusion_fields(
        accumulators, scored_indices, output_edge, shift // stride
    )
    raw_variants.update(geometric)
    calibrated, calibrated_facts = geometry_calibrated_fields(
        accumulators,
        list(split["calibration_normals"]),
        scored_indices,
        output_edge,
        shift // stride,
        raw_variants[ALARM],
    )
    raw_variants[LOCALIZATION] = calibrated[LOCALIZATION]
    fusion_seconds = time.perf_counter() - fusion_started

    maps = {
        variant: {
            index: score_modes(raw, int(base["scores"]["local_mean_kernel_tokens"]))
            for index, raw in parent_maps.items()
        }
        for variant, parent_maps in raw_variants.items()
    }
    coco = json.loads(
        (data_root / str(base["data"]["coco"])).read_text(encoding="utf-8")
    )
    annotations, union_masks = annotation_masks(coco, rows, *parent_shape)
    evaluation_started = time.perf_counter()
    metrics = {
        variant: evaluate(parent_maps, split, annotations, union_masks, base)
        for variant, parent_maps in maps.items()
    }
    metrics[CANDIDATE] = compose_dual_readout_metrics(
        metrics[LOCALIZATION], metrics[ALARM]
    )
    evaluation_seconds = time.perf_counter() - evaluation_started

    mode = str(config["confirmation"]["primary_score_mode"])
    compact = {
        variant: compact_metrics(current, mode, len(split["evaluation_normals"]))
        for variant, current in metrics.items()
    }
    composition = verify_dual_readout_metrics(
        metrics[CANDIDATE], metrics[LOCALIZATION], metrics[ALARM]
    )
    reference_path = args.reference or Path(
        str(config["reference_result_pattern"]).format(fold=fold)
    )
    replay_check = verify_frozen_reference(compact, reference_path)
    development = development_reproduction(compact, composition, config, fold)

    result = {
        "schema_version": 1,
        "config": config,
        "run": {
            "fold": fold,
            "data_root": str(data_root),
            "device": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
            ),
            "torch": torch.__version__,
            "predict_seconds": predict_seconds,
            "fusion_seconds": fusion_seconds,
            "evaluation_seconds": evaluation_seconds,
            "script_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(args.config),
            "checkpoint_sha256": sha256_file(checkpoint),
            "dependency_sha256": {
                "base_config": sha256_file(Path(str(config["base_config"]))),
                "geometric_audit_config": sha256_file(
                    Path(str(config["parent_geometric_audit_config"]))
                ),
                "geometry_calibration_config": sha256_file(
                    Path(str(config["parent_geometry_calibration_config"]))
                ),
                "physical_field_implementation": sha256_file(
                    Path(__file__).with_name("probe_raw_fabrid_physical_field_k0.py")
                ),
                "geometric_fusion_implementation": sha256_file(
                    Path(__file__).with_name(
                        "probe_raw_fabrid_geometric_fusion_baselines_k0.py"
                    )
                ),
                "geometry_calibration_implementation": sha256_file(
                    Path(__file__).with_name(
                        "probe_raw_fabrid_geometry_calibrated_fusion_k1.py"
                    )
                ),
                "evaluation_implementation": sha256_file(
                    Path(__file__).with_name(
                        "probe_raw_fabrid_anomalib_patchcore.py"
                    )
                ),
            },
            **model_facts,
        },
        "counts": {
            **{key: len(value) for key, value in split.items()},
            "scored_parents": len(scored_indices),
            "crops_per_parent": crops_per_parent,
            "total_crops": len(refs),
        },
        "split_indices": split,
        "dual_readout_identity": composition,
        "development_reproduction": development,
        "geometric_fusion": geometric_facts,
        "geometry_calibration": calibrated_facts,
        "context_response_calibration": context_facts,
        "frozen_replay_check": replay_check,
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
        "dual_readout_identity": safe["dual_readout_identity"],
        "development_reproduction": safe["development_reproduction"],
        "frozen_replay_check": safe["frozen_replay_check"],
        "compact_primary_metrics": safe["compact_primary_metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
