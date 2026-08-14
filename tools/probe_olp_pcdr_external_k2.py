"""Evaluate complete PCDR on frozen OLP scene-grouped PatchCore folds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from probe_raw_fabrid_dual_readout_fusion_k2 import ALARM, LOCALIZATION
from probe_raw_fabrid_geometric_fusion_baselines_k0 import geometric_fusion_fields
from probe_raw_fabrid_geometry_calibrated_fusion_k1 import geometry_calibrated_fields
from probe_raw_fabrid_physical_field_k0 import finalize_fusions, make_crop_refs
from probe_raw_patchcore_memory_quantization import load_model as load_patchcore
from run_olp_pcaf_external import (
    evaluate_variants,
    json_safe,
    load_records,
    score_crops,
    scoring_loader,
    sha256_file,
)


HANN_PCAF = "hann_pcaf_dual"
PCDR = "pcdr"
ALARM_KEYS = (
    "parent_average_precision",
    "parent_roc_auc",
    "source_calibrated_threshold",
    "evaluation_normal_false_positives",
    "evaluation_normal_fpr",
    "defect_parent_recall",
)
LOCALIZATION_KEYS = (
    "pixel_average_precision",
    "pixel_roc_auc",
    "pixel_positive_cells",
    "pixel_evaluated_cells",
)
REPLAY_VARIANTS = ("mean", ALARM)
REPLAY_EXACT_METRICS = {
    "evaluation_normal_false_positives",
    "evaluation_normal_fpr",
    "defect_parent_recall",
    "pixel_positive_cells",
    "pixel_evaluated_cells",
}


def compose_dual_metrics(
    localization: dict[str, Any], alarm: dict[str, Any]
) -> dict[str, Any]:
    """Combine OLP pixel metrics with the frozen PCAF alarm metrics."""
    combined = {key: localization[key] for key in LOCALIZATION_KEYS}
    combined.update({key: alarm[key] for key in ALARM_KEYS})
    return combined


def verify_dual_metrics(
    combined: dict[str, Any], localization: dict[str, Any], alarm: dict[str, Any]
) -> dict[str, Any]:
    local_error = max(
        abs(float(combined[key]) - float(localization[key]))
        for key in LOCALIZATION_KEYS
    )
    alarm_error = max(
        abs(float(combined[key]) - float(alarm[key])) for key in ALARM_KEYS
    )
    if local_error != 0.0 or alarm_error != 0.0:
        raise RuntimeError(
            f"OLP dual-readout identity failed: local={local_error}, "
            f"alarm={alarm_error}"
        )
    return {
        "maximum_localization_metric_error": local_error,
        "maximum_alarm_metric_error": alarm_error,
        "passed": True,
    }


def verify_original_replay(
    observed: dict[str, dict[str, Any]], reference_path: Path
) -> dict[str, Any]:
    """Audit checkpoint replay against results scored from the in-memory model."""
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = reference["metrics"]
    maximum = 0.0
    exact_mismatches: list[tuple[str, str, float, float]] = []
    errors: dict[str, dict[str, float]] = {}
    for variant in REPLAY_VARIANTS:
        errors[variant] = {}
        for metric, value in observed[variant].items():
            observed_value = float(value)
            expected_value = float(expected[variant][metric])
            difference = abs(observed_value - expected_value)
            errors[variant][metric] = difference
            if metric in REPLAY_EXACT_METRICS and difference != 0.0:
                exact_mismatches.append(
                    (variant, metric, observed_value, expected_value)
                )
            maximum = max(maximum, difference)
    if exact_mismatches:
        details = "; ".join(
            f"{variant}.{metric}: observed={observed_value:.12g}, "
            f"expected={expected_value:.12g}"
            for variant, metric, observed_value, expected_value in exact_mismatches
        )
        raise RuntimeError(
            "OLP legacy replay changed an exact count/rate: " + details
        )
    return {
        "reference": str(reference_path),
        "maximum_absolute_metric_error": maximum,
        "continuous_metrics_are_diagnostic_only": True,
        "exact_metrics": sorted(REPLAY_EXACT_METRICS),
        "passed": True,
        "per_metric_errors": errors,
    }


def verify_source_identity(
    records: list[Any],
    split: dict[str, list[int]],
    checkpoint_sha256: str,
    reference_path: Path,
) -> dict[str, Any]:
    """Require the checkpoint and every frozen split path to match exactly."""
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected_hash = str(reference["run"]["checkpoint_sha256"])
    observed_paths = {
        name: [records[index].relative_path for index in indices]
        for name, indices in split.items()
    }
    expected_paths = reference["split_paths"]
    observed_counts = {name: len(indices) for name, indices in split.items()}
    expected_counts = {name: int(value) for name, value in reference["counts"].items()}
    if checkpoint_sha256 != expected_hash:
        raise RuntimeError("OLP checkpoint SHA-256 differs from the frozen result")
    if observed_paths != expected_paths:
        raise RuntimeError("OLP split paths differ from the frozen result")
    if observed_counts != expected_counts:
        raise RuntimeError("OLP split counts differ from the frozen result")
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_matches": True,
        "split_paths_match": True,
        "split_counts_match": True,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--textile-id", required=True, type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    textile_id = int(args.textile_id)
    if textile_id not in config["eligible_textiles"]:
        raise ValueError(
            f"Textile {textile_id} is not in {config['eligible_textiles']}"
        )
    base_path = Path(config["base_config"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if args.data_root is not None:
        base["data"]["root"] = str(args.data_root)
    if args.metadata_root is not None:
        base["data"]["metadata_root"] = str(args.metadata_root)
    if args.audit is not None:
        base["data"]["audit"] = str(args.audit)

    records, split = load_records(base, textile_id)
    if args.smoke:
        split = {
            "training_normals": split["training_normals"][:1],
            "calibration_normals": split["calibration_normals"][:1],
            "evaluation_normals": split["evaluation_normals"][:1],
            "anomalies": split["anomalies"][:1],
        }
    edge = int(base["tiling"]["crop_edge"])
    shift = int(base["tiling"]["phase_shift_pixels"])
    stride = int(base["tiling"]["output_stride_pixels"])
    height = int(base["data"]["padded_height"])
    width = int(base["data"]["padded_width"])
    scored_indices = sorted(
        split["calibration_normals"]
        + split["evaluation_normals"]
        + split["anomalies"]
    )
    refs = make_crop_refs(
        scored_indices,
        height,
        width,
        edge,
        shift,
        list(base["tiling"]["views"]),
    )
    workers = 0 if args.smoke else int(base["model"]["num_workers"])
    loader = scoring_loader(records, refs, base, workers)

    seed = int(config["seed"]) + textile_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False

    checkpoint = args.checkpoint or Path(
        str(config["checkpoint_pattern"]).format(textile_id=textile_id)
    )
    reference_path = args.reference or Path(
        str(config["reference_result_pattern"]).format(textile_id=textile_id)
    )
    checkpoint_sha256 = sha256_file(checkpoint)
    source_identity = (
        {"passed": False, "skipped_for_smoke": True}
        if args.smoke
        else verify_source_identity(
            records, split, checkpoint_sha256, reference_path
        )
    )
    model, bank = load_patchcore(base, checkpoint, device)
    accumulators, predict_seconds = score_crops(model, loader, refs, device, base)

    fusion_started = time.perf_counter()
    fields, _, context_facts = finalize_fusions(
        accumulators,
        scored_indices,
        ["mean", ALARM],
        list(split["calibration_normals"]),
        edge // stride,
        shift // stride,
    )
    geometric, geometric_facts = geometric_fusion_fields(
        accumulators, scored_indices, edge // stride, shift // stride
    )
    fields.update(geometric)
    calibrated, calibrated_facts = geometry_calibrated_fields(
        accumulators,
        list(split["calibration_normals"]),
        scored_indices,
        edge // stride,
        shift // stride,
        fields[ALARM],
    )
    fields[LOCALIZATION] = calibrated[LOCALIZATION]
    metrics = evaluate_variants(fields, records, split, base)
    metrics[HANN_PCAF] = compose_dual_metrics(metrics["hann"], metrics[ALARM])
    metrics[PCDR] = compose_dual_metrics(metrics[LOCALIZATION], metrics[ALARM])
    identities = {
        HANN_PCAF: verify_dual_metrics(
            metrics[HANN_PCAF], metrics["hann"], metrics[ALARM]
        ),
        PCDR: verify_dual_metrics(metrics[PCDR], metrics[LOCALIZATION], metrics[ALARM]),
    }
    fusion_seconds = time.perf_counter() - fusion_started

    replay = verify_original_replay(metrics, reference_path)
    result = {
        "schema_version": 1,
        "purpose": "Complete PCDR external evaluation on OLP",
        "claim_boundary": config["claim_boundary"],
        "config": config,
        "run": {
            "textile_id": textile_id,
            "smoke": args.smoke,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "predict_seconds": predict_seconds,
            "fusion_and_evaluation_seconds": fusion_seconds,
            "checkpoint_sha256": checkpoint_sha256,
            "memory_vectors": int(bank.shape[0]),
            "memory_dimensions": int(bank.shape[1]),
            "script_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(args.config),
            "base_config_sha256": sha256_file(base_path),
        },
        "counts": {key: len(value) for key, value in split.items()},
        "split_scenes": {
            key: sorted({records[index].scene for index in indices})
            for key, indices in split.items()
        },
        "split_paths": {
            key: [records[index].relative_path for index in indices]
            for key, indices in split.items()
        },
        "frozen_source_identity": source_identity,
        "frozen_replay": replay,
        "dual_readout_identity": identities,
        "geometric_fusion": geometric_facts,
        "geometry_calibration": calibrated_facts,
        "context_response_calibration": context_facts,
        "metrics": metrics,
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
                "textile_id": textile_id,
                "frozen_replay": safe["frozen_replay"],
                "pixel_ap": {
                    name: safe["metrics"][name]["pixel_average_precision"]
                    for name in config["reported_variants"]
                },
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
