"""Frozen one-fold gate for PatchCore memory-bank precision and low-FPR tails.

The checkpoint, backbone, split, full-frame tiling, query embeddings, score modes,
and evaluation code are shared across precision variants. Quantized banks are
dequantized before the existing exact nearest-neighbour search, so this probe tests
score perturbation only; it cannot support latency or hardware-efficiency claims.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from anomalib.models import Patchcore

from probe_raw_fabrid_anomalib_patchcore import (
    all_tile_refs,
    evaluate,
    make_loader,
    score_modes,
)
from probe_raw_fabrid_grouped import annotation_masks, load_rows, split_p2


def quantize_dequantize(bank: torch.Tensor, variant: dict[str, object]) -> torch.Tensor:
    name = str(variant["name"])
    if name == "fp32":
        return bank.clone()
    if name == "fp16_roundtrip":
        return bank.half().float()
    bits = int(variant["bits"])
    qmax = float(2 ** (bits - 1) - 1)
    granularity = str(variant["granularity"])
    if granularity == "per_tensor_symmetric":
        scale = bank.abs().amax().clamp_min(torch.finfo(torch.float32).eps) / qmax
    elif granularity == "per_channel_symmetric":
        scale = bank.abs().amax(dim=0, keepdim=True).clamp_min(
            torch.finfo(torch.float32).eps
        ) / qmax
    else:
        raise ValueError(f"Unsupported quantization granularity: {granularity}")
    return torch.round(bank / scale).clamp(-qmax, qmax).mul(scale)


def load_model(
    base: dict[str, object], checkpoint_path: Path, device: torch.device
) -> tuple[Patchcore, torch.Tensor]:
    model_config = base["model"]
    model = Patchcore(
        backbone=str(model_config["backbone"]),
        layers=tuple(model_config["layers"]),
        pre_trained=False,
        coreset_sampling_ratio=float(model_config["coreset_sampling_ratio"]),
        num_neighbors=int(model_config["num_neighbors"]),
        precision=str(model_config["precision"]),
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    bank = state["model.memory_bank"].float().contiguous()
    model.model.memory_bank = bank.clone()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, bank


def embed(model: Patchcore, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    inner = model.model
    output_size = tuple(images.shape[-2:])
    with torch.no_grad():
        features = inner.feature_extractor(images)
        features = {layer: inner.feature_pooler(value) for layer, value in features.items()}
        grid = inner.generate_embedding(features)
        grid_shape = tuple(grid.shape[-2:])
        embedding = inner.reshape_embedding(grid)
    return embedding, grid_shape, output_size


def put_tiles(
    destination: dict[int, np.ndarray],
    refs: list[object],
    tile_maps: torch.Tensor,
    tile_rows: int,
    tile_columns: int,
    tile_grid_height: int,
    tile_grid_width: int,
) -> None:
    arrays = tile_maps[:, 0].detach().float().cpu().numpy()
    for ref, tile_map in zip(refs, arrays, strict=True):
        index = int(ref.parent_index)
        if index not in destination:
            destination[index] = np.empty(
                (tile_rows * tile_grid_height, tile_columns * tile_grid_width),
                dtype=np.float32,
            )
        y0 = int(ref.row) * tile_grid_height
        x0 = int(ref.column) * tile_grid_width
        destination[index][y0 : y0 + tile_grid_height, x0 : x0 + tile_grid_width] = tile_map


def fixed_fp32_threshold_metrics(
    metrics: dict[str, object],
    split: dict[str, list[int]],
    base: dict[str, object],
) -> dict[str, object]:
    baseline = metrics["fp32"]
    result: dict[str, object] = {}
    small_area = float(base["evaluation"]["small_instance_area_px"])
    elongated_aspect = float(base["evaluation"]["elongated_instance_aspect_ratio"])
    for variant, variant_metrics in metrics.items():
        result[variant] = {}
        for mode, current in variant_metrics.items():
            image_threshold = float(baseline[mode]["source_calibrated"]["image_threshold"])
            pixel_threshold = float(baseline[mode]["source_calibrated"]["pixel_threshold"])
            parents = current["parent_scores"]
            records = current["instance_records"]
            buckets = {
                "all": records,
                "small": [x for x in records if float(x["area_px"]) <= small_area],
                "elongated": [
                    x for x in records if float(x["aspect_ratio"]) >= elongated_aspect
                ],
            }
            result[variant][mode] = {
                "normal_parent_fpr": float(np.mean([
                    float(parents[str(index)]["image_score"]) > image_threshold
                    for index in split["evaluation_normals"]
                ])),
                "defect_parent_recall": float(np.mean([
                    float(parents[str(index)]["image_score"]) > image_threshold
                    for index in split["anomalies"]
                ])),
                "instance_recall": {
                    name: float(np.mean([
                        float(item["maximum_score"]) > pixel_threshold for item in items
                    ])) if items else float("nan")
                    for name, items in buckets.items()
                },
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    probe = json.loads(args.config.read_text(encoding="utf-8"))
    base_path = Path(str(probe["base_config"]))
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    base = json.loads(base_path.read_text(encoding="utf-8"))
    data_root = args.data_root or Path(str(base["data"]["root"]))
    rows = load_rows(data_root, base["data"])
    by_roll: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_roll[str(row["roll_id"])].append(index)
    fold = str(probe["fold"])
    split = split_p2(fold, by_roll, rows)
    if args.smoke:
        split = {
            "training_normals": split["training_normals"][:2],
            "calibration_normals": split["calibration_normals"][:1],
            "evaluation_normals": split["evaluation_normals"][:1],
            "anomalies": split["anomalies"][:1],
        }

    tiling = base["tiling"]
    tile_rows = int(tiling["rows"])
    tile_columns = int(tiling["columns"])
    tile_height = int(tiling["tile_height"])
    tile_width = int(tiling["tile_width"])
    stride = int(tiling["evaluation_stride"])
    tile_grid_height = tile_height // stride
    tile_grid_width = tile_width // stride
    scored_indices = sorted(set(
        split["calibration_normals"] + split["evaluation_normals"] + split["anomalies"]
    ))
    refs = all_tile_refs(scored_indices, tile_rows, tile_columns)
    workers = 0 if args.smoke else int(base["model"]["num_workers"])
    loader = make_loader(
        data_root,
        rows,
        refs,
        tile_height,
        tile_width,
        int(base["model"]["predict_batch_size"]),
        workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fp32_bank = load_model(base, args.checkpoint, device)
    variants = {
        str(item["name"]): quantize_dequantize(fp32_bank, item).to(device)
        for item in probe["variants"]
    }
    reconstruction = {
        name: {
            "maximum_absolute_error": float((bank.cpu() - fp32_bank).abs().amax()),
            "root_mean_squared_error": float(torch.sqrt(torch.mean(
                (bank.cpu() - fp32_bank) ** 2
            ))),
        }
        for name, bank in variants.items()
    }
    raw_maps: dict[str, dict[int, np.ndarray]] = {name: {} for name in variants}
    offset = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            images = batch.image.to(device, non_blocking=True)
            embedding, grid_shape, output_size = embed(model, images)
            batch_refs = refs[offset : offset + images.shape[0]]
            offset += images.shape[0]
            for name, bank in variants.items():
                model.model.memory_bank = bank
                patch_scores, _ = model.model.nearest_neighbors(embedding, n_neighbors=1)
                patch_scores = patch_scores.reshape(images.shape[0], 1, *grid_shape)
                maps = model.model.anomaly_map_generator(patch_scores, output_size)
                maps = F.interpolate(
                    maps.float(),
                    size=(tile_grid_height, tile_grid_width),
                    mode="bilinear",
                    align_corners=False,
                )
                put_tiles(
                    raw_maps[name], batch_refs, maps, tile_rows, tile_columns,
                    tile_grid_height, tile_grid_width,
                )
    elapsed = time.perf_counter() - start
    if offset != len(refs):
        raise RuntimeError(f"Prediction count mismatch: {offset} != {len(refs)}")

    maps = {
        variant: {
            index: score_modes(raw, int(base["scores"]["local_mean_kernel_tokens"]))
            for index, raw in parent_maps.items()
        }
        for variant, parent_maps in raw_maps.items()
    }
    coco = json.loads((data_root / str(base["data"]["coco"])).read_text(encoding="utf-8"))
    annotations, union_masks = annotation_masks(
        coco,
        rows,
        int(base["data"]["image_height"]) // stride,
        int(base["data"]["image_width"]) // stride,
    )
    metrics = {
        variant: evaluate(current, split, annotations, union_masks, base)
        for variant, current in maps.items()
    }
    fixed = fixed_fp32_threshold_metrics(metrics, split, base)

    decision_config = probe["decision"]
    primary = str(decision_config["primary_variant"])
    mode = "robust_z_local_mean_max"
    ap_loss = float(metrics["fp32"][mode]["average_precision"]) - float(
        metrics[primary][mode]["average_precision"]
    )
    parent_loss = float(fixed["fp32"][mode]["defect_parent_recall"]) - float(
        fixed[primary][mode]["defect_parent_recall"]
    )
    small_loss = float(fixed["fp32"][mode]["instance_recall"]["small"]) - float(
        fixed[primary][mode]["instance_recall"]["small"]
    )
    elongated_loss = float(fixed["fp32"][mode]["instance_recall"]["elongated"]) - float(
        fixed[primary][mode]["instance_recall"]["elongated"]
    )
    gates = {
        "parent_ap_preserved": ap_loss <= float(
            decision_config["maximum_parent_ap_loss_for_tail_only_failure"]
        ),
        "fixed_threshold_parent_recall_fails": parent_loss >= float(
            decision_config["minimum_fixed_fp32_threshold_parent_recall_loss"]
        ),
        "fixed_threshold_small_or_elongated_recall_fails": max(small_loss, elongated_loss)
        >= float(decision_config["minimum_fixed_fp32_threshold_small_or_elongated_recall_loss"]),
    }
    result = {
        "schema_version": 1,
        "probe_config": probe,
        "base_config": base,
        "run": {
            "fold": fold,
            "smoke": args.smoke,
            "checkpoint": str(args.checkpoint),
            "data_root": str(data_root),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "elapsed_seconds": elapsed,
            "scored_parents": len(scored_indices),
            "prediction_tiles": len(refs),
            "shared_query_embeddings_across_variants": True,
            "simulated_quantize_dequantize_only": True,
        },
        "memory_bank": {
            "shape": list(fp32_bank.shape),
            "reconstruction": reconstruction,
        },
        "split_indices": split,
        "metrics": metrics,
        "fixed_fp32_threshold_metrics": fixed,
        "decision": {
            "status": "tail_candidate_for_seven_fold_replay" if all(gates.values()) else "one_fold_tail_gate_failed",
            "gates": gates,
            "deltas": {
                "parent_ap_loss": ap_loss,
                "fixed_threshold_parent_recall_loss": parent_loss,
                "fixed_threshold_small_instance_recall_loss": small_loss,
                "fixed_threshold_elongated_instance_recall_loss": elongated_loss,
            },
            "scope": decision_config["interpretation"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "run": result["run"],
        "memory_bank": result["memory_bank"],
        "decision": result["decision"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
