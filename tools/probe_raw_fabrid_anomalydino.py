"""Frozen AnomalyDINO-mechanism probe for full-width RAW-FABRID P2.

The probe keeps the project's 256-pixel physical tile grid but resizes each
tile to 252x252, the nearest DINOv2-S/14-compatible size. It therefore scans
96.9% as many model-input pixels as the 256x256 PatchCore protocol. The memory
mechanism follows AnomalyDINO: last-layer DINOv2 patches, L2 normalization,
one-nearest-neighbour cosine distance, and mean top-1% image scoring.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from probe_raw_fabrid_anomalib_patchcore import (
    TileRef,
    all_tile_refs,
    evaluate,
    evaluate_spatial_extreme_controls,
    score_modes,
    select_training_refs,
)
from probe_raw_fabrid_grouped import annotation_masks, load_rows, split_p2


class DinoTileDataset(Dataset[tuple[torch.Tensor, int, int, int]]):
    """Read fixed grayscale tiles and apply the audited AnomalyDINO transform."""

    def __init__(
        self,
        data_root: Path,
        rows: list[dict[str, object]],
        refs: list[TileRef],
        tile_height: int,
        tile_width: int,
        model_edge: int,
    ) -> None:
        self.data_root = data_root
        self.rows = rows
        self.refs = refs
        self.tile_height = tile_height
        self.tile_width = tile_width
        self._cached_index: int | None = None
        self._cached_image: np.ndarray | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size=model_edge,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.refs)

    def _parent(self, index: int) -> np.ndarray:
        if self._cached_index != index:
            path = self.data_root / "images" / str(self.rows[index]["filename"])
            self._cached_image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            self._cached_index = index
        assert self._cached_image is not None
        return self._cached_image

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int, int]:
        ref = self.refs[position]
        parent = self._parent(ref.parent_index)
        y0 = ref.row * self.tile_height
        x0 = ref.column * self.tile_width
        tile = np.ascontiguousarray(parent[y0 : y0 + self.tile_height, x0 : x0 + self.tile_width])
        image = Image.fromarray(tile, mode="L").convert("RGB")
        return self.transform(image), ref.parent_index, ref.row, ref.column


def make_loader(
    data_root: Path,
    rows: list[dict[str, object]],
    refs: list[TileRef],
    tile_height: int,
    tile_width: int,
    model_edge: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        DinoTileDataset(data_root, rows, refs, tile_height, tile_width, model_edge),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
    )


@torch.inference_mode()
def extract_tokens(model: torch.nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    tokens = model.get_intermediate_layers(images.to(device, non_blocking=True))[0]
    if tokens.ndim != 3:
        raise RuntimeError(f"Unexpected DINOv2 token shape: {tuple(tokens.shape)}")
    return F.normalize(tokens.float(), dim=-1)


@torch.inference_mode()
def build_memory(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    batches: list[torch.Tensor] = []
    for images, _, _, _ in loader:
        batches.append(extract_tokens(model, images, device).reshape(-1, model.embed_dim))
    memory = torch.cat(batches, dim=0).contiguous()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return memory, time.perf_counter() - start


@torch.inference_mode()
def score_tiles(
    model: torch.nn.Module,
    loader: DataLoader,
    memory: torch.Tensor,
    device: torch.device,
    tile_rows: int,
    tile_columns: int,
    token_rows: int,
    token_columns: int,
    expected_tiles: int,
) -> tuple[dict[int, np.ndarray], float]:
    maps: dict[int, np.ndarray] = {}
    seen = 0
    memory_t = memory.T.contiguous()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for images, parent_indices, tile_row_indices, tile_column_indices in loader:
        tokens = extract_tokens(model, images, device)
        batch, token_count, channels = tokens.shape
        if token_count != token_rows * token_columns:
            raise RuntimeError(
                f"Token-grid mismatch: {token_count} != {token_rows}x{token_columns}"
            )
        similarities = tokens.reshape(-1, channels) @ memory_t
        distances = (1.0 - similarities.max(dim=1).values).reshape(
            batch, token_rows, token_columns
        )
        distances_cpu = distances.cpu().numpy().astype(np.float32, copy=False)
        for offset in range(batch):
            parent_index = int(parent_indices[offset])
            tile_row = int(tile_row_indices[offset])
            tile_column = int(tile_column_indices[offset])
            if parent_index not in maps:
                maps[parent_index] = np.empty(
                    (tile_rows * token_rows, tile_columns * token_columns), dtype=np.float32
                )
            y0 = tile_row * token_rows
            x0 = tile_column * token_columns
            maps[parent_index][y0 : y0 + token_rows, x0 : x0 + token_columns] = distances_cpu[offset]
        seen += batch
        if seen == batch or seen % 800 < batch:
            print(f"score-tiles {seen}/{expected_tiles}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if seen != expected_tiles:
        raise RuntimeError(f"Scored tile count mismatch: {seen} != {expected_tiles}")
    return maps, seconds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def compare_with_patchcore(
    metrics: dict[str, Any],
    strongest: str,
    normal_count: int,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    current = metrics[strongest]
    reference = comparison["reference_values"]
    tolerance = float(comparison["noninferiority_tolerance"])
    improvement = float(comparison["material_localization_improvement"])
    normal_fp = int(round(float(current["source_calibrated"]["normal_parent_fpr"]) * normal_count))
    values = {
        "parent_average_precision": float(current["average_precision"]),
        "pixel_average_precision": float(current["pixel_average_precision"]),
        "source_normal_false_positives": normal_fp,
        "source_parent_recall": float(
            current["source_calibrated"]["defect_parent_recall_at_1pct_fpr"]
        ),
        "instance_auc_all": float(
            current["instance_recall"]["all"]["vs_target_normal_maximum_pairwise_auc"]
        ),
        "instance_auc_small": float(
            current["instance_recall"]["small"]["vs_target_normal_maximum_pairwise_auc"]
        ),
        "instance_auc_elongated": float(
            current["instance_recall"]["elongated"]["vs_target_normal_maximum_pairwise_auc"]
        ),
    }
    noninferiority_keys = [
        "parent_average_precision",
        "pixel_average_precision",
        "source_parent_recall",
        "instance_auc_all",
        "instance_auc_small",
        "instance_auc_elongated",
    ]
    checks = {
        key: values[key] >= float(reference[key]) - tolerance for key in noninferiority_keys
    }
    checks["source_normal_false_positives"] = normal_fp <= int(
        reference["source_normal_false_positives"]
    ) + int(comparison["maximum_additional_normal_false_positives"])
    localization_keys = [
        "pixel_average_precision",
        "instance_auc_all",
        "instance_auc_small",
        "instance_auc_elongated",
    ]
    improvements = {
        key: values[key] - float(reference[key]) for key in localization_keys
    }
    competitive = all(checks.values())
    residual_explained = competitive and max(improvements.values()) >= improvement
    return {
        "strongest_mode": strongest,
        "values": values,
        "reference_values": reference,
        "noninferiority_checks": checks,
        "localization_deltas": improvements,
        "competitive_gate_passed": competitive,
        "residual_explained_gate_passed": residual_explained,
        "scope": (
            "One-fold baseline gate only. Failure does not establish novelty and does not reject other "
            "DINO/ViT memories, scales, adaptation, or learning objectives."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fold", default="Rollo4A")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--position-counterfactual",
        action="store_true",
        help="Record label-assisted same-position normal comparisons for failure diagnosis.",
    )
    parser.add_argument(
        "--spatial-control-config",
        type=Path,
        help="Evaluate frozen source-only spatial extreme controls from this config.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = args.data_root or Path(str(config["data"]["root"]))
    rows = load_rows(data_root, config["data"])
    by_roll: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_roll[str(row["roll_id"])].append(index)
    if args.fold not in by_roll:
        raise ValueError(f"Unknown fold {args.fold}; available: {sorted(by_roll)}")
    split = split_p2(args.fold, by_roll, rows)
    if args.smoke:
        calibration_smoke_count = 2 if args.spatial_control_config is not None else 1
        split = {
            "training_normals": split["training_normals"][:2],
            "calibration_normals": split["calibration_normals"][:calibration_smoke_count],
            "evaluation_normals": split["evaluation_normals"][:1],
            "anomalies": split["anomalies"][:1],
        }

    tiling = config["tiling"]
    model_config = config["model"]
    tile_rows = int(tiling["rows"])
    tile_columns = int(tiling["columns"])
    tile_height = int(tiling["source_tile_height"])
    tile_width = int(tiling["source_tile_width"])
    model_edge = int(tiling["model_edge_size"])
    patch_size = int(model_config["patch_size"])
    if model_edge % patch_size:
        raise ValueError("DINO input edge must be divisible by its patch size")
    if (tile_rows * tile_height, tile_columns * tile_width) != (
        int(config["data"]["image_height"]),
        int(config["data"]["image_width"]),
    ):
        raise ValueError("Physical tiling must exactly cover every full parent")
    expected_ratio = model_edge**2 / (tile_height * tile_width)
    if not math.isclose(expected_ratio, float(tiling["pixel_budget_ratio_to_256_square_scan"])):
        raise ValueError("Recorded pixel-budget ratio does not match the frozen geometry")

    reference_budget = min(2, int(tiling["reference_tile_budget"])) if args.smoke else int(
        tiling["reference_tile_budget"]
    )
    reference_refs = select_training_refs(
        split["training_normals"], tile_rows, tile_columns, reference_budget, int(config["seed"])
    )
    if len({ref.parent_index for ref in reference_refs}) != len(reference_refs):
        raise RuntimeError("The frozen few-shot memory must cover distinct source-normal parents")
    scored_indices = sorted(
        set(split["calibration_normals"] + split["evaluation_normals"] + split["anomalies"])
    )
    predict_refs = all_tile_refs(scored_indices, tile_rows, tile_columns)
    workers = 0 if args.smoke else int(model_config["num_workers"])

    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
    repository = (
        f"{model_config['backbone_repository']}:"
        f"{model_config['backbone_repository_commit']}"
    )
    model = torch.hub.load(
        repository,
        str(model_config["backbone"]),
        trust_repo=True,
        skip_validation=True,
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    reference_loader = make_loader(
        data_root,
        rows,
        reference_refs,
        tile_height,
        tile_width,
        model_edge,
        int(model_config["reference_batch_size"]),
        workers,
    )
    predict_loader = make_loader(
        data_root,
        rows,
        predict_refs,
        tile_height,
        tile_width,
        model_edge,
        int(model_config["predict_batch_size"]),
        workers,
    )
    memory, memory_seconds = build_memory(model, reference_loader, device)
    token_edge = model_edge // patch_size
    raw_maps, predict_seconds = score_tiles(
        model,
        predict_loader,
        memory,
        device,
        tile_rows,
        tile_columns,
        token_edge,
        token_edge,
        len(predict_refs),
    )
    maps = {
        index: score_modes(raw, int(config["scores"]["local_mean_kernel_tokens"]))
        for index, raw in raw_maps.items()
    }
    coco = json.loads((data_root / str(config["data"]["coco"])).read_text(encoding="utf-8"))
    annotations, union_masks = annotation_masks(
        coco,
        rows,
        tile_rows * token_edge,
        tile_columns * token_edge,
    )
    official_config = copy.deepcopy(config)
    official_config["scores"]["image_top_fraction"] = float(
        config["scores"]["official_image_top_fraction"]
    )
    matched_config = copy.deepcopy(config)
    matched_config["scores"]["image_top_fraction"] = float(
        config["scores"]["matched_image_top_fraction"]
    )
    official_metrics = evaluate(
        maps,
        split,
        annotations,
        union_masks,
        official_config,
        position_counterfactual=args.position_counterfactual,
    )
    matched_metrics = evaluate(
        maps,
        split,
        annotations,
        union_masks,
        matched_config,
        position_counterfactual=args.position_counterfactual,
    )
    spatial_extreme_control = None
    if args.spatial_control_config is not None:
        spatial_control_config = json.loads(
            args.spatial_control_config.read_text(encoding="utf-8")
        )
        if args.smoke:
            spatial_control_config["calibration_partition"]["minimum_parents_each"] = 1
        spatial_extreme_control = evaluate_spatial_extreme_controls(
            maps,
            split,
            annotations,
            union_masks,
            config,
            spatial_control_config,
        )
    strongest = max(
        config["scores"]["modes"],
        key=lambda mode: float(matched_metrics[mode]["average_precision"]),
    )
    comparison = compare_with_patchcore(
        matched_metrics,
        strongest,
        len(split["evaluation_normals"]),
        config["frozen_Rollo4A_comparison"],
    )
    run = {
        "fold": args.fold,
        "protocol": "P2",
        "smoke": args.smoke,
        "position_counterfactual": args.position_counterfactual,
        "data_root": str(data_root),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "memory_build_seconds": memory_seconds,
        "predict_seconds": predict_seconds,
        "milliseconds_per_scored_parent": 1000.0 * predict_seconds / len(scored_indices),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.config),
    }
    result = {
        "schema_version": 1,
        "config": config,
        "run": run,
        "counts": {
            **{key: len(value) for key, value in split.items()},
            "reference_tiles": len(reference_refs),
            "reference_parents_covered": len({ref.parent_index for ref in reference_refs}),
            "reference_tokens": int(memory.shape[0]),
            "scored_parents": len(scored_indices),
            "prediction_tiles": len(predict_refs),
            "tokens_per_tile": token_edge**2,
            "stitched_token_grid": [tile_rows * token_edge, tile_columns * token_edge],
        },
        "split_indices": split,
        "reference_tile_refs": [
            {"parent_index": ref.parent_index, "row": ref.row, "column": ref.column}
            for ref in reference_refs
        ],
        "metrics": {
            "official_top_1_percent": official_metrics,
            "matched_top_0_1_percent": matched_metrics,
        },
        "decision": {
            "comparison_to_patchcore_Rollo4A": comparison,
            "scope": config["interpretation"],
        },
    }
    if spatial_extreme_control is not None:
        result["spatial_extreme_control"] = spatial_extreme_control
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run": run, "counts": result["counts"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
