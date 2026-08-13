"""Frozen RAW-FABRID K0 for crop observations of one physical anomaly field.

The probe scores one label-independent base grid plus horizontal and vertical
64-pixel phase shifts.  All crop anomaly maps are returned to the same parent
coordinates before fixed fusion.  It does not train or tune a proposed method;
it asks whether repeated crop observations contain usable real-defect evidence
beyond a single grid, while retaining mean/min/max as mandatory controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from probe_raw_fabrid_anomalib_patchcore import evaluate, score_modes, select_training_refs
from probe_raw_fabrid_anomalydino import (
    build_memory as build_dino_memory,
    extract_tokens,
    make_loader as make_dino_reference_loader,
)
from probe_raw_fabrid_grouped import annotation_masks, load_rows, split_p2
from probe_raw_patchcore_memory_quantization import embed, load_model as load_patchcore


OBSERVATION_VIEWS = ("base_grid", "x_shifted_grid", "y_shifted_grid")


@dataclass(frozen=True)
class CropRef:
    parent_index: int
    y0: int
    x0: int
    view: str


class CropDataset(Dataset[tuple[torch.Tensor, int, int, int, str]]):
    def __init__(
        self,
        root: Path,
        rows: list[dict[str, object]],
        refs: list[CropRef],
        edge: int,
        model: str,
        model_edge: int,
    ) -> None:
        self.root = root
        self.rows = rows
        self.refs = refs
        self.edge = edge
        self.model = model
        self._cached_index: int | None = None
        self._cached_image: np.ndarray | None = None
        self.dino_transform = transforms.Compose([
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
        ])

    def __len__(self) -> int:
        return len(self.refs)

    def _parent(self, index: int) -> np.ndarray:
        if self._cached_index != index:
            path = self.root / "images" / str(self.rows[index]["filename"])
            self._cached_image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            self._cached_index = index
        assert self._cached_image is not None
        return self._cached_image

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int, int, str]:
        ref = self.refs[position]
        parent = self._parent(ref.parent_index)
        crop = np.ascontiguousarray(
            parent[ref.y0 : ref.y0 + self.edge, ref.x0 : ref.x0 + self.edge]
        )
        if crop.shape != (self.edge, self.edge):
            raise RuntimeError(f"Invalid crop {ref}: {crop.shape}")
        if self.model == "anomalydino":
            tensor = self.dino_transform(Image.fromarray(crop, mode="L").convert("RGB"))
        else:
            tensor = torch.from_numpy(crop).float().div_(255.0).unsqueeze(0).repeat(3, 1, 1)
        return tensor, ref.parent_index, ref.y0, ref.x0, ref.view


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_safe(value: Any) -> Any:
    """Replace non-finite diagnostic values from empty smoke buckets with null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def make_crop_refs(
    indices: list[int],
    height: int,
    width: int,
    edge: int,
    shift: int,
    views: list[str],
) -> list[CropRef]:
    unknown = set(views) - set(OBSERVATION_VIEWS)
    if unknown:
        raise ValueError(f"Unknown observation views: {sorted(unknown)}")
    if "base_grid" not in views:
        raise ValueError("Observation ablations must retain base_grid")
    base_y = list(range(0, height - edge + 1, edge))
    base_x = list(range(0, width - edge + 1, edge))
    shifted_y = list(range(shift, height - edge + 1, edge))
    shifted_x = list(range(shift, width - edge + 1, edge))
    refs: list[CropRef] = []
    for index in sorted(indices):
        if "base_grid" in views:
            refs.extend(CropRef(index, y, x, "base_grid") for y in base_y for x in base_x)
        if "x_shifted_grid" in views:
            refs.extend(
                CropRef(index, y, x, "x_shifted_grid")
                for y in base_y
                for x in shifted_x
            )
        if "y_shifted_grid" in views:
            refs.extend(
                CropRef(index, y, x, "y_shifted_grid")
                for y in shifted_y
                for x in base_x
            )
    if len(set(refs)) != len(refs):
        raise RuntimeError("Crop geometry produced duplicate observations")
    return refs


def make_loader(
    root: Path,
    rows: list[dict[str, object]],
    refs: list[CropRef],
    edge: int,
    model: str,
    model_edge: int,
    batch_size: int,
    workers: int,
) -> DataLoader:
    return DataLoader(
        CropDataset(root, rows, refs, edge, model, model_edge),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(workers),
    )


def empty_accumulator(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    accumulator = {
        "sum": np.zeros(shape, dtype=np.float64),
        "sum_sq": np.zeros(shape, dtype=np.float64),
        "minimum": np.full(shape, np.inf, dtype=np.float32),
        "maximum": np.full(shape, -np.inf, dtype=np.float32),
        "count": np.zeros(shape, dtype=np.uint8),
        "single_base": np.full(shape, np.nan, dtype=np.float32),
    }
    accumulator.update({
        f"view_{view}": np.full(shape, np.nan, dtype=np.float32)
        for view in OBSERVATION_VIEWS
    })
    return accumulator


def add_crop(
    accumulators: dict[int, dict[str, np.ndarray]],
    parent_index: int,
    y0: int,
    x0: int,
    view: str,
    crop_map: np.ndarray,
    stride: int,
    parent_shape: tuple[int, int],
) -> None:
    accumulator = accumulators.setdefault(parent_index, empty_accumulator(parent_shape))
    y = y0 // stride
    x = x0 // stride
    height, width = crop_map.shape
    target = np.s_[y : y + height, x : x + width]
    accumulator["sum"][target] += crop_map
    accumulator["sum_sq"][target] += crop_map.astype(np.float64) ** 2
    accumulator["minimum"][target] = np.minimum(accumulator["minimum"][target], crop_map)
    accumulator["maximum"][target] = np.maximum(accumulator["maximum"][target], crop_map)
    accumulator["count"][target] += 1
    view_key = f"view_{view}"
    if view_key not in accumulator:
        raise ValueError(f"Unknown observation view: {view}")
    if np.isfinite(accumulator[view_key][target]).any():
        raise RuntimeError(f"Observation view overlaps itself: {view} at {(y0, x0)}")
    accumulator[view_key][target] = crop_map
    if view == "base_grid":
        accumulator["single_base"][target] = crop_map


def context_response_fields(
    accumulators: dict[int, dict[str, np.ndarray]],
    fit_indices: list[int],
    expected_indices: list[int],
    crop_tokens: int,
    shift_tokens: int,
    requested_names: set[str],
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    """Estimate a label-free crop-context response likelihood field.

    Each crop observation is robustly standardized by its crop-relative token
    coordinate using source calibration normals.  Positive residual energy is
    then converted to a coverage-pattern-conditional empirical tail score.  The
    pattern conditioning makes cells with one, two, or three observations
    comparable without assuming a Gaussian response distribution.
    """
    calibration_started = time.perf_counter()
    shape = accumulators[expected_indices[0]]["count"].shape
    yy, xx = np.indices(shape)
    offsets = {
        "base_grid": (0, 0),
        "x_shifted_grid": (0, shift_tokens),
        "y_shifted_grid": (shift_tokens, 0),
    }
    relative_ids = {
        view: (
            ((yy - y_offset) % crop_tokens) * crop_tokens
            + ((xx - x_offset) % crop_tokens)
        ).astype(np.int32)
        for view, (y_offset, x_offset) in offsets.items()
    }
    fit_stacks = {
        view: np.stack(
            [accumulators[index][f"view_{view}"] for index in fit_indices], axis=0
        ).astype(np.float32, copy=False)
        for view in OBSERVATION_VIEWS
    }
    relative_bins = crop_tokens * crop_tokens
    centers = np.empty(relative_bins, dtype=np.float32)
    scales = np.empty(relative_bins, dtype=np.float32)
    sample_counts = np.empty(relative_bins, dtype=np.int64)
    for relative_id in range(relative_bins):
        chunks = []
        for view in OBSERVATION_VIEWS:
            values = fit_stacks[view][:, relative_ids[view] == relative_id].reshape(-1)
            chunks.append(values[np.isfinite(values)])
        samples = np.concatenate(chunks)
        if not samples.size:
            raise RuntimeError(f"No normal observations for relative bin {relative_id}")
        center = float(np.median(samples))
        mad = float(np.median(np.abs(samples - center)))
        centers[relative_id] = center
        scales[relative_id] = max(1.4826 * mad, 1e-6)
        sample_counts[relative_id] = samples.size
    coordinate_calibration_seconds = time.perf_counter() - calibration_started

    def response(index: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        standardized = []
        pattern = np.zeros(shape, dtype=np.uint8)
        weighted_sum = np.zeros(shape, dtype=np.float64)
        raw_weighted_sum = np.zeros(shape, dtype=np.float64)
        weight_sum = np.zeros(shape, dtype=np.float64)
        bias_corrected_sum = np.zeros(shape, dtype=np.float64)
        observation_count = np.zeros(shape, dtype=np.float64)
        for bit, view in enumerate(OBSERVATION_VIEWS):
            values = accumulators[index][f"view_{view}"]
            valid = np.isfinite(values)
            pattern[valid] |= np.uint8(1 << bit)
            z = np.full(shape, np.nan, dtype=np.float32)
            ids = relative_ids[view]
            z[valid] = (
                values[valid] - centers[ids[valid]]
            ) / scales[ids[valid]]
            weights = 1.0 / scales[ids[valid]].astype(np.float64) ** 2
            weighted_sum[valid] += (
                values[valid].astype(np.float64) - centers[ids[valid]]
            ) * weights
            raw_weighted_sum[valid] += values[valid].astype(np.float64) * weights
            weight_sum[valid] += weights
            bias_corrected_sum[valid] += (
                values[valid].astype(np.float64) - centers[ids[valid]]
            )
            observation_count[valid] += 1.0
            standardized.append(z)
        stack = np.stack(standardized, axis=0)
        positive = np.maximum(np.nan_to_num(stack, nan=0.0), 0.0)
        energy = np.sum(positive.astype(np.float64) ** 2, axis=0)
        fields = {
            "context_bias_equal_mean": (
                bias_corrected_sum / observation_count
            ).astype(np.float32),
            "context_scale_weighted_mean": (
                raw_weighted_sum / weight_sum
            ).astype(np.float32),
            "context_bias_weighted_mean": (
                weighted_sum / weight_sum
            ).astype(np.float32),
        }
        return energy, pattern, fields

    pattern_reference: dict[int, np.ndarray] = {}
    pattern_counts: dict[str, int] = {}
    if "context_tail_energy" in requested_names:
        fit_energy = {index: response(index) for index in fit_indices}
        observed_patterns = sorted({
            int(value)
            for _, pattern, _ in fit_energy.values()
            for value in np.unique(pattern)
            if int(value) > 0
        })
        for pattern_id in observed_patterns:
            values = np.concatenate([
                energy[pattern == pattern_id]
                for energy, pattern, _ in fit_energy.values()
            ])
            pattern_reference[pattern_id] = np.sort(values)
            pattern_counts[str(pattern_id)] = int(values.size)

    tail_fields: dict[int, np.ndarray] = {}
    physical_fields: dict[str, dict[int, np.ndarray]] = {
        "context_bias_equal_mean": {},
        "context_scale_weighted_mean": {},
        "context_bias_weighted_mean": {},
    }
    recovery_started = time.perf_counter()
    for index in expected_indices:
        energy, pattern, current_fields = response(index)
        if pattern_reference:
            field = np.zeros(shape, dtype=np.float32)
            for pattern_id, reference in pattern_reference.items():
                selected = pattern == pattern_id
                ranks = np.searchsorted(reference, energy[selected], side="right")
                survival = (reference.size - ranks + 1.0) / (reference.size + 1.0)
                field[selected] = -np.log(survival).astype(np.float32)
            tail_fields[index] = field
        for name, current in current_fields.items():
            physical_fields[name][index] = current
    field_recovery_seconds = time.perf_counter() - recovery_started
    returned_fields = {name: fields for name, fields in physical_fields.items() if name in requested_names}
    if "context_tail_energy" in requested_names:
        returned_fields["context_tail_energy"] = tail_fields
    return returned_fields, {
        "fit_normal_parents": len(fit_indices),
        "coordinate_calibration_seconds": coordinate_calibration_seconds,
        "field_recovery_seconds": field_recovery_seconds,
        "field_recovery_milliseconds_per_parent": (
            1000.0 * field_recovery_seconds / len(expected_indices)
        ),
        "relative_coordinate_bins": relative_bins,
        "samples_per_relative_bin_minimum": int(sample_counts.min()),
        "samples_per_relative_bin_median": float(np.median(sample_counts)),
        "robust_scale_minimum": float(scales.min()),
        "robust_scale_median": float(np.median(scales)),
        "coverage_pattern_cell_counts": pattern_counts,
        "score": "pattern-conditional empirical upper-tail of one-sided robust residual energy",
        "physical_field_estimator": "inverse-robust-variance weighted mean after crop-relative median bias removal",
        "ablation_estimators": {
            "context_bias_equal_mean": "equal mean after crop-relative median bias removal",
            "context_scale_weighted_mean": "inverse-robust-variance weighted raw response without bias removal",
        },
        "uses_defect_labels": False,
    }


def finalize_fusions(
    accumulators: dict[int, dict[str, np.ndarray]],
    expected_indices: list[int],
    fusion_names: list[str],
    fit_normal_indices: list[int],
    crop_tokens: int,
    shift_tokens: int,
) -> tuple[
    dict[str, dict[int, np.ndarray]], dict[int, dict[str, np.ndarray]], dict[str, Any]
]:
    supported = {
        "single_base", "mean", "minimum", "maximum", "mean_minus_std",
        "mean_plus_std", "observation_std", "observation_range",
        "context_tail_energy",
        "context_bias_equal_mean",
        "context_scale_weighted_mean",
        "context_bias_weighted_mean",
    }
    unknown = set(fusion_names) - supported
    if unknown:
        raise ValueError(f"Unsupported fusion variants: {sorted(unknown)}")
    variants: dict[str, dict[int, np.ndarray]] = {name: {} for name in fusion_names}
    diagnostics: dict[int, dict[str, np.ndarray]] = {}
    for index in expected_indices:
        current = accumulators[index]
        count = current["count"].astype(np.float64)
        if np.any(count == 0) or np.isnan(current["single_base"]).any():
            raise RuntimeError(f"Parent {index} was not fully covered by the base grid")
        mean = current["sum"] / count
        variance = np.maximum(current["sum_sq"] / count - mean**2, 0.0)
        std = np.sqrt(variance)
        available: dict[str, np.ndarray] = {
            "single_base": current["single_base"],
            "mean": mean,
            "minimum": current["minimum"],
            "maximum": current["maximum"],
            "mean_minus_std": mean - std,
            "mean_plus_std": mean + std,
            "observation_std": std,
            "observation_range": current["maximum"] - current["minimum"],
        }
        for name in fusion_names:
            if name.startswith("context_"):
                continue
            variants[name][index] = available[name].astype(np.float32)
        diagnostics[index] = {
            "count": current["count"],
            "std": std.astype(np.float32),
            "range": (current["maximum"] - current["minimum"]).astype(np.float32),
        }
    context_facts: dict[str, Any] = {}
    context_names = {
        "context_tail_energy",
        "context_bias_equal_mean",
        "context_scale_weighted_mean",
        "context_bias_weighted_mean",
    }.intersection(fusion_names)
    if context_names:
        context_variants, context_facts = context_response_fields(
            accumulators,
            fit_normal_indices,
            expected_indices,
            crop_tokens,
            shift_tokens,
            context_names,
        )
        for name in context_names:
            variants[name] = context_variants[name]
    return variants, diagnostics, context_facts


def observation_diagnostics(
    diagnostics: dict[int, dict[str, np.ndarray]],
    union_masks: dict[int, np.ndarray],
    split: dict[str, list[int]],
) -> dict[str, Any]:
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    scored = split["calibration_normals"] + split["evaluation_normals"] + split["anomalies"]
    anomaly_set = set(split["anomalies"])
    for index in scored:
        valid = diagnostics[index]["count"] >= 2
        mask = union_masks.get(index, np.zeros_like(valid, dtype=bool))
        groups["defect_std"].append(diagnostics[index]["std"][valid & mask])
        groups["defect_range"].append(diagnostics[index]["range"][valid & mask])
        normal_region = valid & ~mask
        prefix = "anomaly_parent_normal_region" if index in anomaly_set else "normal_parent"
        groups[f"{prefix}_std"].append(diagnostics[index]["std"][normal_region])
        groups[f"{prefix}_range"].append(diagnostics[index]["range"][normal_region])

    def summarize(arrays: list[np.ndarray]) -> dict[str, Any]:
        nonempty = [array.reshape(-1) for array in arrays if array.size]
        values = np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float32)
        if not values.size:
            return {"cells": 0}
        return {
            "cells": int(values.size),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.9)),
            "q99": float(np.quantile(values, 0.99)),
            "mean": float(np.mean(values)),
        }

    counts = np.concatenate([item["count"].reshape(-1) for item in diagnostics.values()])
    return {
        "observation_count_histogram": {
            str(value): int(np.sum(counts == value)) for value in np.unique(counts)
        },
        "score_dispersion": {name: summarize(arrays) for name, arrays in groups.items()},
    }


@torch.inference_mode()
def score_patchcore(
    base: dict[str, Any], checkpoint: Path, loader: DataLoader, refs: list[CropRef],
    device: torch.device, output_edge: int, stride: int, parent_shape: tuple[int, int]
) -> tuple[dict[int, dict[str, np.ndarray]], float, dict[str, Any]]:
    model, bank = load_patchcore(base, checkpoint, device)
    accumulators: dict[int, dict[str, np.ndarray]] = {}
    offset = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in loader:
        images, parents, ys, xs, views = batch
        images = images.to(device, non_blocking=True)
        # Engine.predict applies PatchCore's PreProcessor as a Lightning callback.
        # Direct inner-model replay must apply the same resize/normalization here;
        # otherwise the query embeddings do not match the checkpoint memory bank.
        if model.pre_processor is not None:
            images = model.pre_processor(images)
        embedding, grid_shape, output_size = embed(model, images)
        patch_scores, _ = model.model.nearest_neighbors(embedding, n_neighbors=1)
        patch_scores = patch_scores.reshape(images.shape[0], 1, *grid_shape)
        maps = model.model.anomaly_map_generator(patch_scores, output_size)
        maps = F.interpolate(
            maps.float(), size=(output_edge, output_edge), mode="bilinear", align_corners=False
        )[:, 0].cpu().numpy().astype(np.float32, copy=False)
        for position, crop_map in enumerate(maps):
            add_crop(
                accumulators, int(parents[position]), int(ys[position]), int(xs[position]),
                str(views[position]), crop_map, stride, parent_shape,
            )
        offset += images.shape[0]
        if offset == images.shape[0] or offset % 2000 < images.shape[0]:
            print(f"patchcore crops {offset}/{len(refs)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return accumulators, elapsed, {
        "checkpoint_sha256": sha256_file(checkpoint),
        "memory_vectors": int(bank.shape[0]),
        "memory_dimensions": int(bank.shape[1]),
    }


@torch.inference_mode()
def score_dino(
    base: dict[str, Any], data_root: Path, rows: list[dict[str, object]],
    split: dict[str, list[int]], loader: DataLoader, refs: list[CropRef],
    device: torch.device, edge: int, output_edge: int, stride: int,
    parent_shape: tuple[int, int], smoke: bool,
) -> tuple[dict[int, dict[str, np.ndarray]], float, dict[str, Any]]:
    tiling = base["tiling"]
    model_config = base["model"]
    tile_rows = int(tiling["rows"])
    tile_columns = int(tiling["columns"])
    budget = 2 if smoke else int(tiling["reference_tile_budget"])
    reference_refs = select_training_refs(
        split["training_normals"], tile_rows, tile_columns, budget, int(base["seed"])
    )
    repository = f"{model_config['backbone_repository']}:{model_config['backbone_repository_commit']}"
    model = torch.hub.load(
        repository, str(model_config["backbone"]), trust_repo=True, skip_validation=True
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    reference_loader = make_dino_reference_loader(
        data_root, rows, reference_refs, edge, edge, int(tiling["model_edge_size"]),
        int(model_config["reference_batch_size"]), 0 if smoke else int(model_config["num_workers"]),
    )
    memory, memory_seconds = build_dino_memory(model, reference_loader, device)
    memory_t = memory.T.contiguous()
    token_edge = int(tiling["model_edge_size"]) // int(model_config["patch_size"])
    accumulators: dict[int, dict[str, np.ndarray]] = {}
    offset = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in loader:
        images, parents, ys, xs, views = batch
        tokens = extract_tokens(model, images, device)
        similarities = tokens.reshape(-1, tokens.shape[-1]) @ memory_t
        maps = (1.0 - similarities.max(dim=1).values).reshape(
            images.shape[0], 1, token_edge, token_edge
        )
        maps = F.interpolate(
            maps.float(), size=(output_edge, output_edge), mode="bilinear", align_corners=False
        )[:, 0].cpu().numpy().astype(np.float32, copy=False)
        for position, crop_map in enumerate(maps):
            add_crop(
                accumulators, int(parents[position]), int(ys[position]), int(xs[position]),
                str(views[position]), crop_map, stride, parent_shape,
            )
        offset += images.shape[0]
        if offset == images.shape[0] or offset % 2000 < images.shape[0]:
            print(f"anomalydino crops {offset}/{len(refs)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return accumulators, elapsed, {
        "memory_build_seconds": memory_seconds,
        "memory_vectors": int(memory.shape[0]),
        "memory_dimensions": int(memory.shape[1]),
        "reference_tiles": len(reference_refs),
    }


def compact_metrics(metrics: dict[str, Any], mode: str, normal_count: int) -> dict[str, Any]:
    current = metrics[mode]
    return {
        "parent_average_precision": float(current["average_precision"]),
        "pixel_average_precision": float(current["pixel_average_precision"]),
        "source_parent_recall": float(
            current["source_calibrated"]["defect_parent_recall_at_1pct_fpr"]
        ),
        "target_normal_false_positives": int(round(
            float(current["source_calibrated"]["normal_parent_fpr"]) * normal_count
        )),
        "instance_auc_all": float(
            current["instance_recall"]["all"]["vs_target_normal_maximum_pairwise_auc"]
        ),
        "instance_auc_small": float(
            current["instance_recall"]["small"]["vs_target_normal_maximum_pairwise_auc"]
        ),
        "instance_auc_elongated": float(
            current["instance_recall"]["elongated"]["vs_target_normal_maximum_pairwise_auc"]
        ),
        "instance_recall_all": float(current["instance_recall"]["all"]["recall"]),
        "instance_recall_small": float(current["instance_recall"]["small"]["recall"]),
        "instance_recall_elongated": float(
            current["instance_recall"]["elongated"]["recall"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", choices=("patchcore", "anomalydino"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fold")
    parser.add_argument("--context-calibration-parent-limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    probe = json.loads(args.config.read_text(encoding="utf-8"))
    base_path = Path(str(probe["base_configs"][args.model]))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    data_root = args.data_root or Path(str(base["data"]["root"]))
    rows = load_rows(data_root, base["data"])
    by_roll: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_roll[str(row["roll_id"])].append(index)
    fold = str(args.fold or probe["fold"])
    split = split_p2(fold, by_roll, rows)
    if args.smoke:
        split = {
            "training_normals": split["training_normals"][:2],
            "calibration_normals": split["calibration_normals"][:2],
            "evaluation_normals": split["evaluation_normals"][:2],
            "anomalies": split["anomalies"][:2],
        }

    geometry = probe["crop_observations"]
    edge = int(geometry["physical_crop_edge"])
    shift = int(geometry["phase_shift_pixels"])
    stride = int(geometry["output_stride_pixels"])
    height = int(base["data"]["image_height"])
    width = int(base["data"]["image_width"])
    if edge % stride or shift % stride or height % stride or width % stride:
        raise ValueError("Crop, shift and parent geometry must align to the output stride")
    scored_indices = sorted(set(
        split["calibration_normals"] + split["evaluation_normals"] + split["anomalies"]
    ))
    observation_views = [str(view) for view in geometry["views"]]
    refs = make_crop_refs(
        scored_indices, height, width, edge, shift, observation_views
    )
    crops_per_parent = len(refs) // len(scored_indices)
    expected_ratio = crops_per_parent / ((height // edge) * (width // edge))
    if not np.isclose(expected_ratio, float(geometry["processed_pixel_ratio_to_single_grid"])):
        raise ValueError(f"Recorded pixel ratio mismatch: {expected_ratio}")

    model_config = base["model"]
    model_edge = int(base["tiling"].get("model_edge_size", edge))
    workers = 0 if args.smoke else int(model_config["num_workers"])
    batch_size = int(model_config.get("predict_batch_size", 8))
    loader = make_loader(
        data_root, rows, refs, edge, args.model, model_edge, batch_size, workers
    )
    torch.manual_seed(int(probe["seed"]))
    np.random.seed(int(probe["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(probe["seed"]))
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
    parent_shape = (height // stride, width // stride)
    output_edge = edge // stride

    if args.model == "patchcore":
        checkpoint = args.checkpoint or Path(str(probe["models"]["patchcore"]["checkpoint"]))
        accumulators, predict_seconds, model_facts = score_patchcore(
            base, checkpoint, loader, refs, device, output_edge, stride, parent_shape
        )
    else:
        accumulators, predict_seconds, model_facts = score_dino(
            base, data_root, rows, split, loader, refs, device, edge, output_edge,
            stride, parent_shape, args.smoke,
        )
    fusion_started = time.perf_counter()
    fit_normal_indices = list(split["calibration_normals"])
    if args.context_calibration_parent_limit is not None:
        if args.context_calibration_parent_limit <= 0:
            raise ValueError("context calibration parent limit must be positive")
        fit_normal_indices = fit_normal_indices[: args.context_calibration_parent_limit]
    raw_variants, diagnostics, context_facts = finalize_fusions(
        accumulators,
        scored_indices,
        list(probe["fusion_controls"]),
        fit_normal_indices,
        edge // stride,
        shift // stride,
    )
    fusion_seconds = time.perf_counter() - fusion_started

    if args.model == "anomalydino":
        base["scores"]["image_top_fraction"] = float(
            base["scores"]["matched_image_top_fraction"]
        )
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
    mode = str(probe["primary_score_mode"])
    compact = {
        variant: compact_metrics(current, mode, len(split["evaluation_normals"]))
        for variant, current in metrics.items()
    }
    baseline = compact["single_base"]
    deltas = {
        variant: {
            key: float(value - baseline[key])
            for key, value in values.items()
            if isinstance(value, (int, float))
        }
        for variant, values in compact.items()
        if variant != "single_base"
    }
    gate = probe["evidence_gate"]
    candidate = compact[str(gate["fixed_candidate_fusion"])]
    checks = {
        "parent_recall_retained": candidate["source_parent_recall"]
        >= baseline["source_parent_recall"] - float(gate["maximum_parent_recall_loss"]),
        "small_auc_retained": candidate["instance_auc_small"]
        >= baseline["instance_auc_small"] - float(gate["maximum_small_instance_auc_loss"]),
        "elongated_auc_retained": candidate["instance_auc_elongated"]
        >= baseline["instance_auc_elongated"] - float(gate["maximum_elongated_instance_auc_loss"]),
        "material_signal": (
            candidate["pixel_average_precision"]
            >= baseline["pixel_average_precision"] + float(gate["material_pixel_ap_gain"])
            or candidate["instance_auc_all"]
            >= baseline["instance_auc_all"] + float(gate["material_instance_auc_gain"])
        ),
    }
    simple_name = gate.get("required_simple_fusion")
    simple_comparison = None
    if simple_name is not None:
        simple = compact[str(simple_name)]
        pixel_gain = candidate["pixel_average_precision"] - simple["pixel_average_precision"]
        auc_gain = candidate["instance_auc_all"] - simple["instance_auc_all"]
        beats_simple = (
            pixel_gain >= float(gate["minimum_pixel_ap_gain_over_simple"])
            or auc_gain >= float(gate["minimum_instance_auc_gain_over_simple"])
        )
        checks["beats_required_simple_fusion"] = beats_simple
        simple_comparison = {
            "fusion": simple_name,
            "pixel_average_precision_gain": float(pixel_gain),
            "instance_auc_all_gain": float(auc_gain),
        }
    result = {
        "schema_version": 1,
        "config": probe,
        "run": {
            "model": args.model,
            "fold": fold,
            "smoke": args.smoke,
            "context_calibration_parent_limit": args.context_calibration_parent_limit,
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
            **model_facts,
        },
        "counts": {
            **{key: len(value) for key, value in split.items()},
            "scored_parents": len(scored_indices),
            "crops_per_parent": crops_per_parent,
            "total_crops": len(refs),
            "base_crops_per_parent": (height // edge) * (width // edge),
        },
        "split_indices": split,
        "observation_diagnostics": observation_diagnostics(diagnostics, union_masks, split),
        "context_response_calibration": context_facts or None,
        "compact_primary_metrics": compact,
        "deltas_from_single_base": deltas,
        "fixed_candidate_gate": {
            "fusion": gate["fixed_candidate_fusion"],
            "checks": checks,
            "simple_comparison": simple_comparison,
            "passed_this_model": all(checks.values()),
            "scope": gate["interpretation"],
        },
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
        "compact_primary_metrics": safe_result["compact_primary_metrics"],
        "fixed_candidate_gate": safe_result["fixed_candidate_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
