"""Run one frozen OLP scene-grouped PatchCore/PCAF external fold.

Each eligible textile uses one acquisition scene for normal-only PatchCore
training, another for normal-only coordinate calibration, and all remaining
scenes for evaluation.  The scene assignment and parent subsampling are read
from the metadata-only audit produced before detector outputs existed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from anomalib.data.dataclasses import ImageBatch, ImageItem
from anomalib.engine import Engine
from anomalib.models import Patchcore
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from olp_coco import decode_compressed_coco_rle
from probe_raw_fabrid_anomalib_patchcore import score_modes
from probe_raw_fabrid_physical_field_k0 import (
    CropRef,
    add_crop,
    finalize_fusions,
    make_crop_refs,
)
from probe_raw_patchcore_memory_quantization import embed


@dataclass(frozen=True)
class ParentRecord:
    path: Path
    relative_path: str
    textile_id: int
    scene: str
    role: str
    image_id: int
    is_anomaly: bool
    segmentations: tuple[dict[str, Any], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def load_records(
    config: dict[str, Any], textile_id: int
) -> tuple[list[ParentRecord], dict[str, list[int]]]:
    data_config = config["data"]
    audit = json.loads(Path(data_config["audit"]).read_text(encoding="utf-8"))
    key = str(textile_id)
    if key not in audit["textiles"]:
        raise ValueError(
            f"Textile {textile_id} is not eligible; expected "
            f"{audit['eligible_textiles']}"
        )
    protocol = audit["textiles"][key]
    metadata_path = Path(data_config["metadata_root"]) / f"Textile_{textile_id}" / "dataset.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in metadata["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(
            annotation
        )

    root = Path(data_config["root"])
    records: list[ParentRecord] = []
    split: dict[str, list[int]] = {
        "training_normals": [],
        "calibration_normals": [],
        "evaluation_normals": [],
        "anomalies": [],
    }
    audit_roles = {
        "training_normals": "training_normals",
        "calibration_normals": "calibration_normals",
        "evaluation_normals": "evaluation_normals",
        "evaluation_anomalies": "anomalies",
    }
    expected_geometry = (
        int(data_config["original_width"]),
        int(data_config["original_height"]),
    )
    for audit_role, split_role in audit_roles.items():
        for item in protocol[audit_role]:
            relative_path = str(item["relative_path"])
            path = root / relative_path
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                if image.size != expected_geometry:
                    raise ValueError(
                        f"Unexpected OLP geometry {relative_path}: {image.size}"
                    )
            image_id = int(item["image_id"])
            segmentations = tuple(
                annotation["segmentation"]
                for annotation in annotations_by_image.get(image_id, [])
            )
            is_anomaly = split_role == "anomalies"
            if is_anomaly and not segmentations:
                raise ValueError(f"Missing mask for anomalous parent {relative_path}")
            record = ParentRecord(
                path=path,
                relative_path=relative_path,
                textile_id=textile_id,
                scene=str(item["scene"]),
                role=split_role,
                image_id=image_id,
                is_anomaly=is_anomaly,
                segmentations=segmentations,
            )
            split[split_role].append(len(records))
            records.append(record)
    return records, split


def load_padded_rgb(record: ParentRecord, height: int, width: int) -> np.ndarray:
    image = np.asarray(Image.open(record.path).convert("RGB"), dtype=np.uint8)
    pad_y = height - image.shape[0]
    pad_x = width - image.shape[1]
    if pad_y < 0 or pad_x < 0:
        raise ValueError(f"Parent exceeds padded geometry: {record.relative_path}")
    if pad_y or pad_x:
        image = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), mode="reflect")
    return np.ascontiguousarray(image)


class TrainingCropDataset(Dataset[ImageItem]):
    def __init__(
        self,
        records: list[ParentRecord],
        refs: list[CropRef],
        edge: int,
        height: int,
        width: int,
    ) -> None:
        self.records = records
        self.refs = refs
        self.edge = edge
        self.height = height
        self.width = width
        self.cached_index: int | None = None
        self.cached_image: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, position: int) -> ImageItem:
        ref = self.refs[position]
        if self.cached_index != ref.parent_index:
            self.cached_image = load_padded_rgb(
                self.records[ref.parent_index], self.height, self.width
            )
            self.cached_index = ref.parent_index
        assert self.cached_image is not None
        crop = np.ascontiguousarray(
            self.cached_image[
                ref.y0 : ref.y0 + self.edge, ref.x0 : ref.x0 + self.edge
            ]
        )
        image = torch.from_numpy(crop).permute(2, 0, 1).float().div_(255.0)
        return ImageItem(
            image=image,
            gt_label=torch.tensor(0, dtype=torch.long),
            image_path=self.records[ref.parent_index].path,
        )


class ScoringCropDataset(Dataset[tuple[torch.Tensor, int, int, int, str]]):
    def __init__(
        self,
        records: list[ParentRecord],
        refs: list[CropRef],
        edge: int,
        height: int,
        width: int,
    ) -> None:
        self.records = records
        self.refs = refs
        self.edge = edge
        self.height = height
        self.width = width
        self.cached_index: int | None = None
        self.cached_image: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int, int, str]:
        ref = self.refs[position]
        if self.cached_index != ref.parent_index:
            self.cached_image = load_padded_rgb(
                self.records[ref.parent_index], self.height, self.width
            )
            self.cached_index = ref.parent_index
        assert self.cached_image is not None
        crop = np.ascontiguousarray(
            self.cached_image[
                ref.y0 : ref.y0 + self.edge, ref.x0 : ref.x0 + self.edge
            ]
        )
        image = torch.from_numpy(crop).permute(2, 0, 1).float().div_(255.0)
        return image, ref.parent_index, ref.y0, ref.x0, ref.view


def training_loader(
    records: list[ParentRecord],
    refs: list[CropRef],
    config: dict[str, Any],
    workers: int,
) -> DataLoader:
    data = config["data"]
    return DataLoader(
        TrainingCropDataset(
            records,
            refs,
            int(config["tiling"]["crop_edge"]),
            int(data["padded_height"]),
            int(data["padded_width"]),
        ),
        batch_size=int(config["model"]["train_batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(workers),
        collate_fn=ImageBatch.collate,
    )


def scoring_loader(
    records: list[ParentRecord],
    refs: list[CropRef],
    config: dict[str, Any],
    workers: int,
) -> DataLoader:
    data = config["data"]
    return DataLoader(
        ScoringCropDataset(
            records,
            refs,
            int(config["tiling"]["crop_edge"]),
            int(data["padded_height"]),
            int(data["padded_width"]),
        ),
        batch_size=int(config["model"]["predict_batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(workers),
    )


@torch.inference_mode()
def score_crops(
    model: Patchcore,
    loader: DataLoader,
    refs: list[CropRef],
    device: torch.device,
    config: dict[str, Any],
) -> tuple[dict[int, dict[str, np.ndarray]], float]:
    edge = int(config["tiling"]["crop_edge"])
    stride = int(config["tiling"]["output_stride_pixels"])
    parent_shape = (
        int(config["data"]["padded_height"]) // stride,
        int(config["data"]["padded_width"]) // stride,
    )
    accumulators: dict[int, dict[str, np.ndarray]] = {}
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    processed = 0
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
            size=(edge // stride, edge // stride),
            mode="bilinear",
            align_corners=False,
        )[:, 0].cpu().numpy().astype(np.float32, copy=False)
        for offset, crop_map in enumerate(maps):
            add_crop(
                accumulators,
                int(parents[offset]),
                int(ys[offset]),
                int(xs[offset]),
                str(views[offset]),
                crop_map,
                stride,
                parent_shape,
            )
        processed += images.shape[0]
        if processed == images.shape[0] or processed % 2000 < images.shape[0]:
            print(f"OLP crops {processed}/{len(refs)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return accumulators, time.perf_counter() - started


def downsample_mask(record: ParentRecord, config: dict[str, Any]) -> np.ndarray:
    data = config["data"]
    stride = int(config["tiling"]["output_stride_pixels"])
    original_height = int(data["original_height"])
    original_width = int(data["original_width"])
    padded_height = int(data["padded_height"])
    padded_width = int(data["padded_width"])
    if not record.is_anomaly:
        return np.zeros(
            (math.ceil(original_height / stride), math.ceil(original_width / stride)),
            dtype=bool,
        )
    combined = np.zeros((padded_height, padded_width), dtype=np.uint8)
    for segmentation in record.segmentations:
        current = decode_compressed_coco_rle(segmentation)
        if current.shape != (original_height, original_width):
            raise ValueError(
                f"Mask geometry mismatch {record.relative_path}: {current.shape}"
            )
        combined[:original_height, :original_width] |= current
    tensor = torch.from_numpy(combined.astype(np.float32))[None, None]
    reduced = F.max_pool2d(tensor, kernel_size=stride, stride=stride)[0, 0].numpy() > 0
    return reduced[
        : math.ceil(original_height / stride),
        : math.ceil(original_width / stride),
    ]


def top_fraction_mean(values: np.ndarray, fraction: float) -> float:
    flat = values.reshape(-1)
    count = max(1, int(math.ceil(flat.size * fraction)))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def evaluate_variants(
    fields: dict[str, dict[int, np.ndarray]],
    records: list[ParentRecord],
    split: dict[str, list[int]],
    config: dict[str, Any],
) -> dict[str, Any]:
    stride = int(config["tiling"]["output_stride_pixels"])
    kernel = int(config["score"]["local_mean_kernel_tokens"])
    fraction = float(config["score"]["parent_top_fraction"])
    target_fpr = float(config["score"]["calibration_normal_fpr"])
    valid_height = math.ceil(int(config["data"]["original_height"]) / stride)
    valid_width = math.ceil(int(config["data"]["original_width"]) / stride)
    evaluation = split["evaluation_normals"] + split["anomalies"]
    parent_labels = np.asarray(
        [0] * len(split["evaluation_normals"]) + [1] * len(split["anomalies"]),
        dtype=np.uint8,
    )
    masks = {index: downsample_mask(records[index], config) for index in evaluation}
    result: dict[str, Any] = {}
    for name, parent_fields in fields.items():
        scores: dict[int, np.ndarray] = {}
        parent_scores: dict[int, float] = {}
        for index in split["calibration_normals"] + evaluation:
            raw = parent_fields[index][:valid_height, :valid_width]
            score = score_modes(raw, kernel)["robust_z"]
            scores[index] = score
            parent_scores[index] = top_fraction_mean(score, fraction)
        calibration_scores = np.asarray(
            [parent_scores[index] for index in split["calibration_normals"]],
            dtype=np.float64,
        )
        threshold = float(
            np.quantile(calibration_scores, 1.0 - target_fpr, method="higher")
        )
        evaluation_parent_scores = np.asarray(
            [parent_scores[index] for index in evaluation], dtype=np.float64
        )
        pixel_labels = np.concatenate([masks[index].reshape(-1) for index in evaluation])
        pixel_scores = np.concatenate([scores[index].reshape(-1) for index in evaluation])
        result[name] = {
            "pixel_average_precision": float(
                average_precision_score(pixel_labels, pixel_scores)
            ),
            "pixel_roc_auc": float(roc_auc_score(pixel_labels, pixel_scores)),
            "parent_average_precision": float(
                average_precision_score(parent_labels, evaluation_parent_scores)
            ),
            "parent_roc_auc": float(
                roc_auc_score(parent_labels, evaluation_parent_scores)
            ),
            "source_calibrated_threshold": threshold,
            "evaluation_normal_false_positives": int(
                sum(
                    parent_scores[index] > threshold
                    for index in split["evaluation_normals"]
                )
            ),
            "evaluation_normal_fpr": float(
                np.mean(
                    [
                        parent_scores[index] > threshold
                        for index in split["evaluation_normals"]
                    ]
                )
            ),
            "defect_parent_recall": float(
                np.mean(
                    [
                        parent_scores[index] > threshold
                        for index in split["anomalies"]
                    ]
                )
            ),
            "pixel_positive_cells": int(pixel_labels.sum()),
            "pixel_evaluated_cells": int(pixel_labels.size),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--textile-id", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    records, split = load_records(config, args.textile_id)
    if args.smoke:
        split = {
            "training_normals": split["training_normals"][:1],
            "calibration_normals": split["calibration_normals"][:1],
            "evaluation_normals": split["evaluation_normals"][:1],
            "anomalies": split["anomalies"][:1],
        }

    edge = int(config["tiling"]["crop_edge"])
    shift = int(config["tiling"]["phase_shift_pixels"])
    height = int(config["data"]["padded_height"])
    width = int(config["data"]["padded_width"])
    training_refs = make_crop_refs(
        split["training_normals"], height, width, edge, shift, ["base_grid"]
    )
    if args.smoke:
        training_refs = training_refs[:8]
    scored_indices = sorted(
        split["calibration_normals"]
        + split["evaluation_normals"]
        + split["anomalies"]
    )
    score_refs = make_crop_refs(
        scored_indices,
        height,
        width,
        edge,
        shift,
        list(config["tiling"]["views"]),
    )
    workers = 0 if args.smoke else int(config["model"]["num_workers"])
    train_loader = training_loader(records, training_refs, config, workers)
    predict_loader = scoring_loader(records, score_refs, config, workers)

    seed = int(config["seed"]) + args.textile_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    model_config = config["model"]
    model = Patchcore(
        backbone=str(model_config["backbone"]),
        layers=tuple(model_config["layers"]),
        pre_trained=bool(model_config["pre_trained"]),
        coreset_sampling_ratio=float(model_config["coreset_sampling_ratio"]),
        num_neighbors=int(model_config["num_neighbors"]),
        precision=str(model_config["precision"]),
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    engine = Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
        default_root_dir=args.output.parent / f"lightning_{args.textile_id:02d}",
    )
    started = time.perf_counter()
    engine.fit(model=model, train_dataloaders=train_loader)
    train_seconds = time.perf_counter() - started
    checkpoint = args.output.parent / "checkpoints" / f"textile_{args.textile_id:02d}.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    engine.trainer.save_checkpoint(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accumulators, predict_seconds = score_crops(
        model, predict_loader, score_refs, device, config
    )
    fusion_started = time.perf_counter()
    fields, diagnostics, calibration = finalize_fusions(
        accumulators,
        scored_indices,
        list(config["fusion"]),
        split["calibration_normals"],
        edge // int(config["tiling"]["output_stride_pixels"]),
        shift // int(config["tiling"]["output_stride_pixels"]),
    )
    metrics = evaluate_variants(fields, records, split, config)
    fusion_seconds = time.perf_counter() - fusion_started
    base_crops = (height // edge) * (width // edge)
    result = {
        "schema_version": 1,
        "purpose": "OLP scene-grouped patterned-fabric external validation",
        "claim_boundary": config["claim_boundary"],
        "config": config,
        "run": {
            "textile_id": args.textile_id,
            "smoke": args.smoke,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds,
            "fusion_and_evaluation_seconds": fusion_seconds,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(args.config),
            "audit_sha256": sha256_file(Path(config["data"]["audit"])),
        },
        "counts": {name: len(indices) for name, indices in split.items()},
        "split_paths": {
            name: [records[index].relative_path for index in indices]
            for name, indices in split.items()
        },
        "split_scenes": {
            name: sorted({records[index].scene for index in indices})
            for name, indices in split.items()
        },
        "crop_budget": {
            "training_base_crops": len(training_refs),
            "scored_parents": len(scored_indices),
            "scored_crops": len(score_refs),
            "base_crops_per_parent": base_crops,
            "matched_views_crops_per_parent": len(score_refs) / len(scored_indices),
            "processed_pixel_ratio_to_single_grid": (
                len(score_refs) / len(scored_indices) / base_crops
            ),
        },
        "calibration": calibration,
        "metrics": metrics,
        "diagnostic_observation_count_histogram": {
            str(value): int(
                sum(np.sum(item["count"] == value) for item in diagnostics.values())
            )
            for value in sorted(
                {
                    int(value)
                    for item in diagnostics.values()
                    for value in np.unique(item["count"])
                }
            )
        },
    }
    args.output.write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "textile_id": args.textile_id,
                "smoke": args.smoke,
                "counts": result["counts"],
                "pixel_ap": {
                    name: values["pixel_average_precision"]
                    for name, values in metrics.items()
                },
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
