#!/usr/bin/env python3
"""Recover exact overlap relationships between released ISP-AD patches.

The paper states that 256x256 patches were extracted with stride 160.  If the
release preserves those pixels, horizontal or vertical neighbours share an
exact 96-pixel strip.  This audit hashes those strips, builds an undirected
overlap graph per modality, and reports connected components without modifying
the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sample:
    path: Path
    modality: str
    split: str
    label: str


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=160)
    parser.add_argument(
        "--max-hash-multiplicity",
        type=int,
        default=16,
        help="Ignore non-unique border hashes that would create ambiguous cliques.",
    )
    return parser.parse_args()


def discover(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for modality in ("ASM", "LSM_1", "LSM_2"):
        modality_root = root / modality
        for split in ("train", "test"):
            split_root = modality_root / split
            if not split_root.is_dir():
                continue
            for label_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
                # good_reduced is an exact subset of train/good and would create
                # artificial duplicate nodes in this provenance graph.
                if label_dir.name == "good_reduced":
                    continue
                for path in sorted(label_dir.glob("*.png")):
                    samples.append(Sample(path, modality, split, label_dir.name))
    return samples


def informative_hash(array: np.ndarray) -> str | None:
    if np.unique(array).size <= 1:
        return None
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def border_hashes(path: Path, overlap: int, patch_size: int) -> dict[str, str | None]:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.shape[:2] != (patch_size, patch_size):
        raise ValueError(f"Unexpected image size {array.shape[:2]}: {path}")
    return {
        "left": informative_hash(np.ascontiguousarray(array[:, :overlap, ...])),
        "right": informative_hash(np.ascontiguousarray(array[:, -overlap:, ...])),
        "top": informative_hash(np.ascontiguousarray(array[:overlap, :, ...])),
        "bottom": informative_hash(np.ascontiguousarray(array[-overlap:, :, ...])),
    }


def component_summary(
    samples: list[Sample], indices: list[int], edges: set[tuple[int, int, str]]
) -> dict[str, object]:
    local = {global_index: local_index for local_index, global_index in enumerate(indices)}
    union_find = UnionFind(len(indices))
    for a, b, _ in edges:
        union_find.union(local[a], local[b])
    groups: dict[int, list[int]] = defaultdict(list)
    for global_index in indices:
        groups[union_find.find(local[global_index])].append(global_index)
    components = sorted(groups.values(), key=lambda members: (-len(members), members[0]))
    sizes = [len(members) for members in components]
    adjacency: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for a, b, orientation in edges:
        dx, dy = (1, 0) if orientation == "horizontal" else (0, 1)
        adjacency[a].append((b, dx, dy))
        adjacency[b].append((a, -dx, -dy))

    grid_summaries: list[dict[str, object]] = []
    total_coordinate_conflicts = 0
    total_coordinate_collisions = 0
    for members in components:
        coordinates: dict[int, tuple[int, int]] = {members[0]: (0, 0)}
        queue = [members[0]]
        conflicts = 0
        while queue:
            current = queue.pop()
            x, y = coordinates[current]
            for neighbour, dx, dy in adjacency[current]:
                proposed = (x + dx, y + dy)
                if neighbour not in coordinates:
                    coordinates[neighbour] = proposed
                    queue.append(neighbour)
                elif coordinates[neighbour] != proposed:
                    conflicts += 1
        coordinate_collisions = len(coordinates) - len(set(coordinates.values()))
        total_coordinate_conflicts += conflicts
        total_coordinate_collisions += coordinate_collisions
        xs = [coordinate[0] for coordinate in coordinates.values()]
        ys = [coordinate[1] for coordinate in coordinates.values()]
        width = max(xs) - min(xs) + 1
        height = max(ys) - min(ys) + 1
        grid_summaries.append(
            {
                "width": width,
                "height": height,
                "occupancy": len(set(coordinates.values())) / (width * height),
                "coordinate_conflicts": conflicts,
                "coordinate_collisions": coordinate_collisions,
            }
        )
    return {
        "sample_count": len(indices),
        "component_count": len(components),
        "non_singleton_component_count": sum(size > 1 for size in sizes),
        "samples_in_non_singleton_components": sum(size for size in sizes if size > 1),
        "largest_component_size": max(sizes, default=0),
        "coordinate_conflict_count": total_coordinate_conflicts,
        "coordinate_collision_count": total_coordinate_collisions,
        "component_size_histogram": {
            str(size): count for size, count in sorted(Counter(sizes).items())
        },
        "largest_components": [
            {
                "size": len(members),
                "splits": dict(Counter(samples[index].split for index in members)),
                "labels": dict(Counter(samples[index].label for index in members)),
                "grid": grid_summaries[component_index],
                "files": [samples[index].path.name for index in members[:24]],
            }
            for component_index, members in enumerate(components[:10])
            if len(members) > 1
        ],
    }


def candidate_pairs(
    source: dict[str, list[int]], target: dict[str, list[int]], max_multiplicity: int
) -> tuple[Iterable[tuple[int, int]], int]:
    ambiguous = 0
    pairs: list[tuple[int, int]] = []
    for digest, left_indices in source.items():
        right_indices = target.get(digest, [])
        multiplicity = len(left_indices) + len(right_indices)
        if not right_indices:
            continue
        if multiplicity > max_multiplicity:
            ambiguous += 1
            continue
        pairs.extend((a, b) for a in left_indices for b in right_indices if a != b)
    return pairs, ambiguous


def assignment_records(
    samples: list[Sample],
    indices: list[int],
    edges: set[tuple[int, int, str]],
    dataset_root: Path,
) -> list[dict[str, object]]:
    """Return stable component IDs and overlap-grid coordinates for every sample."""

    local = {global_index: local_index for local_index, global_index in enumerate(indices)}
    union_find = UnionFind(len(indices))
    adjacency: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for a, b, orientation in edges:
        union_find.union(local[a], local[b])
        dx, dy = (1, 0) if orientation == "horizontal" else (0, 1)
        adjacency[a].append((b, dx, dy))
        adjacency[b].append((a, -dx, -dy))

    groups: dict[int, list[int]] = defaultdict(list)
    for global_index in indices:
        groups[union_find.find(local[global_index])].append(global_index)
    components = sorted(groups.values(), key=lambda members: min(members))

    records: list[dict[str, object]] = []
    for component_index, members in enumerate(components):
        coordinates: dict[int, tuple[int, int]] = {members[0]: (0, 0)}
        queue = [members[0]]
        while queue:
            current = queue.pop()
            x, y = coordinates[current]
            for neighbour, dx, dy in adjacency[current]:
                proposed = (x + dx, y + dy)
                if neighbour not in coordinates:
                    coordinates[neighbour] = proposed
                    queue.append(neighbour)
                elif coordinates[neighbour] != proposed:
                    raise RuntimeError("Coordinate conflict escaped component audit")
        for global_index in sorted(members):
            sample = samples[global_index]
            x, y = coordinates[global_index]
            records.append(
                {
                    "relative_path": sample.path.relative_to(dataset_root).as_posix(),
                    "split": sample.split,
                    "label": sample.label,
                    "component_id": component_index,
                    "component_size": len(members),
                    "grid_x": x,
                    "grid_y": y,
                }
            )
    return records


def main() -> None:
    args = parse_args()
    overlap = args.patch_size - args.stride
    if overlap <= 0:
        raise ValueError("patch size must exceed stride")

    samples = discover(args.dataset_root)
    hashes: list[dict[str, str | None]] = []
    for sample in samples:
        hashes.append(border_hashes(sample.path, overlap, args.patch_size))

    report: dict[str, object] = {
        "dataset_root": str(args.dataset_root.resolve()),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "overlap": overlap,
        "excluded_directory": "train/good_reduced (exact subset of train/good)",
        "modalities": {},
    }

    for modality in ("ASM", "LSM_1", "LSM_2"):
        indices = [i for i, sample in enumerate(samples) if sample.modality == modality]
        border_maps: dict[str, dict[str, list[int]]] = {
            side: defaultdict(list) for side in ("left", "right", "top", "bottom")
        }
        null_hashes = Counter()
        for index in indices:
            for side, digest in hashes[index].items():
                if digest is None:
                    null_hashes[side] += 1
                else:
                    border_maps[side][digest].append(index)

        horizontal, ambiguous_horizontal = candidate_pairs(
            border_maps["right"], border_maps["left"], args.max_hash_multiplicity
        )
        vertical, ambiguous_vertical = candidate_pairs(
            border_maps["bottom"], border_maps["top"], args.max_hash_multiplicity
        )
        edges: set[tuple[int, int, str]] = set()
        edges.update((a, b, "horizontal") for a, b in horizontal)
        edges.update((a, b, "vertical") for a, b in vertical)

        cross_split = [edge for edge in edges if samples[edge[0]].split != samples[edge[1]].split]
        cross_label = [edge for edge in edges if samples[edge[0]].label != samples[edge[1]].label]
        modality_report = component_summary(samples, indices, edges)
        modality_report.update(
            {
                "edge_count": len(edges),
                "horizontal_edge_count": sum(edge[2] == "horizontal" for edge in edges),
                "vertical_edge_count": sum(edge[2] == "vertical" for edge in edges),
                "cross_split_edge_count": len(cross_split),
                "cross_label_edge_count": len(cross_label),
                "uninformative_border_count": dict(null_hashes),
                "ambiguous_hash_bucket_count": {
                    "horizontal": ambiguous_horizontal,
                    "vertical": ambiguous_vertical,
                },
                "by_split": {},
                "by_split_label": {},
                "assignments": assignment_records(
                    samples, indices, edges, args.dataset_root
                ),
            }
        )
        for split in ("train", "test"):
            split_indices = [index for index in indices if samples[index].split == split]
            split_set = set(split_indices)
            split_edges = {
                edge for edge in edges if edge[0] in split_set and edge[1] in split_set
            }
            modality_report["by_split"][split] = component_summary(
                samples, split_indices, split_edges
            )
            modality_report["by_split_label"][split] = {}
            for label in sorted({samples[index].label for index in split_indices}):
                label_indices = [
                    index for index in split_indices if samples[index].label == label
                ]
                label_set = set(label_indices)
                label_edges = {
                    edge for edge in split_edges if edge[0] in label_set and edge[1] in label_set
                }
                modality_report["by_split_label"][split][label] = component_summary(
                    samples, label_indices, label_edges
                )
        report["modalities"][modality] = modality_report

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
