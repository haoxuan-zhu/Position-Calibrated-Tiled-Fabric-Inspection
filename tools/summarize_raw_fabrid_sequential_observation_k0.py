"""Summarize the frozen seven-roll sequential-observation confirmation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = (
    "base_y_pcaf",
    "matched_uniform_sparse_pcaf",
    "sequential_pcaf",
    "full_pcaf",
)
METRICS = (
    "parent_average_precision",
    "pixel_average_precision",
    "source_parent_recall",
    "target_normal_false_positives",
    "instance_auc_all",
    "instance_auc_small",
    "instance_auc_elongated",
    "instance_recall_all",
    "instance_recall_small",
    "instance_recall_elongated",
)
REFERENCES = (
    "full_pcaf",
    "matched_uniform_sparse_pcaf",
    "base_y_pcaf",
)


def paired_bootstrap(
    values: np.ndarray, generator: np.random.Generator, replicates: int
) -> tuple[float, float]:
    samples = generator.choice(values, size=(replicates, values.size), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def exact_sign_p(wins: int, losses: int) -> float | None:
    count = wins + losses
    if count == 0:
        return None
    tail = min(wins, losses)
    probability = 2.0 * sum(math.comb(count, k) for k in range(tail + 1)) / 2**count
    return min(probability, 1.0)


def finite_summary(
    values: np.ndarray, generator: np.random.Generator, replicates: int
) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    lower, upper = paired_bootstrap(finite, generator, replicates)
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
        "roll_bootstrap_95_ci": [lower, upper],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--replicates", type=int, default=100000)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("Rollo*A.json"))
    if len(paths) != 7:
        raise RuntimeError(f"Expected seven fold results, found {len(paths)}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    folds = [str(row["run"]["fold"]) for row in rows]
    if folds != [f"Rollo{index}A" for index in range(1, 8)]:
        raise RuntimeError(f"Unexpected fold order: {folds}")
    generator = np.random.default_rng(args.seed)

    macro: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        macro[variant] = {
            metric: float(np.mean([
                row["compact_primary_metrics"][variant][metric] for row in rows
            ]))
            for metric in METRICS
        }

    deltas: dict[str, dict[str, Any]] = {}
    for reference in REFERENCES:
        by_metric: dict[str, Any] = {}
        for metric in METRICS:
            values = np.asarray([
                row["compact_primary_metrics"]["sequential_pcaf"][metric]
                - row["compact_primary_metrics"][reference][metric]
                for row in rows
            ], dtype=np.float64)
            wins = int(np.sum(values > 0))
            ties = int(np.sum(values == 0))
            losses = int(np.sum(values < 0))
            by_metric[metric] = {
                **finite_summary(values, generator, args.replicates),
                "per_roll": values.tolist(),
                "wins_ties_losses": [wins, ties, losses],
                "exact_two_sided_sign_p": exact_sign_p(wins, losses),
            }
        deltas[reference] = by_metric

    budget_values = np.asarray([
        row["counts"]["evaluation_online_budget_reduction_vs_full"] for row in rows
    ], dtype=np.float64)
    crop_values = np.asarray([
        row["counts"]["evaluation_online_crops_per_parent_mean"] for row in rows
    ], dtype=np.float64)
    selection_groups = ("calibration_normals", "evaluation_normals", "anomalies")
    selection = {
        group: {
            "optional_crop_count_roll_macro": float(np.mean([
                row["selector"]["selection_summary"][group]["optional_crop_count_mean"]
                for row in rows
            ])),
            "optional_crop_fraction_roll_macro": float(np.mean([
                row["selector"]["selection_summary"][group]["optional_crop_fraction_mean"]
                for row in rows
            ])),
        }
        for group in selection_groups
    }
    gate_names = list(rows[0]["fixed_candidate_gate"]["checks"])
    gate_counts = {
        gate: int(sum(bool(row["fixed_candidate_gate"]["checks"][gate]) for row in rows))
        for gate in gate_names
    }

    per_roll = []
    for row in rows:
        fold = str(row["run"]["fold"])
        sequential = row["compact_primary_metrics"]["sequential_pcaf"]
        per_roll.append({
            "fold": fold,
            "evaluation_online_crops_per_parent_mean": row["counts"][
                "evaluation_online_crops_per_parent_mean"
            ],
            "budget_reduction_vs_full": row["counts"][
                "evaluation_online_budget_reduction_vs_full"
            ],
            "sequential_metrics": sequential,
            "pixel_ap_delta_vs_full": sequential["pixel_average_precision"]
            - row["compact_primary_metrics"]["full_pcaf"]["pixel_average_precision"],
            "pixel_ap_delta_vs_matched_uniform": sequential["pixel_average_precision"]
            - row["compact_primary_metrics"]["matched_uniform_sparse_pcaf"][
                "pixel_average_precision"
            ],
            "pixel_ap_delta_vs_base_y": sequential["pixel_average_precision"]
            - row["compact_primary_metrics"]["base_y_pcaf"]["pixel_average_precision"],
            "gate_passed": bool(row["fixed_candidate_gate"]["passed"]),
            "gate_checks": row["fixed_candidate_gate"]["checks"],
            "full_reference_max_absolute_compact_difference": row[
                "full_reference_max_absolute_compact_difference"
            ],
        })

    result = {
        "schema_version": 1,
        "seed": args.seed,
        "bootstrap_replicates": args.replicates,
        "folds": folds,
        "rolls": len(rows),
        "integrity": {
            "maximum_full_reference_replay_difference": float(max(
                row["full_reference_max_absolute_compact_difference"] for row in rows
            )),
            "unique_probe_script_hashes": sorted({
                str(row["run"]["script_sha256"]) for row in rows
            }),
            "unique_config_hashes": sorted({
                str(row["run"]["config_sha256"]) for row in rows
            }),
        },
        "budget": {
            "online_crops_per_parent": finite_summary(
                crop_values, generator, args.replicates
            ),
            "reduction_vs_full": finite_summary(
                budget_values, generator, args.replicates
            ),
            "full_crops_per_parent": 73,
        },
        "selection": selection,
        "macro_metrics": macro,
        "sequential_deltas": deltas,
        "gate_pass_counts_out_of_seven": gate_counts,
        "complete_fold_gate_passes": int(sum(
            bool(row["fixed_candidate_gate"]["passed"]) for row in rows
        )),
        "per_roll": per_roll,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "integrity": result["integrity"],
        "budget": result["budget"],
        "pixel_ap_vs_full": result["sequential_deltas"]["full_pcaf"][
            "pixel_average_precision"
        ],
        "pixel_ap_vs_matched": result["sequential_deltas"][
            "matched_uniform_sparse_pcaf"
        ]["pixel_average_precision"],
        "pixel_ap_vs_base_y": result["sequential_deltas"]["base_y_pcaf"][
            "pixel_average_precision"
        ],
        "gate_pass_counts": result["gate_pass_counts_out_of_seven"],
        "complete_fold_gate_passes": result["complete_fold_gate_passes"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
