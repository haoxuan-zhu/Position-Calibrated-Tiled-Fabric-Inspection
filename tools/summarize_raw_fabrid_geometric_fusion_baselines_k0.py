"""Aggregate the preregistered seven-fold geometric-fusion comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FOLDS = [f"Rollo{index}A" for index in range(1, 8)]
VARIANTS = [
    "mean",
    "gaussian",
    "hann",
    "hard_center",
    "context_bias_weighted_mean",
]
METRICS = [
    "pixel_average_precision",
    "parent_average_precision",
    "instance_auc_all",
    "instance_auc_small",
    "instance_auc_elongated",
    "source_parent_recall",
    "target_normal_false_positives",
]


def bootstrap_interval(values: np.ndarray, seed: int, samples: int) -> list[float]:
    generator = np.random.default_rng(seed)
    selections = generator.integers(0, values.size, size=(samples, values.size))
    means = values[selections].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def exact_sign_test(positive: int, negative: int) -> float | None:
    non_ties = positive + negative
    if non_ties == 0:
        return None
    tail = sum(
        math.comb(non_ties, successes)
        for successes in range(min(positive, negative) + 1)
    ) / (2**non_ties)
    return float(min(1.0, 2.0 * tail))


def summarize(results_dir: Path, seed: int, samples: int) -> dict[str, Any]:
    per_fold: dict[str, Any] = {}
    for fold in FOLDS:
        path = results_dir / f"{fold}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["run"]["fold"] != fold or result["run"]["smoke"]:
            raise ValueError(f"Result identity mismatch: {path}")
        if result["frozen_replay_check"]["maximum_absolute_error"] > 1e-8:
            raise ValueError(f"Frozen replay mismatch: {path}")
        missing = set(VARIANTS) - set(result["compact_primary_metrics"])
        if missing:
            raise ValueError(f"Missing variants in {path}: {sorted(missing)}")
        per_fold[fold] = {
            "path": str(path),
            "counts": result["counts"],
            "metrics": {
                variant: result["compact_primary_metrics"][variant]
                for variant in VARIANTS
            },
            "script_sha256": result["run"]["script_sha256"],
            "config_sha256": result["run"]["config_sha256"],
            "checkpoint_sha256": result["run"]["checkpoint_sha256"],
            "reference_sha256": result["frozen_replay_check"]["reference_sha256"],
        }

    macro = {
        variant: {
            metric: float(np.mean([
                per_fold[fold]["metrics"][variant][metric] for fold in FOLDS
            ]))
            for metric in METRICS
        }
        for variant in VARIANTS
    }
    fixed_variants = ["mean", "gaussian", "hann", "hard_center"]
    strongest_fixed = max(
        fixed_variants,
        key=lambda variant: macro[variant]["pixel_average_precision"],
    )
    comparisons: dict[str, Any] = {}
    pcaf = "context_bias_weighted_mean"
    for baseline_offset, baseline in enumerate(fixed_variants):
        comparisons[baseline] = {}
        for metric_offset, metric in enumerate(METRICS):
            deltas = np.asarray([
                per_fold[fold]["metrics"][pcaf][metric]
                - per_fold[fold]["metrics"][baseline][metric]
                for fold in FOLDS
            ], dtype=np.float64)
            wins = int(np.sum(deltas > 0))
            ties = int(np.sum(deltas == 0))
            losses = int(np.sum(deltas < 0))
            comparisons[baseline][metric] = {
                "fold_deltas": {
                    fold: float(value)
                    for fold, value in zip(FOLDS, deltas, strict=True)
                },
                "macro_mean_delta": float(deltas.mean()),
                "macro_median_delta": float(np.median(deltas)),
                "bootstrap_95pct_ci_of_macro_mean": bootstrap_interval(
                    deltas,
                    seed + baseline_offset * len(METRICS) + metric_offset,
                    samples,
                ),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "exact_two_sided_sign_test_p": exact_sign_test(wins, losses),
            }

    return {
        "schema_version": 1,
        "folds": FOLDS,
        "variants": VARIANTS,
        "bootstrap": {
            "seed": seed,
            "samples": samples,
            "resampling_unit": "roll",
        },
        "per_fold": per_fold,
        "macro": macro,
        "strongest_fixed_pixel_ap_baseline": strongest_fixed,
        "pcaf_minus_fixed": comparisons,
        "preregistered_support_condition_passed": (
            macro[pcaf]["pixel_average_precision"]
            > macro[strongest_fixed]["pixel_average_precision"]
        ),
    }


def markdown(summary: dict[str, Any]) -> str:
    labels = {
        "mean": "Mean",
        "gaussian": "Gaussian",
        "hann": "Hann",
        "hard_center": "Hard-center",
        "context_bias_weighted_mean": "PCAF",
    }
    lines = [
        "# RAW-FABRID deterministic geometric-fusion audit",
        "",
        "> Frozen seven-roll comparison. All methods receive the same 73 crop maps per parent.",
        "",
        "## Pixel AP by held-out roll",
        "",
        "| Roll | Mean | Gaussian | Hann | Hard-center | PCAF |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fold in FOLDS:
        metrics = summary["per_fold"][fold]["metrics"]
        lines.append(
            f"| {fold} | "
            + " | ".join(
                f"{metrics[variant]['pixel_average_precision']:.4f}"
                for variant in VARIANTS
            )
            + " |"
        )
    lines.extend([
        "",
        "## Roll-macro metrics",
        "",
        "| Fusion | Pixel AP | Parent AP | All I-AUC | Small I-AUC | Elongated I-AUC | Recall | N-FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in VARIANTS:
        metric = summary["macro"][variant]
        lines.append(
            f"| {labels[variant]} | {metric['pixel_average_precision']:.4f} | "
            f"{metric['parent_average_precision']:.4f} | {metric['instance_auc_all']:.4f} | "
            f"{metric['instance_auc_small']:.4f} | {metric['instance_auc_elongated']:.4f} | "
            f"{metric['source_parent_recall']:.4f} | "
            f"{metric['target_normal_false_positives']:.2f} |"
        )
    lines.extend([
        "",
        "## PCAF minus deterministic baselines",
        "",
        "| Baseline | Pixel AP delta | wins/ties/losses | 95% roll-bootstrap CI | sign p |",
        "|---|---:|---:|---:|---:|",
    ])
    for baseline in ("mean", "gaussian", "hann", "hard_center"):
        item = summary["pcaf_minus_fixed"][baseline]["pixel_average_precision"]
        interval = item["bootstrap_95pct_ci_of_macro_mean"]
        sign_p = item["exact_two_sided_sign_test_p"]
        lines.append(
            f"| {labels[baseline]} | {item['macro_mean_delta']:+.4f} | "
            f"{item['wins']}/{item['ties']}/{item['losses']} | "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}] | "
            f"{sign_p:.6f} |"
        )
    lines.extend([
        "",
        f"Strongest fixed Pixel-AP baseline: `{summary['strongest_fixed_pixel_ap_baseline']}`.",
        f"Preregistered support condition passed: `{summary['preregistered_support_condition_passed']}`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    summary = summarize(args.results_dir, args.seed, args.bootstrap_samples)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = markdown(summary)
    args.output_markdown.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
