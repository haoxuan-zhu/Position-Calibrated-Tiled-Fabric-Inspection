"""Aggregate the seven-fold same-location fixed-effect audit.

The script keeps every held-out roll visible and derives all manuscript numbers
from the immutable per-fold JSON outputs.  The independent unit for paired
accuracy statistics is a fabric roll, not a crop or a same-location pair.
"""

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
    "context_bias_equal_mean",
    "context_bias_weighted_mean",
    "fixed_effect_equal_mean",
    "fixed_effect_weighted_mean",
]
PRIMARY_METRICS = ["pixel_average_precision", "instance_auc_all"]


def exact_two_sided_sign_test(positive: int, negative: int) -> float | None:
    non_ties = positive + negative
    if non_ties == 0:
        return None
    lower_tail = sum(
        math.comb(non_ties, successes)
        for successes in range(min(positive, negative) + 1)
    ) / (2**non_ties)
    return float(min(1.0, 2.0 * lower_tail))


def bootstrap_interval(values: np.ndarray, seed: int, samples: int) -> list[float]:
    generator = np.random.default_rng(seed)
    selections = generator.integers(0, values.size, size=(samples, values.size))
    means = values[selections].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def paired_summary(
    left: list[float], right: list[float], seed: int, samples: int
) -> dict[str, Any]:
    deltas = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    wins = int(np.sum(deltas > 0))
    ties = int(np.sum(deltas == 0))
    losses = int(np.sum(deltas < 0))
    return {
        "fold_deltas": {
            fold: float(delta) for fold, delta in zip(FOLDS, deltas, strict=True)
        },
        "macro_mean_delta": float(deltas.mean()),
        "macro_median_delta": float(np.median(deltas)),
        "bootstrap_95pct_ci_of_macro_mean": bootstrap_interval(deltas, seed, samples),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "exact_two_sided_sign_test_p": exact_two_sided_sign_test(wins, losses),
    }


def load_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for fold in FOLDS:
        path = results_dir / f"{fold}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing fixed-effect audit: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        run = result.get("run", {})
        if run.get("fold") != fold or not run.get("checkpoint_sha256"):
            raise ValueError(f"Result identity mismatch: {path}")
        missing = set(VARIANTS) - set(result.get("compact_primary_metrics", {}))
        if missing:
            raise ValueError(f"Missing variants in {path}: {sorted(missing)}")
        results[fold] = result
    return results


def aggregate(
    results: dict[str, dict[str, Any]], seed: int, samples: int
) -> dict[str, Any]:
    per_fold: dict[str, Any] = {}
    for fold in FOLDS:
        result = results[fold]
        calibration = result["calibration"]
        pairs = calibration["overlap_pair_differences"]
        metrics = result["compact_primary_metrics"]
        per_fold[fold] = {
            "paired_differences": int(pairs["raw"]["paired_differences"]),
            "pair_rmse": {
                "raw": float(pairs["raw"]["rmse"]),
                "pooled_corrected": float(pairs["pooled_corrected"]["rmse"]),
                "fixed_effect_corrected": float(pairs["fixed_effect_corrected"]["rmse"]),
            },
            "pair_absolute_median": {
                "raw": float(pairs["raw"]["absolute_median"]),
                "pooled_corrected": float(pairs["pooled_corrected"]["absolute_median"]),
                "fixed_effect_corrected": float(
                    pairs["fixed_effect_corrected"]["absolute_median"]
                ),
            },
            "pooled_vs_anchored": {
                "pearson": float(calibration["pooled_vs_anchored_pearson"]),
                "spearman": float(calibration["pooled_vs_anchored_spearman"]),
            },
            "within_component_pearson": {
                "minimum": float(
                    calibration["within_component_pooled_vs_fixed_pearson_minimum"]
                ),
                "median": float(
                    calibration["within_component_pooled_vs_fixed_pearson_median"]
                ),
            },
            "source_roll_fixed_effect_stability": calibration["source_roll_stability"][
                "fixed_effect_anchored"
            ],
            "metrics": {
                variant: {
                    metric: float(metrics[variant][metric])
                    for metric in PRIMARY_METRICS
                }
                for variant in VARIANTS
            },
        }

    aggregate_fields: dict[str, Any] = {
        "paired_differences_total": int(
            sum(per_fold[fold]["paired_differences"] for fold in FOLDS)
        )
    }
    for family in ["pair_rmse", "pair_absolute_median"]:
        aggregate_fields[family] = {}
        for variant in ["raw", "pooled_corrected", "fixed_effect_corrected"]:
            aggregate_fields[family][variant] = describe(
                [per_fold[fold][family][variant] for fold in FOLDS]
            )

    raw_rmse = np.asarray(
        [per_fold[fold]["pair_rmse"]["raw"] for fold in FOLDS], dtype=np.float64
    )
    for variant in ["pooled_corrected", "fixed_effect_corrected"]:
        corrected = np.asarray(
            [per_fold[fold]["pair_rmse"][variant] for fold in FOLDS],
            dtype=np.float64,
        )
        aggregate_fields[f"{variant}_rmse_fraction_reduced"] = describe(
            ((raw_rmse - corrected) / raw_rmse).tolist()
        )

    aggregate_fields["pooled_vs_anchored_pearson"] = describe(
        [per_fold[fold]["pooled_vs_anchored"]["pearson"] for fold in FOLDS]
    )
    aggregate_fields["pooled_vs_anchored_spearman"] = describe(
        [per_fold[fold]["pooled_vs_anchored"]["spearman"] for fold in FOLDS]
    )
    aggregate_fields["within_component_pearson_median"] = describe(
        [per_fold[fold]["within_component_pearson"]["median"] for fold in FOLDS]
    )
    aggregate_fields["within_component_pearson_minimum"] = describe(
        [per_fold[fold]["within_component_pearson"]["minimum"] for fold in FOLDS]
    )
    aggregate_fields["source_roll_stability_pearson_median"] = describe(
        [
            float(
                per_fold[fold]["source_roll_fixed_effect_stability"]["pearson_median"]
            )
            for fold in FOLDS
        ]
    )
    aggregate_fields["source_roll_stability_pearson_minimum"] = describe(
        [
            float(
                per_fold[fold]["source_roll_fixed_effect_stability"]["pearson_minimum"]
            )
            for fold in FOLDS
        ]
    )

    metric_macro: dict[str, Any] = {}
    for variant in VARIANTS:
        metric_macro[variant] = {
            metric: describe(
                [per_fold[fold]["metrics"][variant][metric] for fold in FOLDS]
            )
            for metric in PRIMARY_METRICS
        }

    comparisons: dict[str, Any] = {}
    for offset, (name, left_variant, right_variant) in enumerate(
        [
            ("fixed_effect_weighted_minus_mean", "fixed_effect_weighted_mean", "mean"),
            (
                "fixed_effect_weighted_minus_pooled_weighted",
                "fixed_effect_weighted_mean",
                "context_bias_weighted_mean",
            ),
        ]
    ):
        comparisons[name] = {}
        for metric_offset, metric in enumerate(PRIMARY_METRICS):
            comparisons[name][metric] = paired_summary(
                [per_fold[fold]["metrics"][left_variant][metric] for fold in FOLDS],
                [per_fold[fold]["metrics"][right_variant][metric] for fold in FOLDS],
                seed + 10 * offset + metric_offset,
                samples,
            )

    return {
        "schema_version": 1,
        "purpose": "Seven-fold content-controlled crop-coordinate mechanism audit",
        "folds": FOLDS,
        "independent_accuracy_unit": "held-out fabric roll",
        "identifiability": {
            "relative_coordinate_bins": 1024,
            "connected_components": 64,
            "identifiable_contrasts": 960,
            "component_constants_requiring_pooled_anchor": 64,
        },
        "bootstrap": {"seed": seed, "samples": samples, "unit": "roll"},
        "per_fold": per_fold,
        "aggregate": aggregate_fields,
        "macro": metric_macro,
        "paired_comparisons": comparisons,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RAW-FABRID 七折固定效应审计",
        "",
        "> 自动生成。准确率统计单元为布卷；同位置配对数只描述机制拟合规模，不充当独立样本数。",
        "",
        "| Roll | raw pair RMSE | pooled RMSE | fixed-effect RMSE | map Pearson | Mean Pixel AP | pooled Pixel AP | FE Pixel AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in FOLDS:
        item = summary["per_fold"][fold]
        rmse = item["pair_rmse"]
        metrics = item["metrics"]
        lines.append(
            f"| {fold} | {rmse['raw']:.4f} | {rmse['pooled_corrected']:.4f} | "
            f"{rmse['fixed_effect_corrected']:.4f} | "
            f"{item['pooled_vs_anchored']['pearson']:.4f} | "
            f"{metrics['mean']['pixel_average_precision']:.4f} | "
            f"{metrics['context_bias_weighted_mean']['pixel_average_precision']:.4f} | "
            f"{metrics['fixed_effect_weighted_mean']['pixel_average_precision']:.4f} |"
        )
    comparison = summary["paired_comparisons"]["fixed_effect_weighted_minus_mean"]
    pixel = comparison["pixel_average_precision"]
    lines.extend(
        [
            "",
            f"FE vs Mean Pixel AP: macro delta {pixel['macro_mean_delta']:+.6f}; "
            f"wins/ties/losses {pixel['wins']}/{pixel['ties']}/{pixel['losses']}; "
            f"exact sign p={pixel['exact_two_sided_sign_test_p']:.6f}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    summary = aggregate(
        load_results(args.results_dir), args.seed, args.bootstrap_samples
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rendered = markdown(summary)
    args.output_markdown.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
