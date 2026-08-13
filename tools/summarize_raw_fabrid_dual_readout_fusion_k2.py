"""Aggregate the preregistered K2 development/confirmation experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEVELOPMENT_FOLD = "Rollo4A"
CONFIRMATION_FOLDS = [
    "Rollo1A",
    "Rollo2A",
    "Rollo3A",
    "Rollo5A",
    "Rollo6A",
    "Rollo7A",
]
ALL_FOLDS = [f"Rollo{index}A" for index in range(1, 8)]
CANDIDATE = "dual_gaussian_bias_localization_pcaf_alarm"
PCAF = "context_bias_weighted_mean"
REFERENCES = ["gaussian", "hann", PCAF]
VARIANTS = [*REFERENCES, CANDIDATE]
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
    choices = generator.integers(0, values.size, size=(samples, values.size))
    means = values[choices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def exact_sign_test(positive: int, negative: int) -> float | None:
    non_ties = positive + negative
    if non_ties == 0:
        return None
    tail = sum(
        math.comb(non_ties, value)
        for value in range(min(positive, negative) + 1)
    ) / (2**non_ties)
    return float(min(1.0, 2.0 * tail))


def macro(
    per_fold: dict[str, Any], folds: list[str]
) -> dict[str, dict[str, float]]:
    return {
        variant: {
            metric: float(np.mean([
                per_fold[fold]["metrics"][variant][metric] for fold in folds
            ]))
            for metric in METRICS
        }
        for variant in VARIANTS
    }


def comparisons(
    per_fold: dict[str, Any], folds: list[str], seed: int, samples: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for reference_offset, reference in enumerate(REFERENCES):
        output[reference] = {}
        for metric_offset, metric in enumerate(METRICS):
            values = np.asarray([
                per_fold[fold]["metrics"][CANDIDATE][metric]
                - per_fold[fold]["metrics"][reference][metric]
                for fold in folds
            ], dtype=np.float64)
            wins = int(np.sum(values > 0))
            ties = int(np.sum(values == 0))
            losses = int(np.sum(values < 0))
            output[reference][metric] = {
                "fold_deltas": {
                    fold: float(value)
                    for fold, value in zip(folds, values, strict=True)
                },
                "macro_mean_delta": float(values.mean()),
                "bootstrap_95pct_ci_of_macro_mean": bootstrap_interval(
                    values,
                    seed + reference_offset * len(METRICS) + metric_offset,
                    samples,
                ),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "exact_two_sided_sign_test_p": exact_sign_test(wins, losses),
            }
    return output


def summarize(results_dir: Path, seed: int, samples: int) -> dict[str, Any]:
    per_fold: dict[str, Any] = {}
    script_hashes: set[str] = set()
    config_hashes: set[str] = set()
    dependency_hashes: set[str] = set()
    config: dict[str, Any] | None = None
    for fold in ALL_FOLDS:
        path = results_dir / f"{fold}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["run"]["fold"] != fold:
            raise ValueError(f"Result identity mismatch: {path}")
        if result["counts"]["crops_per_parent"] != 73:
            raise ValueError(f"Crop-count mismatch: {path}")
        if result["frozen_replay_check"]["maximum_absolute_error"] > 1e-8:
            raise ValueError(f"Frozen replay mismatch: {path}")
        identity = result["dual_readout_identity"]
        if not identity["exact_pcaf_parent_readout"]:
            raise ValueError(f"PCAF parent identity failed: {path}")
        if not identity["exact_gaussian_bias_localization_readout"]:
            raise ValueError(f"Localization identity failed: {path}")
        identity_errors = [
            float(value)
            for key, value in identity.items()
            if key.startswith("maximum_")
        ]
        if any(value != 0.0 for value in identity_errors):
            raise ValueError(f"Nonzero readout identity error: {path}")
        missing = set(VARIANTS) - set(result["compact_primary_metrics"])
        if missing:
            raise ValueError(f"Missing variants in {path}: {sorted(missing)}")
        if fold == DEVELOPMENT_FOLD:
            development = result["development_reproduction"]
            if development is None or not development["passed"]:
                raise ValueError(f"Development reproduction failed: {path}")
        elif result["development_reproduction"] is not None:
            raise ValueError(f"Confirmation fold marked as development: {path}")
        script_hashes.add(str(result["run"]["script_sha256"]))
        config_hashes.add(str(result["run"]["config_sha256"]))
        dependency_hashes.add(json.dumps(
            result["run"]["dependency_sha256"], sort_keys=True
        ))
        config = result["config"] if config is None else config
        if result["config"] != config:
            raise ValueError(f"Embedded configuration differs: {path}")
        per_fold[fold] = {
            "path": str(path),
            "counts": result["counts"],
            "metrics": {
                variant: result["compact_primary_metrics"][variant]
                for variant in VARIANTS
            },
            "dual_readout_identity": identity,
            "script_sha256": result["run"]["script_sha256"],
            "config_sha256": result["run"]["config_sha256"],
            "checkpoint_sha256": result["run"]["checkpoint_sha256"],
            "dependency_sha256": result["run"]["dependency_sha256"],
            "reference_sha256": result["frozen_replay_check"]["reference_sha256"],
        }
    if (
        len(script_hashes) != 1
        or len(config_hashes) != 1
        or len(dependency_hashes) != 1
    ):
        raise ValueError(
            "All folds must use one frozen script, configuration, and dependency set"
        )
    assert config is not None

    confirmation_macro = macro(per_fold, CONFIRMATION_FOLDS)
    all_macro = macro(per_fold, ALL_FOLDS)
    confirmation_comparisons = comparisons(
        per_fold, CONFIRMATION_FOLDS, seed, samples
    )
    all_comparisons = comparisons(
        per_fold, ALL_FOLDS, seed + 1000, samples
    )
    specification = config["confirmation"]
    strongest_pixel = max(
        REFERENCES,
        key=lambda name: confirmation_macro[name]["pixel_average_precision"],
    )
    strongest_auc = max(
        REFERENCES,
        key=lambda name: confirmation_macro[name]["instance_auc_all"],
    )
    fold_wins = sum(
        per_fold[fold]["metrics"][CANDIDATE]["pixel_average_precision"]
        > max(
            per_fold[fold]["metrics"][reference]["pixel_average_precision"]
            for reference in REFERENCES
        )
        for fold in CONFIRMATION_FOLDS
    )
    parent_identity = all(
        all(
            per_fold[fold]["metrics"][CANDIDATE][metric]
            == per_fold[fold]["metrics"][PCAF][metric]
            for metric in (
                "parent_average_precision",
                "source_parent_recall",
                "target_normal_false_positives",
            )
        )
        for fold in ALL_FOLDS
    )
    checks = {
        "macro_pixel_ap": (
            confirmation_macro[CANDIDATE]["pixel_average_precision"]
            - confirmation_macro[strongest_pixel]["pixel_average_precision"]
            >= float(
                specification[
                    "minimum_macro_pixel_ap_gain_over_strongest_reference"
                ]
            )
        ),
        "macro_instance_auc": (
            confirmation_macro[CANDIDATE]["instance_auc_all"]
            >= confirmation_macro[strongest_auc]["instance_auc_all"]
            - float(
                specification[
                    "maximum_macro_instance_auc_loss_from_strongest_reference"
                ]
            )
        ),
        "fold_wins": (
            fold_wins
            >= int(specification["minimum_fold_wins_against_each_fold_best_reference"])
        ),
        "exact_pcaf_parent_readout": parent_identity,
    }
    strong_pareto = (
        confirmation_macro[CANDIDATE]["pixel_average_precision"]
        > max(
            confirmation_macro[name]["pixel_average_precision"]
            for name in REFERENCES
        )
        and confirmation_macro[CANDIDATE]["instance_auc_all"]
        > max(confirmation_macro[name]["instance_auc_all"] for name in REFERENCES)
        and parent_identity
    )
    return {
        "schema_version": 1,
        "development_fold": DEVELOPMENT_FOLD,
        "confirmation_folds": CONFIRMATION_FOLDS,
        "all_folds": ALL_FOLDS,
        "candidate": CANDIDATE,
        "references": REFERENCES,
        "bootstrap": {
            "seed": seed,
            "samples": samples,
            "resampling_unit": "roll",
        },
        "per_fold": per_fold,
        "confirmation_macro": confirmation_macro,
        "all_roll_macro_descriptive": all_macro,
        "confirmation_candidate_minus_reference": confirmation_comparisons,
        "all_roll_candidate_minus_reference_descriptive": all_comparisons,
        "confirmation_gate": {
            "strongest_pixel_ap_reference": strongest_pixel,
            "strongest_instance_auc_reference": strongest_auc,
            "fold_wins_against_each_fold_best_reference": fold_wins,
            "checks": checks,
            "passed": all(checks.values()),
            "strong_pareto_condition_passed": strong_pareto,
            "support_condition": specification["support_condition"],
            "strong_pareto_condition": specification["strong_pareto_condition"],
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    labels = {
        "gaussian": "Gaussian",
        "hann": "Hann",
        PCAF: "PCAF",
        CANDIDATE: "Dual readout",
    }
    lines = [
        "# RAW-FABRID K2 dual-readout confirmation",
        "",
        "> Rollo4A is disclosed development data. Rollo1A/2A/3A/5A/6A/7A are the frozen confirmation set.",
        "",
        "## Pixel AP by roll",
        "",
        "| Roll | Role | Gaussian | Hann | PCAF | Dual readout |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for fold in ALL_FOLDS:
        metrics = summary["per_fold"][fold]["metrics"]
        role = "development" if fold == DEVELOPMENT_FOLD else "confirmation"
        lines.append(
            f"| {fold} | {role} | "
            + " | ".join(
                f"{metrics[name]['pixel_average_precision']:.4f}"
                for name in VARIANTS
            )
            + " |"
        )
    lines.extend([
        "",
        "## Six-roll confirmation macro",
        "",
        "| Readout | Pixel AP | All I-AUC | Parent AP | Recall | N-FP |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in VARIANTS:
        metric = summary["confirmation_macro"][name]
        lines.append(
            f"| {labels[name]} | {metric['pixel_average_precision']:.4f} | "
            f"{metric['instance_auc_all']:.4f} | "
            f"{metric['parent_average_precision']:.4f} | "
            f"{metric['source_parent_recall']:.4f} | "
            f"{metric['target_normal_false_positives']:.2f} |"
        )
    gate = summary["confirmation_gate"]
    lines.extend([
        "",
        "## Frozen decision",
        "",
        f"- Confirmation gate passed: `{gate['passed']}`.",
        f"- Strong Pareto condition passed: `{gate['strong_pareto_condition_passed']}`.",
        f"- Wins against each roll's best reference: `{gate['fold_wins_against_each_fold_best_reference']}/6`.",
        f"- Strongest Pixel-AP reference: `{gate['strongest_pixel_ap_reference']}`.",
        f"- Strongest all-instance-AUC reference: `{gate['strongest_instance_auc_reference']}`.",
        "",
        "Parent AP, source-threshold recall, and target-normal false positives of the dual readout must equal PCAF exactly by construction and are independently checked in every fold.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("--seed", type=int, default=20260813)
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
