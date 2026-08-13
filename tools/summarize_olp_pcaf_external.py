"""Aggregate the frozen eligible-textile OLP PCAF external results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = [
    "single_base",
    "mean",
    "context_bias_equal_mean",
    "context_bias_weighted_mean",
]
METRICS = [
    "pixel_average_precision",
    "pixel_roc_auc",
    "parent_average_precision",
    "parent_roc_auc",
    "evaluation_normal_false_positives",
    "evaluation_normal_fpr",
    "defect_parent_recall",
]


def exact_two_sided_sign_test(positive: int, negative: int) -> float | None:
    non_ties = positive + negative
    if non_ties == 0:
        return None
    tail = sum(
        math.comb(non_ties, successes)
        for successes in range(min(positive, negative) + 1)
    ) / (2**non_ties)
    return float(min(1.0, 2.0 * tail))


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


def load_results(
    results_dir: Path, audit: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    identity: tuple[str, str, str] | None = None
    for textile_id in audit["eligible_textiles"]:
        path = results_dir / f"textile_{int(textile_id):02d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing OLP textile result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        run = result["run"]
        if int(run["textile_id"]) != int(textile_id) or run["smoke"]:
            raise ValueError(f"Result identity mismatch: {path}")
        current_identity = (
            run["script_sha256"],
            run["config_sha256"],
            run["audit_sha256"],
        )
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError(f"Script/config/audit hash drift at {path}")
        if set(VARIANTS) - set(result["metrics"]):
            raise ValueError(f"Missing variants in {path}")
        results[int(textile_id)] = result
    return results


def aggregate(
    results: dict[int, dict[str, Any]],
    audit: dict[str, Any],
    seed: int,
    samples: int,
) -> dict[str, Any]:
    textile_ids = [int(value) for value in audit["eligible_textiles"]]
    per_textile: dict[str, Any] = {}
    for textile_id in textile_ids:
        result = results[textile_id]
        per_textile[str(textile_id)] = {
            "counts": result["counts"],
            "split_scenes": result["split_scenes"],
            "crop_budget": result["crop_budget"],
            "run": result["run"],
            "metrics": result["metrics"],
            "pcaf_minus_mean": {
                metric: float(
                    result["metrics"]["context_bias_weighted_mean"][metric]
                    - result["metrics"]["mean"][metric]
                )
                for metric in METRICS
            },
        }

    macro = {
        variant: {
            metric: describe(
                [
                    float(per_textile[str(textile_id)]["metrics"][variant][metric])
                    for textile_id in textile_ids
                ]
            )
            for metric in METRICS
        }
        for variant in VARIANTS
    }
    paired: dict[str, Any] = {}
    for offset, metric in enumerate(METRICS):
        deltas = np.asarray(
            [
                per_textile[str(textile_id)]["pcaf_minus_mean"][metric]
                for textile_id in textile_ids
            ],
            dtype=np.float64,
        )
        wins = int(np.sum(deltas > 0))
        ties = int(np.sum(deltas == 0))
        losses = int(np.sum(deltas < 0))
        paired[metric] = {
            "textile_deltas": {
                str(textile_id): float(delta)
                for textile_id, delta in zip(textile_ids, deltas, strict=True)
            },
            "textile_macro_mean_delta": float(deltas.mean()),
            "textile_macro_median_delta": float(np.median(deltas)),
            "textile_bootstrap_95pct_ci": bootstrap_interval(
                deltas, seed + offset, samples
            ),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "exact_two_sided_sign_test_p": exact_two_sided_sign_test(wins, losses),
        }

    pixel = paired["pixel_average_precision"]
    rule = next(iter(results.values()))["config"][
        "decision_rule_frozen_before_outputs"
    ]
    external_success = (
        pixel["textile_macro_mean_delta"] > 0
        and pixel["textile_bootstrap_95pct_ci"][0] > 0
        and pixel["wins"] >= 10
        and min(pixel["textile_deltas"].values()) >= -0.05
    )
    supporting_only = (
        not external_success
        and pixel["textile_macro_mean_delta"] > 0
        and pixel["wins"] >= 8
    )
    decision = (
        "external_success"
        if external_success
        else "supporting_only"
        if supporting_only
        else "failure"
    )
    return {
        "schema_version": 1,
        "purpose": "OLP scene-grouped patterned-fabric external validation",
        "claim_boundary": next(iter(results.values()))["claim_boundary"],
        "textiles": textile_ids,
        "independent_summary_unit": "textile",
        "protocol_totals": audit["totals"],
        "bootstrap": {"seed": seed, "samples": samples, "unit": "textile"},
        "frozen_decision_rule": rule,
        "decision": decision,
        "per_textile": per_textile,
        "macro": macro,
        "paired_pcaf_minus_mean": paired,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# OLP 场景分组外部验证",
        "",
        "> 自动生成。每种织物按采集场景隔离训练、校准与测试；只使用前光 RGB。",
        "",
        "| Textile | eval scenes | defects | Mean Pixel AP | PCAF Pixel AP | delta | Mean Parent AUC | PCAF Parent AUC | normal FP delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for textile_id in summary["textiles"]:
        item = summary["per_textile"][str(textile_id)]
        mean = item["metrics"]["mean"]
        pcaf = item["metrics"]["context_bias_weighted_mean"]
        delta = item["pcaf_minus_mean"]
        lines.append(
            f"| {textile_id} | {len(item['split_scenes']['anomalies'])} | "
            f"{item['counts']['anomalies']} | "
            f"{mean['pixel_average_precision']:.4f} | "
            f"{pcaf['pixel_average_precision']:.4f} | "
            f"{delta['pixel_average_precision']:+.4f} | "
            f"{mean['parent_roc_auc']:.4f} | {pcaf['parent_roc_auc']:.4f} | "
            f"{delta['evaluation_normal_false_positives']:+.0f} |"
        )
    pixel = summary["paired_pcaf_minus_mean"]["pixel_average_precision"]
    lines.extend(
        [
            "",
            f"Decision: `{summary['decision']}`.",
            "",
            f"Pixel AP textile-macro delta {pixel['textile_macro_mean_delta']:+.6f}; "
            f"95% textile-bootstrap CI "
            f"[{pixel['textile_bootstrap_95pct_ci'][0]:+.6f}, "
            f"{pixel['textile_bootstrap_95pct_ci'][1]:+.6f}]; "
            f"wins/ties/losses {pixel['wins']}/{pixel['ties']}/{pixel['losses']}; "
            f"exact sign p={pixel['exact_two_sided_sign_test_p']:.6f}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    summary = aggregate(
        load_results(args.results_dir, audit),
        audit,
        args.seed,
        args.bootstrap_samples,
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
