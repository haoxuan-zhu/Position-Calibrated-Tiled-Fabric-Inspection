"""Aggregate complete PCDR results across the 15 eligible OLP textiles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from probe_olp_pcdr_external_k2 import HANN_PCAF, PCDR


VARIANTS = ("gaussian", "hann", HANN_PCAF, PCDR)
METRICS = (
    "pixel_average_precision",
    "pixel_roc_auc",
    "parent_average_precision",
    "parent_roc_auc",
    "evaluation_normal_false_positives",
    "evaluation_normal_fpr",
    "defect_parent_recall",
)


def bootstrap_interval(values: np.ndarray, seed: int, samples: int) -> list[float]:
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(values), size=(samples, len(values)))
    means = values[choices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def exact_sign_test(values: np.ndarray) -> float | None:
    positive = int(np.sum(values > 0))
    negative = int(np.sum(values < 0))
    non_ties = positive + negative
    if non_ties == 0:
        return None
    tail = sum(
        math.comb(non_ties, index)
        for index in range(min(positive, negative) + 1)
    ) / (2**non_ties)
    return float(min(1.0, 2.0 * tail))


def summarize(results_dir: Path, seed: int, samples: int) -> dict[str, Any]:
    paths = sorted(results_dir.glob("textile_*.json"))
    if len(paths) != 15:
        raise ValueError(f"Expected 15 textile results in {results_dir}, found {len(paths)}")
    per_textile: dict[str, Any] = {}
    script_hashes: set[str] = set()
    config_hashes: set[str] = set()
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        textile = str(result["run"]["textile_id"])
        if textile in per_textile:
            raise ValueError(f"Duplicate textile {textile}")
        if not result["frozen_source_identity"]["passed"]:
            raise ValueError(f"Frozen source identity failed for textile {textile}")
        if not result["frozen_replay"]["passed"]:
            raise ValueError(f"Frozen discrete replay failed for textile {textile}")
        for name in (HANN_PCAF, PCDR):
            if not result["dual_readout_identity"][name]["passed"]:
                raise ValueError(f"Dual identity failed for textile {textile}/{name}")
        script_hashes.add(str(result["run"]["script_sha256"]))
        config_hashes.add(str(result["run"]["config_sha256"]))
        per_textile[textile] = {
            "counts": result["counts"],
            "metrics": {name: result["metrics"][name] for name in VARIANTS},
            "checkpoint_sha256": result["run"]["checkpoint_sha256"],
            "legacy_maximum_absolute_metric_error": result["frozen_replay"][
                "maximum_absolute_metric_error"
            ],
        }
    if len(script_hashes) != 1 or len(config_hashes) != 1:
        raise ValueError("All textiles must share one script and configuration")

    textile_ids = sorted(per_textile, key=int)
    macro = {
        variant: {
            metric: float(
                np.mean(
                    [per_textile[textile]["metrics"][variant][metric] for textile in textile_ids]
                )
            )
            for metric in METRICS
        }
        for variant in VARIANTS
    }
    comparisons: dict[str, Any] = {}
    for reference_offset, reference in enumerate(("gaussian", "hann", HANN_PCAF)):
        comparisons[reference] = {}
        for metric_offset, metric in enumerate(METRICS):
            deltas = np.asarray(
                [
                    per_textile[textile]["metrics"][PCDR][metric]
                    - per_textile[textile]["metrics"][reference][metric]
                    for textile in textile_ids
                ],
                dtype=np.float64,
            )
            comparisons[reference][metric] = {
                "textile_macro_mean_delta": float(deltas.mean()),
                "bootstrap_95pct_ci_of_macro_mean": bootstrap_interval(
                    deltas,
                    seed + 100 * reference_offset + metric_offset,
                    samples,
                ),
                "wins": int(np.sum(deltas > 0)),
                "ties": int(np.sum(deltas == 0)),
                "losses": int(np.sum(deltas < 0)),
                "exact_two_sided_sign_test_p": exact_sign_test(deltas),
                "worst_delta": float(deltas.min()),
                "best_delta": float(deltas.max()),
                "per_textile_delta": {
                    textile: float(value)
                    for textile, value in zip(textile_ids, deltas, strict=True)
                },
            }
    return {
        "schema_version": 1,
        "textiles": [int(value) for value in textile_ids],
        "per_textile": per_textile,
        "textile_macro": macro,
        "pcdr_minus_reference": comparisons,
        "script_sha256": next(iter(script_hashes)),
        "config_sha256": next(iter(config_hashes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    summary = summarize(args.results_dir, args.seed, args.bootstrap_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "textile_macro": summary["textile_macro"],
        "pcdr_minus_hann_pcaf": summary["pcdr_minus_reference"][HANN_PCAF],
    }, indent=2))


if __name__ == "__main__":
    main()
