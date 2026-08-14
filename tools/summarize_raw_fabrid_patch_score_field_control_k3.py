"""Aggregate the seven-fold PatchCore field-rendering control."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from patchcore_field_variants import ANOMALIB_DEFAULT, FIELD_VARIANTS
from probe_raw_fabrid_dual_readout_fusion_k2 import CANDIDATE


CONFIRMATION_FOLDS = [
    "Rollo1A",
    "Rollo2A",
    "Rollo3A",
    "Rollo5A",
    "Rollo6A",
    "Rollo7A",
]
METRICS = ("pixel_average_precision", "instance_auc_all")


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
    paths = sorted(results_dir.glob("Rollo*A.json"))
    if len(paths) != 7:
        raise ValueError(f"Expected seven fold files in {results_dir}, found {len(paths)}")
    per_fold: dict[str, Any] = {}
    script_hashes: set[str] = set()
    config_hashes: set[str] = set()
    helper_hashes: set[str] = set()
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        fold = str(result["run"]["fold"])
        if fold in per_fold:
            raise ValueError(f"Duplicate fold {fold}")
        if not result["standard_replay"]["passed"]:
            raise ValueError(f"Standard replay failed for {fold}")
        if result["standard_replay"]["maximum_absolute_metric_error"] > 1e-8:
            raise ValueError(f"Nonzero standard replay drift for {fold}")
        if result["counts"]["crops_per_parent"] != 73:
            raise ValueError(f"Crop-count mismatch for {fold}")
        if set(result["field_results"]) != set(FIELD_VARIANTS):
            raise ValueError(f"Field variants differ for {fold}")
        for name in FIELD_VARIANTS:
            if not result["field_results"][name]["dual_readout_identity"]["exact_pcaf_parent_readout"]:
                raise ValueError(f"PCDR identity failed for {fold}/{name}")
        script_hashes.add(str(result["run"]["script_sha256"]))
        config_hashes.add(str(result["run"]["config_sha256"]))
        helper_hashes.add(str(result["run"]["field_helper_sha256"]))
        per_fold[fold] = result
    if len(script_hashes) != 1 or len(config_hashes) != 1 or len(helper_hashes) != 1:
        raise ValueError("All folds must share one script, config, and field helper")
    if set(per_fold) != {f"Rollo{index}A" for index in range(1, 8)}:
        raise ValueError(f"Unexpected folds: {sorted(per_fold)}")

    output: dict[str, Any] = {
        "schema_version": 1,
        "field_variants": list(FIELD_VARIANTS),
        "confirmation_folds": CONFIRMATION_FOLDS,
        "per_fold": {},
        "confirmation_macro": {},
        "all_fold_descriptive_macro": {},
        "coordinate_center_correlations_to_anomalib_default": {},
        "script_sha256": next(iter(script_hashes)),
        "config_sha256": next(iter(config_hashes)),
        "field_helper_sha256": next(iter(helper_hashes)),
    }
    for fold, result in sorted(per_fold.items()):
        output["per_fold"][fold] = {
            name: {
                "hann": result["field_results"][name]["compact_primary_metrics"]["hann"],
                "pcdr": result["field_results"][name]["compact_primary_metrics"][CANDIDATE],
                "pcdr_minus_hann": result["field_results"][name]["pcdr_minus_hann"],
            }
            for name in FIELD_VARIANTS
        }
        standard_center = np.asarray(
            result["field_results"][ANOMALIB_DEFAULT]["coordinate_calibration"]["center_map"],
            dtype=np.float64,
        ).reshape(-1)
        output["coordinate_center_correlations_to_anomalib_default"][fold] = {
            name: float(
                np.corrcoef(
                    np.asarray(
                        result["field_results"][name]["coordinate_calibration"]["center_map"],
                        dtype=np.float64,
                    ).reshape(-1),
                    standard_center,
                )[0, 1]
            )
            for name in FIELD_VARIANTS
        }

    all_folds = sorted(per_fold)
    for label, folds in (
        ("confirmation_macro", CONFIRMATION_FOLDS),
        ("all_fold_descriptive_macro", all_folds),
    ):
        for name in FIELD_VARIANTS:
            current: dict[str, Any] = {}
            for metric_offset, metric in enumerate(METRICS):
                hann = np.asarray(
                    [
                        per_fold[fold]["field_results"][name]["compact_primary_metrics"]["hann"][metric]
                        for fold in folds
                    ],
                    dtype=np.float64,
                )
                pcdr = np.asarray(
                    [
                        per_fold[fold]["field_results"][name]["compact_primary_metrics"][CANDIDATE][metric]
                        for fold in folds
                    ],
                    dtype=np.float64,
                )
                delta = pcdr - hann
                current[metric] = {
                    "hann": float(hann.mean()),
                    "pcdr": float(pcdr.mean()),
                    "macro_mean_delta": float(delta.mean()),
                    "bootstrap_95pct_ci_of_macro_mean": bootstrap_interval(
                        delta,
                        seed + 100 * list(FIELD_VARIANTS).index(name) + metric_offset,
                        samples,
                    ),
                    "wins": int(np.sum(delta > 0)),
                    "ties": int(np.sum(delta == 0)),
                    "losses": int(np.sum(delta < 0)),
                    "exact_two_sided_sign_test_p": exact_sign_test(delta),
                    "fold_deltas": {
                        fold: float(value)
                        for fold, value in zip(folds, delta, strict=True)
                    },
                }
            output[label][name] = current
    return output


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
    print(json.dumps(summary["confirmation_macro"], indent=2))


if __name__ == "__main__":
    main()

