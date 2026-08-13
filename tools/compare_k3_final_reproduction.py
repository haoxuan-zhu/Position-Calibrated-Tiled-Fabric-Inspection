"""Compare current-script K3 reproductions against the audited seven-fold results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FOLDS = [f"Rollo{index}A" for index in range(1, 8)]
VARIANTS = ["single_base", "mean", "context_bias_weighted_mean"]
METRICS = [
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
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-dir", type=Path,
        default=ROOT / "runs/raw_fabrid_context_bias_field_k3/all_folds",
    )
    parser.add_argument(
        "--reference-roll4", type=Path,
        default=ROOT / "runs/raw_fabrid_context_bias_field_k3/sensitivity/shift64_Rollo4A.json",
    )
    parser.add_argument(
        "--candidate-dir", type=Path,
        default=ROOT / "runs/raw_fabrid_context_bias_field_k3/final_reproduction",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/raw_fabrid_context_bias_field_k3/final_reproduction_comparison.json",
    )
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    script_hash = sha256(ROOT / "tools/probe_raw_fabrid_physical_field_k0.py")
    config_hash = sha256(ROOT / "configs/raw_fabrid_context_bias_field_k3.json")
    report: dict[str, Any] = {
        "schema_version": 1,
        "absolute_tolerance": args.absolute_tolerance,
        "expected_current_script_sha256": script_hash,
        "expected_config_sha256": config_hash,
        "folds": {},
    }
    all_pass = True
    for fold in FOLDS:
        reference_path = args.reference_roll4 if fold == "Rollo4A" else args.reference_dir / f"{fold}.json"
        candidate_path = args.candidate_dir / f"{fold}.json"
        reference = load(reference_path)
        candidate = load(candidate_path)
        metric_differences: dict[str, float] = {}
        metric_matches = True
        for variant in VARIANTS:
            for metric in METRICS:
                old = float(reference["compact_primary_metrics"][variant][metric])
                new = float(candidate["compact_primary_metrics"][variant][metric])
                difference = new - old
                metric_differences[f"{variant}.{metric}"] = difference
                metric_matches &= math.isclose(
                    old, new, rel_tol=0.0, abs_tol=args.absolute_tolerance
                )
        checks = {
            "identity": candidate["run"]["fold"] == fold
            and candidate["run"]["model"] == "patchcore",
            "split_indices_exact": candidate["split_indices"] == reference["split_indices"],
            "checkpoint_hash_exact": candidate["run"]["checkpoint_sha256"]
            == reference["run"]["checkpoint_sha256"],
            "current_script_hash_exact": candidate["run"]["script_sha256"] == script_hash,
            "config_hash_exact": candidate["run"]["config_sha256"] == config_hash,
            "metrics_within_tolerance": metric_matches,
        }
        passed = all(checks.values())
        all_pass &= passed
        report["folds"][fold] = {
            "reference_path": str(reference_path.resolve()),
            "reference_sha256": sha256(reference_path),
            "candidate_path": str(candidate_path.resolve()),
            "candidate_sha256": sha256(candidate_path),
            "checks": checks,
            "maximum_absolute_metric_difference": max(
                abs(value) for value in metric_differences.values()
            ),
            "metric_differences": metric_differences,
            "passed": passed,
        }
    report["passed"] = all_pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"K3_FINAL_REPRODUCTION_{'OK' if all_pass else 'FAILED'} {args.output}")
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
