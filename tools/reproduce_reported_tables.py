#!/usr/bin/env python3
"""Render the manuscript's headline tables from frozen aggregate JSON files.

This reader-facing script does not score images or recompute fold statistics.
The experiment-specific ``summarize_*`` programs perform those operations.
Here we expose the final mapping from audited summaries to the compact values
reported in the manuscript, including the constructed Gaussian--PCAF OLP row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict[str, Any]:
    """Load one released JSON summary relative to the repository root."""
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def same_alarm(left: dict[str, float], right: dict[str, float], keys: tuple[str, ...]) -> None:
    """Fail if two dual-output rows do not share the promised alarm branch."""
    for key in keys:
        if left[key] != right[key]:
            raise RuntimeError(f"Parent-alarm identity failed for {key}: {left[key]} != {right[key]}")


def fmt(value: float) -> str:
    """Use the four-decimal precision shown in the manuscript tables."""
    return f"{value:.4f}"


def raw_table() -> None:
    """Print the six-roll RAW-FABRID dual-readout comparison."""
    summary = load("runs/raw_fabrid_dual_readout_fusion_k2/seven_fold_summary.json")
    macro = summary["confirmation_macro"]
    candidate = macro["dual_gaussian_bias_localization_pcaf_alarm"]
    pcaf = macro["context_bias_weighted_mean"]
    alarm_keys = (
        "parent_average_precision",
        "source_parent_recall",
        "target_normal_false_positives",
    )
    same_alarm(candidate, pcaf, alarm_keys)

    # The fair controls reuse the fixed localization fields while replacing
    # only their parent-level columns with the exact PCAF alarm branch.
    gaussian_pcaf = dict(macro["gaussian"])
    hann_pcaf = dict(macro["hann"])
    for row in (gaussian_pcaf, hann_pcaf):
        for key in alarm_keys:
            row[key] = pcaf[key]

    print("## RAW-FABRID: six-roll candidate evaluation")
    print("| Readout | Pixel AP | All I-AUC | Parent AP | Recall | N-FP |")
    print("|---|---:|---:|---:|---:|---:|")
    rows = (
        ("Gaussian", macro["gaussian"]),
        ("Hann", macro["hann"]),
        ("PCAF", pcaf),
        ("Gaussian--PCAF", gaussian_pcaf),
        ("Hann--PCAF", hann_pcaf),
        ("PCDR", candidate),
    )
    for label, row in rows:
        print(
            f"| {label} | {fmt(row['pixel_average_precision'])} | "
            f"{fmt(row['instance_auc_all'])} | {fmt(row['parent_average_precision'])} | "
            f"{fmt(row['source_parent_recall'])} | "
            f"{int(row['target_normal_false_positives'])} |"
        )
    comparison = summary["confirmation_candidate_minus_reference"]["hann"][
        "pixel_average_precision"
    ]
    low, high = comparison["bootstrap_95pct_ci_of_macro_mean"]
    print(
        f"\nPCDR - Hann--PCAF Pixel AP: {comparison['macro_mean_delta']:+.4f} "
        f"(95% roll-bootstrap CI [{low:+.4f}, {high:+.4f}]); "
        f"W/T/L={comparison['wins']}/{comparison['ties']}/{comparison['losses']}.\n"
    )


def olp_table() -> None:
    """Print the 15-textile OLP table, including Gaussian--PCAF by construction."""
    summary = load("runs/olp_pcdr_external_k2/scene_grouped_summary.json")
    macro = summary["textile_macro"]
    alarm = macro["hann_pcaf_dual"]
    same_alarm(
        macro["pcdr"],
        alarm,
        (
            "parent_average_precision",
            "parent_roc_auc",
            "evaluation_normal_false_positives",
            "defect_parent_recall",
        ),
    )

    # Gaussian--PCAF needs no new detector output: its localization columns are
    # Gaussian's, while every parent-level column is copied from frozen PCAF.
    gaussian_pcaf = dict(alarm)
    gaussian_pcaf["pixel_average_precision"] = macro["gaussian"][
        "pixel_average_precision"
    ]

    print("## OLP: 15-textile external evaluation")
    print("| Readout | Pixel AP | Parent AP | Parent AUC | N-FP / 280 | Recall |")
    print("|---|---:|---:|---:|---:|---:|")
    rows = (
        ("Gaussian", macro["gaussian"]),
        ("Hann", macro["hann"]),
        ("Gaussian--PCAF", gaussian_pcaf),
        ("Hann--PCAF", alarm),
        ("PCDR", macro["pcdr"]),
    )
    for label, row in rows:
        # N-FP is a total in the paper; the released macro stores its per-textile mean.
        nfp_total = round(row["evaluation_normal_false_positives"] * len(summary["textiles"]))
        print(
            f"| {label} | {fmt(row['pixel_average_precision'])} | "
            f"{fmt(row['parent_average_precision'])} | {fmt(row['parent_roc_auc'])} | "
            f"{nfp_total} | {fmt(row['defect_parent_recall'])} |"
        )
    # Hann and Hann--PCAF have identical localization values.  The manuscript
    # retains the prereported Hann-comparator bootstrap stream (seed 20260914)
    # while naming the fair dual-output comparator.
    comparison = summary["pcdr_minus_reference"]["hann"][
        "pixel_average_precision"
    ]
    low, high = comparison["bootstrap_95pct_ci_of_macro_mean"]
    print(
        f"\nPCDR - Hann--PCAF Pixel AP: {comparison['textile_macro_mean_delta']:+.4f} "
        f"(95% textile-bootstrap CI [{low:+.4f}, {high:+.4f}]); "
        f"W/T/L={comparison['wins']}/{comparison['ties']}/{comparison['losses']}.\n"
    )


def field_control() -> None:
    """Print the native-score-field control used to rule out rendering confounding."""
    summary = load(
        "runs/raw_fabrid_patch_score_field_control_k3/seven_fold_summary.json"
    )
    print("## RAW-FABRID score-field control")
    print("| Field | Hann Pixel AP | PCDR Pixel AP | Delta | W/L |")
    print("|---|---:|---:|---:|---:|")
    labels = {
        "raw_patch": "Raw patch",
        "resize_roundtrip": "Resize round trip",
        "anomalib_default": "Anomalib default",
    }
    for key, label in labels.items():
        row = summary["confirmation_macro"][key]["pixel_average_precision"]
        print(
            f"| {label} | {fmt(row['hann'])} | {fmt(row['pcdr'])} | "
            f"{row['macro_mean_delta']:+.4f} | {row['wins']}/{row['losses']} |"
        )


def main() -> None:
    raw_table()
    olp_table()
    field_control()
    print("\nREPORTED_TABLES_OK")


if __name__ == "__main__":
    main()
