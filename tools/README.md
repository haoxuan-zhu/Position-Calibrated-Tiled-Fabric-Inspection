# Code map

The programs in this directory are retained with the bytes recorded by the
released experiment JSON files. This guide supplies a reading order without
changing those executed sources.

## Start with the reported method

1. `probe_raw_fabrid_physical_field_k0.py` defines the crop references,
   parent-coordinate registration, fixed windows, and aligned fusion shared by
   the reconstruction experiments.
2. `probe_raw_fabrid_dual_readout_fusion_k2.py` applies the final RAW-FABRID
   PCDR localization/alarm contract to one held-out roll.
3. `summarize_raw_fabrid_dual_readout_fusion_k2.py` aggregates the seven stored
   roll records and reports the six-roll candidate evaluation.
4. `probe_olp_pcdr_external_k2.py` and
   `summarize_olp_pcdr_external_k2.py` implement the scene-grouped external
   evaluation.
5. `patchcore_field_variants.py` and
   `probe_raw_fabrid_patch_score_field_control_k3.py` implement the native
   patch-score control.

For a quick check that does not score images, use
`reproduce_reported_tables.py`. For the end-to-end command sequence and data
layout, see the root [`REPRODUCING.md`](../REPRODUCING.md).

## File-name roles

- `probe_*`: train, score, or replay one experiment unit and write JSON.
- `audit_*`: test a mechanism or data relationship without defining the main
  accuracy estimate.
- `summarize_*`: aggregate existing JSON records; these programs do not train
  a detector.
- `verify_*` and `compare_*`: check release, subset, or replay identities.
- `run_*_remote.sh`: preserve the original multi-fold server command order.

The `k0`--`k3` suffixes are historical experiment-stage identifiers, not
method rankings. Formal settings are in `../configs/`, and immutable reported
outputs are in `../runs/`. All Python runners expose their command-line
arguments through `--help`.
