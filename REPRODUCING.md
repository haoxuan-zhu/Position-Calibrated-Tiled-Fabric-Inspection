# Reproducing the reported results

This repository supports two different levels of reproduction. They should not
be confused:

1. **Record-level reproduction is self-contained.** A fresh clone can verify
   every released file, recompute the reported aggregate statistics from the
   committed per-fold JSON records, and run the score-field unit test.
2. **End-to-end detector reproduction is conditional.** It additionally needs
   third-party images and fold-specific PatchCore checkpoints, or enough GPU
   time to retrain those checkpoints. Neither asset is redistributed here.

The experiment JSON files are immutable evidence. They contain the hashes of
the scripts, configurations, and checkpoints used by the reported runs. For
that reason, the executed experiment scripts retain their original bytes;
reader guidance is added here instead of silently changing a hash-locked
implementation after the results were obtained.

## 1. Environment

The reported GPU runs used Python 3.10, PyTorch 2.5.1+cu124,
torchvision 0.20.1, anomalib 2.5.1, and an NVIDIA RTX 4090. A matching setup is:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda124.txt
python -m pip install -r requirements.txt
```

The checksum and table-reproduction steps below need no GPU. The checksum step
uses only the Python standard library.

## 2. Self-contained checks from a fresh clone

Run these commands from the repository root:

```bash
python tools/verify_release.py
python tools/reproduce_reported_tables.py
python tools/test_patchcore_field_variants.py
```

`verify_release.py` checks the SHA-256 inventory. The table script reads only
the frozen summaries under `runs/` and reconstructs the headline RAW-FABRID,
OLP, and native-score-field comparisons. In the OLP table, Gaussian--PCAF is a
deterministic dual-output composition: Gaussian supplies the localization
metrics and the frozen PCAF branch supplies the parent-alarm metrics. It does
not require an additional detector run.

The more detailed summary commands in the root `README.md` rebuild the complete
aggregate JSON files from the released per-fold or per-textile records. Their
scientific values match the committed summaries; on Windows, regenerated path
strings may use backslashes rather than forward slashes.

## 3. Code map

- `probe_raw_fabrid_anomalib_patchcore.py`: trains/evaluates the grouped base
  PatchCore fold and writes its checkpoint.
- `probe_raw_fabrid_dual_readout_fusion_k2.py`: final RAW-FABRID PCDR readout;
  `compose_dual_readout_metrics` makes the localization/alarm contract explicit.
- `probe_olp_pcdr_external_k2.py`: complete external PCDR replay for one OLP
  textile with a frozen checkpoint and scene split.
- `probe_raw_fabrid_patch_score_field_control_k3.py` and
  `patchcore_field_variants.py`: derive raw, resize-only, and standard Anomalib
  fields from one PatchCore forward pass.
- `summarize_*`: aggregate only machine-readable fold/textile records; they do
  not run a detector or choose a model.
- `runs/*/README.md`: experiment-specific claim boundaries and stopping rules.

All runner programs expose their arguments with `--help`. The frozen base
runner's legacy help text gives `R1` as an example; the dataset metadata and all
formal runs use the full identifiers `Rollo1A` through `Rollo7A`.

## 4. Third-party data layout

Obtain the datasets from their cited source publications and comply with their
licenses:

- [RAW-FABRID source publication](https://doi.org/10.3390/data11050116)
- [ISP-AD source publication](https://doi.org/10.1007/s10845-025-02778-z)
- [OLP source publication](https://doi.org/10.3390/s22134750)

The RAW-FABRID root expected by `--data-root` is:

```text
RAW_FABRID/
├── RAW_FABRID_HighRes_Metadata.csv
├── RAW_FABRID_HighRes_COCO.json
└── images/
    └── <filenames referenced by the metadata CSV>
```

For OLP, keep the official RGB images/COCO metadata locally. The released
`data/OLP/scene_grouped_audit.json`, `subset_members.txt`, and
`subset_manifest.json` define the eligible subset and frozen scene assignment.
Copy `configs/olp_patchcore_pcaf_external.json` before replacing its three
machine-specific data paths; do not edit the frozen configuration in place.

## 5. RAW-FABRID end-to-end example

First train and score one grouped base fold. `Rollo1A` is the held-out target:

```bash
python tools/probe_raw_fabrid_anomalib_patchcore.py \
  configs/raw_fabrid_anomalib_patchcore_k0.json \
  outputs/base/P2_Rollo1A.json \
  --fold Rollo1A --protocol P2 --data-root /path/to/RAW_FABRID
```

The runner writes the corresponding Lightning checkpoint below the output
directory. Pass that checkpoint into the frozen PCDR readout:

```bash
python tools/probe_raw_fabrid_dual_readout_fusion_k2.py \
  configs/raw_fabrid_dual_readout_fusion_k2.json \
  outputs/pcdr/Rollo1A.json \
  --fold Rollo1A \
  --checkpoint outputs/base/lightning_Rollo1A/Patchcore/v0/weights/lightning/model.ckpt \
  --reference runs/raw_fabrid_context_bias_field_k3/final_reproduction/Rollo1A.json \
  --data-root /path/to/RAW_FABRID
```

Repeat for all seven full roll identifiers and aggregate the resulting files
with `summarize_raw_fabrid_dual_readout_fusion_k2.py`. Exact equality with the
released floating-point detector records additionally requires the archived
checkpoint hashes recorded in those JSON files; a deterministic retraining run
is a scientific replication, not a byte-for-byte checkpoint recovery.

## 6. Reproduction boundary

The public package is sufficient to audit every headline number and the exact
code/configuration identities. It is not a turnkey image-level replay without
the third-party datasets and checkpoints. Historical absolute paths in frozen
JSON/config files are provenance, not secrets; use command-line overrides or a
copied configuration for a new machine.
