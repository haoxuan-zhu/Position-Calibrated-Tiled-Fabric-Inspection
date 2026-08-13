# Position-Calibrated Tiled Fabric Inspection

Official reproducibility repository for **Position-Calibrated Dual Readout for PatchCore-Based Tiled Fabric Anomaly Inspection**.

High-resolution fabric images are commonly inspected as overlapping tiles. A physical point can then receive different anomaly responses depending on where it falls inside a tile. Position-Calibrated Dual Readout (PCDR) estimates this repeatable tile-coordinate response pattern from source-normal images and uses two task-specific readouts:

- a bias-corrected Gaussian reconstruction for dense defect localization;
- Position-Calibrated Aligned Fusion (PCAF) for parent-image alarming and source-normal threshold calibration.

Both readouts reuse the same frozen PatchCore crop maps. PCDR requires no defect labels, detector retraining, second network, or additional crop acquisition beyond the chosen tiled scan.

![Crop-coordinate calibration audit](assets/coordinate_calibration_audit.png)

## Main result

Rollo4A was used for method development. On the other six RAW-FABRID rolls, the frozen PCDR localization readout reaches a macro Pixel AP of **0.3017**, compared with **0.2963** for the strongest fixed reference, a paired change of **+0.0053**. It exceeds each roll's strongest reference on **5/6** confirmation rolls. All parent scores and source-calibrated operating-point metrics are exactly inherited from the PCAF alarm readout.

The repository also retains the complete seven-roll diagnostic records, fixed-window controls, a same-location fixed-effect audit, the grouped OLP external evaluation, and a computation-aware sequential-acquisition extension. These supporting studies are kept separate from the six-roll confirmation estimate.

## Repository layout

- `configs/`: frozen experiment specifications;
- `tools/`: PCDR, PCAF, baseline, audit, and summarization programs;
- `runs/`: immutable per-fold JSON outputs and aggregate summaries;
- `data/OLP/`: metadata-only scene-grouping and subset manifests;
- `assets/`: the coordinate-calibration audit figure;
- `release_manifest.json`: SHA-256 inventory of the public package.

The repository intentionally excludes the paper source, internal work logs, model checkpoints, and third-party image data.

## Verify the released records

Python 3.10 is recommended. The checksum audit uses only the standard library:

```bash
python tools/verify_release.py
```

The principal aggregate tables can be regenerated directly from the committed fold records:

```bash
python tools/summarize_raw_fabrid_dual_readout_fusion_k2.py \
  runs/raw_fabrid_dual_readout_fusion_k2/all_folds \
  outputs/dual_readout_summary.json outputs/dual_readout_summary.md

python tools/summarize_raw_fabrid_coordinate_fixed_effect_audit.py \
  runs/raw_fabrid_coordinate_fixed_effect_audit/full \
  outputs/fixed_effect_summary.json outputs/fixed_effect_summary.md
```

Run each program with `--help` before use; the frozen runner scripts remain available for the original experiment layout.

## End-to-end reproduction

The experiments were executed with Python 3.10, PyTorch 2.1, CUDA 12.1, anomalib 2.5.1, and an RTX 4090. Install a CUDA-compatible PyTorch build first, then install `requirements.txt`.

End-to-end reruns additionally require:

1. the official RAW-FABRID, ISP-AD, or OLP data obtained from their respective providers;
2. the dataset paths in the selected JSON configuration to be adapted to the local machine;
3. fold-specific PatchCore checkpoints, either retrained with the deterministic grouped protocol or supplied separately.

The committed JSON records include configuration, implementation, and checkpoint hashes used by the reported runs. Historical absolute paths in frozen configurations record the original machine layout; they are not credentials and should be changed only in a copied configuration.

## Evaluation boundaries

- Parents and rolls, not crops, are the experimental units.
- The six-roll confirmation result is candidate-unseen but not a fully blind dataset evaluation: reference outputs existed before the candidate readout was frozen.
- ISP-AD is used only as label-free evidence that identical registered content can receive crop-dependent detector responses.
- OLP is grouped by acquisition scene and summarized by textile; it is supporting external evidence, not a cross-roll test.
- The sequential extension reduces average crop count but is not lossless and is not part of the primary PCDR claim.

## Citation

The manuscript is being prepared for journal submission. Citation metadata will be updated with the journal record or preprint identifier when available.

## License

No open-source license is granted in this submission-stage snapshot. A license will be selected before or when the repository is made public.
