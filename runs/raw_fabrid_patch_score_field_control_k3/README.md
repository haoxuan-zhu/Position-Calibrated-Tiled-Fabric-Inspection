# RAW-FABRID PatchCore field-rendering control

Status: `complete_seven_fold_control`.

This control derives three registered crop fields from the same frozen
PatchCore patch distances: the native 32×32 field, a nearest-upsample/bilinear-
downsample round trip without smoothing, and the standard Anomalib anomaly-map
generator followed by the existing 32×32 projection.  Each field is evaluated
with the same 73 crops, source-normal split, Hann readout, coordinate-centered
PCDR localization, masks, and grouped roll protocol.

The standard branch reproduces every frozen PCDR summary metric exactly. Formal
results are in `all_folds/`; the aggregate is `seven_fold_summary.json`. On the
six-roll candidate cohort, the native unsmoothed field retains a `+0.00577`
PCDR-minus-Hann Pixel-AP gain with bootstrap interval
`[+0.00137,+0.00994]`. Raw and resize-round-trip results are identical, and
their coordinate-center maps correlate `0.983--0.986` with the standard
Anomalib maps across all seven folds. The complete rendering chain is therefore
not a necessary cause of the localization gain.
