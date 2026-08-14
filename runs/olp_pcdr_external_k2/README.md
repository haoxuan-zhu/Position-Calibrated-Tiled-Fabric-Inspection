# OLP complete PCDR external evaluation

Status: `complete_formal_checkpoint_replay`.

The experiment reuses the 15 original scene-grouped OLP PatchCore checkpoints,
metadata-frozen textile splits, and crop observations. It adds Gaussian, Hann,
Hann–PCAF, and complete PCDR readouts without retraining. Each accepted result
matches the original checkpoint hash, split paths, split counts, and discrete
false-positive/recall outcomes exactly. Continuous differences from the legacy
in-memory scoring run are retained as diagnostic drift because the formal run
reloads the saved checkpoint; all compared methods are recomputed from the same
replayed crop cache.

Formal results are in `final_checkpoint_replay/`; the textile-macro aggregate is
`scene_grouped_summary.json`. Across 15 textiles, PCDR reaches `0.14841` macro
Pixel AP versus `0.14460` for Hann: `12/15` textiles improve, exact sign
`p=0.03516`, and the textile-bootstrap interval `[-0.00134,+0.00910]` crosses
zero. This is external acquisition-and-texture evidence, not roll or temporal
validation.
