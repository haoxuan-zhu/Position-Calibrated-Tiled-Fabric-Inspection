#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/Fabric_Defect_Detection}"
OUTPUT="${2:-$ROOT/runs/raw_fabrid_patch_score_field_control_k3/all_folds}"
PYTHON="$ROOT/.venv-anomalib/bin/python"
CONFIG="$ROOT/configs/raw_fabrid_patch_score_field_control_k3.json"
FOLDS=(Rollo1A Rollo2A Rollo3A Rollo4A Rollo5A Rollo6A Rollo7A)

cd "$ROOT"
export PYTHONPATH="$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/logs"

status="$OUTPUT/runner_status.json"
current_fold=none
write_status() {
  local state="$1"
  local message="$2"
  printf '{\n  "state": "%s",\n  "fold": "%s",\n  "message": "%s",\n  "pid": %d,\n  "updated_at": "%s"\n}\n' \
    "$state" "$current_fold" "$message" "$$" "$(date --iso-8601=seconds)" > "$status"
}
trap 'write_status failed "runner exited before completion"' ERR

for current_fold in "${FOLDS[@]}"; do
  checkpoint="$ROOT/runs/raw_fabrid_patchcore_k0/instance_audit/lightning_${current_fold}/Patchcore/v0/weights/lightning/model.ckpt"
  reference="$ROOT/runs/raw_fabrid_dual_readout_fusion_k2/all_folds/${current_fold}.json"
  result="$OUTPUT/${current_fold}.json"
  log="$OUTPUT/logs/${current_fold}.log"
  if [[ ! -f "$checkpoint" || ! -f "$reference" ]]; then
    printf 'Missing fold asset: %s or %s\n' "$checkpoint" "$reference" >&2
    exit 1
  fi
  if [[ -e "$result" ]]; then
    printf 'Refusing to overwrite existing result: %s\n' "$result" >&2
    exit 2
  fi
  write_status running "single-forward patch-score field control"
  printf 'START %s %s\n' "$current_fold" "$(date --iso-8601=seconds)"
  "$PYTHON" tools/probe_raw_fabrid_patch_score_field_control_k3.py \
    "$CONFIG" "$result" \
    --fold "$current_fold" \
    --checkpoint "$checkpoint" \
    --reference "$reference" >"$log" 2>&1
  printf 'DONE %s %s\n' "$current_fold" "$(date --iso-8601=seconds)"
done

"$PYTHON" tools/summarize_raw_fabrid_patch_score_field_control_k3.py \
  "$OUTPUT" \
  "$ROOT/runs/raw_fabrid_patch_score_field_control_k3/seven_fold_summary.json"

current_fold=all
write_status complete "seven-fold patch-score field control complete"

