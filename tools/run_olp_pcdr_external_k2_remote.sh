#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/Fabric_Defect_Detection}"
OUTPUT="${2:-$ROOT/runs/olp_pcdr_external_k2/full}"
PYTHON="$ROOT/.venv-anomalib/bin/python"
CONFIG="$ROOT/configs/olp_pcdr_external_k2.json"
TEXTILES=(1 2 3 4 5 6 7 14 16 18 21 23 24 31 38)

cd "$ROOT"
export PYTHONPATH="$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/logs"

status="$OUTPUT/runner_status.json"
current_textile=none
write_status() {
  local state="$1"
  local message="$2"
  printf '{\n  "state": "%s",\n  "textile": "%s",\n  "message": "%s",\n  "pid": %d,\n  "updated_at": "%s"\n}\n' \
    "$state" "$current_textile" "$message" "$$" "$(date --iso-8601=seconds)" > "$status"
}
trap 'write_status failed "runner exited before completion"' ERR

for current_textile in "${TEXTILES[@]}"; do
  checkpoint=$(printf '%s/runs/olp_pcaf_external/full/checkpoints/textile_%02d.ckpt' "$ROOT" "$current_textile")
  reference=$(printf '%s/runs/olp_pcaf_external/full/textile_%02d.json' "$ROOT" "$current_textile")
  result=$(printf '%s/textile_%02d.json' "$OUTPUT" "$current_textile")
  log=$(printf '%s/logs/textile_%02d.log' "$OUTPUT" "$current_textile")
  if [[ ! -f "$checkpoint" || ! -f "$reference" ]]; then
    printf 'Missing textile asset: %s or %s\n' "$checkpoint" "$reference" >&2
    exit 1
  fi
  if [[ -e "$result" ]]; then
    printf 'Refusing to overwrite existing result: %s\n' "$result" >&2
    exit 2
  fi
  write_status running "complete PCDR external evaluation"
  printf 'START textile=%s %s\n' "$current_textile" "$(date --iso-8601=seconds)"
  "$PYTHON" tools/probe_olp_pcdr_external_k2.py \
    "$CONFIG" "$result" \
    --textile-id "$current_textile" \
    --checkpoint "$checkpoint" \
    --reference "$reference" >"$log" 2>&1
  printf 'DONE textile=%s %s\n' "$current_textile" "$(date --iso-8601=seconds)"
done

"$PYTHON" tools/summarize_olp_pcdr_external_k2.py \
  "$OUTPUT" "$ROOT/runs/olp_pcdr_external_k2/scene_grouped_summary.json"

current_textile=all
write_status complete "fifteen-textile complete PCDR evaluation complete"

