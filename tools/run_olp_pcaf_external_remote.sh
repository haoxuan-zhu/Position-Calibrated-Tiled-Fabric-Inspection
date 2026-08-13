#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/Fabric_Defect_Detection}"
OUTPUT="${2:-$ROOT/runs/olp_pcaf_external/full}"
PYTHON="$ROOT/.venv-anomalib/bin/python"
CONFIG="$ROOT/configs/olp_patchcore_pcaf_external.json"
AUDIT="$ROOT/data/OLP/scene_grouped_audit.json"
TEXTILES=(1 2 3 4 5 6 7 14 16 18 21 23 24 31 38)

cd "$ROOT"
export PYTHONPATH="$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/logs"

for textile in "${TEXTILES[@]}"; do
  output_file=$(printf '%s/textile_%02d.json' "$OUTPUT" "$textile")
  log_file=$(printf '%s/logs/textile_%02d.log' "$OUTPUT" "$textile")
  if [[ -e "$output_file" ]]; then
    echo "Refusing to overwrite existing result: $output_file" >&2
    exit 2
  fi
  echo "[$(date --iso-8601=seconds)] textile=$textile start" | tee -a "$log_file"
  "$PYTHON" tools/run_olp_pcaf_external.py "$CONFIG" "$output_file" \
    --textile-id "$textile" 2>&1 | tee -a "$log_file"
  echo "[$(date --iso-8601=seconds)] textile=$textile complete" | tee -a "$log_file"
done

"$PYTHON" tools/summarize_olp_pcaf_external.py \
  "$OUTPUT" "$AUDIT" \
  "$ROOT/runs/olp_pcaf_external/scene_grouped_summary.json" \
  "$ROOT/runs/olp_pcaf_external/README.md"

printf '{"state":"complete","message":"fifteen-textile OLP external validation complete"}\n' \
  > "$OUTPUT/runner_status.json"
