#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/Fabric_Defect_Detection}"
OUTPUT="${2:-$ROOT/runs/raw_fabrid_dual_readout_fusion_k2/qualitative}"
PYTHON="$ROOT/.venv-anomalib/bin/python"
CONFIG="$ROOT/configs/raw_fabrid_dual_readout_fusion_k2.json"
FOLDS=(Rollo1A Rollo2A Rollo3A Rollo5A Rollo6A Rollo7A)

cd "$ROOT"
export PYTHONPATH="$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/logs"

for fold in "${FOLDS[@]}"; do
  checkpoint="$ROOT/runs/raw_fabrid_patchcore_k0/instance_audit/lightning_${fold}/Patchcore/v0/weights/lightning/model.ckpt"
  reference="$ROOT/runs/raw_fabrid_dual_readout_fusion_k2/all_folds/${fold}.json"
  cache="$OUTPUT/${fold}.npz"
  audit="$OUTPUT/${fold}.json"
  log="$OUTPUT/logs/${fold}.log"
  if [[ ! -f "$checkpoint" || ! -f "$reference" ]]; then
    printf 'Missing fold asset: %s or %s\n' "$checkpoint" "$reference" >&2
    exit 1
  fi
  if [[ -e "$cache" || -e "$audit" ]]; then
    printf 'Refusing to overwrite qualitative cache: %s\n' "$fold" >&2
    exit 2
  fi
  "$PYTHON" tools/probe_raw_fabrid_pcdr_qualitative_k2.py \
    "$CONFIG" "$cache" "$audit" \
    --fold "$fold" --checkpoint "$checkpoint" --reference "$reference" \
    >"$log" 2>&1
done

"$PYTHON" tools/select_raw_fabrid_pcdr_qualitative_k2.py \
  "$OUTPUT" "$ROOT/data/RAW_FABRID" \
  "$OUTPUT/qualitative_pcdr.npz" "$OUTPUT/qualitative_pcdr_selection.json"

