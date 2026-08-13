#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Fabric_Defect_Detection
python_bin=.venv-anomalib/bin/python
config=configs/raw_fabrid_dual_readout_fusion_k2.json
output_dir=runs/raw_fabrid_dual_readout_fusion_k2/all_folds
mkdir -p "$output_dir"

status="$output_dir/runner_status.json"
write_status() {
  local state="$1"
  local fold="$2"
  local message="$3"
  printf '{\n  "state": "%s",\n  "fold": "%s",\n  "message": "%s",\n  "pid": %d,\n  "updated_at": "%s"\n}\n' \
    "$state" "$fold" "$message" "$$" "$(date --iso-8601=seconds)" > "$status"
}
trap 'write_status failed "${current_fold:-none}" "runner exited before completion"' ERR

run_fold() {
  current_fold="$1"
  checkpoint="runs/raw_fabrid_patchcore_k0/instance_audit/lightning_${current_fold}/Patchcore/v0/weights/lightning/model.ckpt"
  reference="runs/raw_fabrid_context_bias_field_k3/final_reproduction/${current_fold}.json"
  result="$output_dir/${current_fold}.json"
  log="$output_dir/${current_fold}.log"
  if [[ ! -f "$checkpoint" || ! -f "$reference" ]]; then
    printf 'Missing fold asset: %s or %s\n' "$checkpoint" "$reference" >&2
    exit 1
  fi
  write_status running "$current_fold" "K2 dual-readout replay"
  printf 'START %s %s\n' "$current_fold" "$(date --iso-8601=seconds)"
  "$python_bin" tools/probe_raw_fabrid_dual_readout_fusion_k2.py \
    "$config" "$result" \
    --fold "$current_fold" \
    --checkpoint "$checkpoint" \
    --reference "$reference" >"$log" 2>&1
  printf 'DONE %s %s\n' "$current_fold" "$(date --iso-8601=seconds)"
}

# Rollo4A is disclosed development data. It is rerun only to verify that the
# new dual-readout implementation reproduces the already-known localizer and
# the frozen PCAF parent alarm before any confirmation fold is touched.
run_fold Rollo4A
"$python_bin" -c '
import json
from pathlib import Path
result = json.loads(Path("runs/raw_fabrid_dual_readout_fusion_k2/all_folds/Rollo4A.json").read_text(encoding="utf-8"))
check = result["development_reproduction"]
if check is None or not check["passed"]:
    raise SystemExit("Rollo4A development reproduction failed; confirmation is prohibited")
'

for index in 1 2 3 5 6 7; do
  run_fold "Rollo${index}A"
done

"$python_bin" tools/summarize_raw_fabrid_dual_readout_fusion_k2.py \
  "$output_dir" \
  "runs/raw_fabrid_dual_readout_fusion_k2/seven_fold_summary.json" \
  "runs/raw_fabrid_dual_readout_fusion_k2/README.md"

write_status complete all "K2 development reproduction and six-roll confirmation complete"
