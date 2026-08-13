#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Fabric_Defect_Detection
python_bin=.venv-anomalib/bin/python
config=configs/raw_fabrid_context_bias_field_k3.json
output_dir=runs/raw_fabrid_context_bias_field_k3/final_reproduction
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

for index in 1 2 3 4 5 6 7; do
  current_fold="Rollo${index}A"
  checkpoint="runs/raw_fabrid_patchcore_k0/instance_audit/lightning_${current_fold}/Patchcore/v0/weights/lightning/model.ckpt"
  if [[ ! -f "$checkpoint" ]]; then
    printf 'Missing fold checkpoint: %s\n' "$checkpoint" >&2
    exit 1
  fi
  result="$output_dir/${current_fold}.json"
  log="$output_dir/${current_fold}.log"
  write_status running "$current_fold" "current-script final reproduction"
  printf 'START %s %s\n' "$current_fold" "$(date --iso-8601=seconds)"
  "$python_bin" tools/probe_raw_fabrid_physical_field_k0.py \
    "$config" "$result" --model patchcore --fold "$current_fold" \
    --checkpoint "$checkpoint" >"$log" 2>&1
  printf 'DONE %s %s\n' "$current_fold" "$(date --iso-8601=seconds)"
done

write_status complete all "seven-fold final reproduction complete"
