#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Fabric_Defect_Detection
python_bin=.venv-anomalib/bin/python
config=configs/raw_fabrid_context_bias_field_k3.json
output_dir=runs/raw_fabrid_coordinate_fixed_effect_audit/full
mkdir -p "$output_dir"

status="$output_dir/runner_status.json"
write_status() {
  local state="$1"
  local fold="$2"
  local message="$3"
  printf '{\n  "state": "%s",\n  "fold": "%s",\n  "message": "%s",\n  "pid": %d,\n  "updated_at": "%s"\n}\n' \
    "$state" "$fold" "$message" "$$" "$(date --iso-8601=seconds)" > "$status"
}
trap 'write_status failed "${current_fold:-none}" "fixed-effect audit runner failed"' ERR

for index in 1 2 3 4 5 6 7; do
  current_fold="Rollo${index}A"
  checkpoint="runs/raw_fabrid_patchcore_k0/instance_audit/lightning_${current_fold}/Patchcore/v0/weights/lightning/model.ckpt"
  result="$output_dir/${current_fold}.json"
  calibration="$output_dir/${current_fold}_calibration.npz"
  log="$output_dir/${current_fold}.log"
  if [[ -s "$result" && -s "$calibration" ]]; then
    printf 'SKIP %s existing audited outputs\n' "$current_fold"
    continue
  fi
  if [[ ! -f "$checkpoint" ]]; then
    printf 'Missing fold checkpoint: %s\n' "$checkpoint" >&2
    exit 1
  fi
  write_status running "$current_fold" "physical-location fixed-effect audit"
  printf 'START %s %s\n' "$current_fold" "$(date --iso-8601=seconds)"
  "$python_bin" tools/audit_raw_fabrid_coordinate_fixed_effects.py \
    "$config" "$result" "$calibration" --fold "$current_fold" \
    --checkpoint "$checkpoint" > "$log" 2>&1
  printf 'DONE %s %s\n' "$current_fold" "$(date --iso-8601=seconds)"
done

write_status complete all "seven-fold fixed-effect audit complete"
