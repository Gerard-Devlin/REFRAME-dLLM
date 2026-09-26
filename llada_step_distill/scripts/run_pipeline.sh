#!/usr/bin/env bash
set -euo pipefail

GPU_IDS=${GPU_IDS:?set GPU_IDS, for example 0,2,3,4,5,6}
EXP_ROOT=${EXP_ROOT:?set EXP_ROOT}
NEMOTRON_ROOT=${NEMOTRON_ROOT:?set NEMOTRON_ROOT}
DATASET=${DATASET:?set DATASET}

PREPARED=${PREPARED:-$EXP_ROOT/prepared}
SUPERVISION=${SUPERVISION:-$EXP_ROOT/supervision}
RUN_ROOT=${RUN_ROOT:-$EXP_ROOT/pipeline}

export GPU_IDS EXP_ROOT NEMOTRON_ROOT DATASET PREPARED SUPERVISION RUN_ROOT
export TRAIN_SIZE=${TRAIN_SIZE:-10000000}
export VALIDATION_SIZE=${VALIDATION_SIZE:-20000}
export SHARD_SIZE=${SHARD_SIZE:-2048}
export OVERSAMPLE=${OVERSAMPLE:-1.25}
export SPLIT=${SPLIT:-all}
export ATTEMPTS=${ATTEMPTS:-8}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$EXP_ROOT" "$RUN_ROOT"

stage_complete() {
  local directory=$1
  [[ -f "$directory/exit_code" ]] && [[ "$(tr -d '[:space:]' < "$directory/exit_code")" == 0 ]]
}

run_stage() {
  local name=$1
  local mode=$2
  shift 2
  local directory="$RUN_ROOT/$name"
  if stage_complete "$directory"; then
    echo "[$(date -Is)] skip completed stage: $name"
    return 0
  fi
  mkdir -p "$directory"
  echo "[$(date -Is)] start stage: $name"
  set +e
  (
    export RUN_DIR="$directory"
    unset OVERFIT_RECORDS UPDATES INIT_ADAPTER RESUME ADAPTER MERGED_MODEL CHECKPOINT
    for assignment in "$@"; do
      export "$assignment"
    done
    bash llada_step_distill/scripts/run.sh "$mode"
  ) 2>&1 | tee "$directory/job.log"
  local code=${PIPESTATUS[0]}
  set -e
  echo "$code" > "$directory/exit_code"
  if (( code != 0 )); then
    echo "[$(date -Is)] stage failed: $name (exit $code)" >&2
    exit "$code"
  fi
  echo "[$(date -Is)] finish stage: $name"
}

prepared_complete() {
  [[ -f "$PREPARED/manifest.json" ]] || return 1
  python - "$PREPARED/manifest.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["train_size"] == 10_000_000
assert d["validation_size"] == 20_000
assert d["acceleration_records"] == 1_000_000
assert d["retention_records"] == 9_000_000
assert d["real_transition_records"] == 200_000
p = d["trajectory_policy"]
assert p["complete_trajectories"] is False
assert p["states_per_acceleration_example"] == 1
PY
}

if prepared_complete; then
  echo "[$(date -Is)] prepared data already complete"
else
  if [[ -d "$PREPARED" ]] && find "$PREPARED" -mindepth 1 -print -quit | grep -q .; then
    echo "Partial prepared directory exists without a valid manifest: $PREPARED" >&2
    exit 20
  fi
  run_stage prepare prepare
  prepared_complete
fi

if [[ -f "$SUPERVISION/manifest.json" ]]; then
  echo "[$(date -Is)] supervision already complete"
else
  if [[ -d "$SUPERVISION" ]] && find "$SUPERVISION" -mindepth 1 -print -quit | grep -q .; then
    echo "Partial supervision directory exists without a manifest: $SUPERVISION" >&2
    exit 21
  fi
  run_stage collect collect
  [[ -f "$SUPERVISION/manifest.json" ]]
fi

run_stage overfit32_a smoke-a "OVERFIT_RECORDS=32" "UPDATES=${OVERFIT_UPDATES:-200}"
run_stage smoke200_a smoke-a "UPDATES=${SMOKE_UPDATES:-200}"

latest_checkpoint() {
  local directory=$1
  local candidate
  candidate=$(find "$directory" -maxdepth 1 -type d -name 'checkpoint_*' -print 2>/dev/null | sort | tail -n 1)
  [[ -n "$candidate" ]] && [[ -f "$candidate/complete.json" ]] && printf '%s\n' "$candidate"
}

best_or_latest_checkpoint() {
  local directory=$1
  if [[ -f "$directory/best.txt" ]]; then
    local best="$directory/$(tr -d '[:space:]' < "$directory/best.txt")"
    if [[ -f "$best/complete.json" ]]; then
      printf '%s\n' "$best"
      return 0
    fi
  fi
  latest_checkpoint "$directory"
}

STAGE_A_DIR="$RUN_ROOT/stage_a"
if ! stage_complete "$STAGE_A_DIR"; then
  A_RESUME=$(latest_checkpoint "$STAGE_A_DIR" || true)
  if [[ -n "$A_RESUME" ]]; then
    run_stage stage_a train-a "RESUME=$A_RESUME"
  else
    run_stage stage_a train-a
  fi
fi
A_CHECKPOINT=$(best_or_latest_checkpoint "$STAGE_A_DIR")
[[ -n "$A_CHECKPOINT" ]] || { echo "No complete Stage-A checkpoint" >&2; exit 22; }

STAGE_B_DIR="$RUN_ROOT/stage_b"
if ! stage_complete "$STAGE_B_DIR"; then
  B_RESUME=$(latest_checkpoint "$STAGE_B_DIR" || true)
  if [[ -n "$B_RESUME" ]]; then
    run_stage stage_b train-b "RESUME=$B_RESUME"
  else
    run_stage stage_b train-b "INIT_ADAPTER=$A_CHECKPOINT"
  fi
fi
B_CHECKPOINT=$(best_or_latest_checkpoint "$STAGE_B_DIR")
[[ -n "$B_CHECKPOINT" ]] || { echo "No complete Stage-B checkpoint" >&2; exit 23; }

run_stage export_a export "CHECKPOINT=$A_CHECKPOINT"
run_stage export_b export "CHECKPOINT=$B_CHECKPOINT"

if [[ "${PIPELINE_EVALUATE:-1}" == 1 ]]; then
  for split in dev holdout; do
    run_stage "eval_teacher_$split" evaluate "SPLIT=$split" "STEPS=8 16 32"
    run_stage "eval_stage_a_$split" evaluate "SPLIT=$split" "STEPS=8 16 32" \
      "MERGED_MODEL=$RUN_ROOT/export_a/exported"
    run_stage "eval_stage_b_$split" evaluate "SPLIT=$split" "STEPS=8 16 32" \
      "MERGED_MODEL=$RUN_ROOT/export_b/exported"
  done
fi

echo "[$(date -Is)] complete pipeline: $RUN_ROOT"
