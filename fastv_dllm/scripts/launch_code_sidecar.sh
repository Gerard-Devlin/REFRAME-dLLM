#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="$(realpath -e "${RUN_ROOT:?Existing main campaign RUN_ROOT is required}")"
GPU_IDS="${GPU_IDS:?Explicit physical GPU_IDS required}"
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU_IDS'; exit 2; }
HUMANEVAL_DATASET="$(realpath -e "${HUMANEVAL_DATASET:?HUMANEVAL_DATASET is required}")"
MBPP_DATASET="$(realpath -e "${MBPP_DATASET:?MBPP_DATASET is required}")"
SIDECAR="$RUN_ROOT/sidecar_code"
[[ ! -e "$SIDECAR" ]] || { echo 'Code sidecar already exists'; exit 2; }
mkdir -p "$SIDECAR"
for name in REPO RUN_ROOT SIDECAR GPU_IDS HUMANEVAL_DATASET MBPP_DATASET CONDA_ROOT CONDA_ENV HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE; do
  if [[ -v "$name" ]]; then printf 'export %s=%q\n' "$name" "${!name}" >> "$SIDECAR/sidecar.env"; fi
done
cat > "$SIDECAR/entry.sh" <<'ENTRY'
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/sidecar.env"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${CONDA_ROOT:-/opt/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-fastdllm311}"
cd "$REPO"
exec 9>"$REPO/fastv_dllm/runs/paper_code_sidecar.lock"
flock -n 9 || { echo 'Another code sidecar owns the lock'; exit 73; }

wait_cards() {
  python -u -m fastv_dllm.wait_for_idle_gpus --count 2 --candidates "$GPU_IDS" \
    --poll-seconds 30 --stable-checks 2 --max-memory-mib 1024 --max-utilization 5 >/dev/null
}

run_one() {
  local task="$1" gen="$2" label="$3" cache="$4" decoding="$5"
  local methods="$6" dataset="$7" limit="$8"
  local run="$RUN_ROOT/${task}_g${gen}_${label}"
  if [[ -f "$run/exit_code" && $(cat "$run/exit_code") == 0 ]]; then
    echo "SKIP complete: $run" | tee -a "$RUN_ROOT/sidecar_code_progress.log"
    return
  fi
  [[ ! -e "$run" ]] || { echo "Target already exists: $run"; exit 1; }
  while true; do
    wait_cards
    mkdir -p "$run"
    echo "[$(date -Is)] START task=$task gen=$gen label=$label GPUs=$GPU_IDS" \
      | tee -a "$RUN_ROOT/sidecar_code_progress.log"
    set +e
    MODE=evaluate RUN_DIR="$run" TASK="$task" DATASET="$dataset" LIMIT="$limit" \
      GEN_LENGTH="$gen" BLOCK_LENGTH=32 THRESHOLD=0.90 DECODING_MODE="$decoding" \
      CACHE_MODE="$cache" METHODS="$methods" PRUNE_AFTER_LAYER=4 \
      SUPPORT_KEEP_RATIO=0.3125 REQUIRE_IDLE=1 SKIP_TESTS=1 \
      bash "$REPO/fastv_dllm/scripts/job.sh" >"$run/job.log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" >"$run/exit_code"
    date -Is >"$run/finished_at"
    echo "[$(date -Is)] END rc=$rc task=$task gen=$gen label=$label GPUs=$GPU_IDS" \
      | tee -a "$RUN_ROOT/sidecar_code_progress.log"
    [[ $rc == 0 ]] && break
    failed="${run}.sidecar_failed_$(date +%Y%m%d_%H%M%S)"
    mv "$run" "$failed"
    if grep -Eq 'Selected GPU already has compute PID|Visible GPU count mismatch' "$failed/job.log"; then
      echo "GPU race; retrying: $failed" | tee -a "$RUN_ROOT/sidecar_code_progress.log"
      continue
    fi
    exit "$rc"
  done
}

python -m pytest fastv_dllm/tests -q
for task in humaneval mbpp; do
  if [[ "$task" == humaneval ]]; then dataset="$HUMANEVAL_DATASET"; limit=164
  else dataset="$MBPP_DATASET"; limit=500; fi
  for gen in 256 512; do
    run_one "$task" "$gen" llada none single "flash_native" "$dataset" "$limit"
    run_one "$task" "$gen" cache prefix single "flash_native" "$dataset" "$limit"
    run_one "$task" "$gen" parallel_ours none threshold \
      "flash_native flash_fastv_head" "$dataset" "$limit"
    run_one "$task" "$gen" fastdllm_ours_cache prefix threshold \
      "flash_native flash_fastv_head" "$dataset" "$limit"
  done
done
date -Is >"$SIDECAR/complete"
ENTRY
cat > "$SIDECAR/wrapper.sh" <<EOF
#!/usr/bin/env bash
bash $(printf '%q' "$SIDECAR/entry.sh") >$(printf '%q' "$SIDECAR/sidecar.log") 2>&1
rc=\$?
printf '%s\n' "\$rc" >$(printf '%q' "$SIDECAR/exit_code")
exit "\$rc"
EOF
session="fv-code-$(date +%Y%m%d-%H%M%S)-$$"
tmux new-session -d -s "$session" "bash $(printf '%q' "$SIDECAR/wrapper.sh")"
printf '%s\n' "$session" >"$SIDECAR/tmux_session"
printf 'Session: %s\nSidecar: %s\nLog: %s/sidecar.log\n' "$session" "$SIDECAR" "$SIDECAR"
