#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"

wait_for_run() {
    local run="$1"
    [[ -n "$run" ]] || return 0
    echo "Waiting for prerequisite run: $run"
    while [[ ! -f "$run/exit_code" ]]; do sleep 30; done
    if [[ $(cat "$run/exit_code") != 0 ]]; then
        # A benchmark can finish all shards and write the atomic summary before
        # a later, unrelated shell epilogue fails.  Reuse it only after a
        # strict completeness check; never infer success from the summary's
        # mere existence.
        python - "$run/output" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
expected = len(summary["ids"])
records = []
for path in sorted(root.glob("rank_*.jsonl")):
    records.extend(json.loads(line) for line in path.read_text().splitlines())
indices = sorted(row["index"] for row in records)
if len(records) != expected or indices != list(range(expected)):
    raise SystemExit(f"Incomplete prerequisite: records={len(records)} expected={expected}")
methods = [key for key in summary["results"] if key != "attribution"]
if not methods or any(summary["results"][key]["examples"] != expected for key in methods):
    raise SystemExit("Prerequisite summary is incomplete")
print(f"Validated completed prerequisite despite shell exit: {expected} examples")
PY
    fi
}

run_one() {
    local task="$1" gen="$2" label="$3" cache="$4" decoding="$5"
    local methods="$6" dataset="$7" limit="$8"
    local run="$RUN_ROOT/${task}_g${gen}_${label}"
    if [[ -f "$run/exit_code" && $(cat "$run/exit_code") == 0 ]]; then
        echo "Already complete: $run"
        return 0
    fi
    [[ ! -e "$run" ]] || { echo "Incomplete run exists; refusing overwrite: $run"; exit 1; }
    mkdir -p "$run"
    echo "[$(date -Is)] START task=$task gen=$gen label=$label" | tee -a "$RUN_ROOT/progress.log"
    set +e
    MODE=evaluate RUN_DIR="$run" TASK="$task" DATASET="$dataset" LIMIT="$limit" \
      GEN_LENGTH="$gen" BLOCK_LENGTH=32 THRESHOLD=0.90 DECODING_MODE="$decoding" \
      CACHE_MODE="$cache" METHODS="$methods" PRUNE_AFTER_LAYER=4 \
      SUPPORT_KEEP_RATIO=0.3125 REQUIRE_IDLE=1 SKIP_TESTS=1 \
      bash "$REPO/fastv_dllm/scripts/job.sh" > "$run/job.log" 2>&1
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$run/exit_code"
    date -Is > "$run/finished_at"
    echo "[$(date -Is)] END rc=$rc task=$task gen=$gen label=$label" | tee -a "$RUN_ROOT/progress.log"
    [[ $rc == 0 ]] || exit "$rc"
}

wait_for_run "${CURRENT_GSM256_RUN:-}"
python -m pytest fastv_dllm/tests -q | tee "$RUN_ROOT/tests.log"

declare -A DATASETS=(
  [gsm8k]="$GSM8K_DATASET"
  [math]="$MATH_DATASET"
  [humaneval]="$HUMANEVAL_DATASET"
  [mbpp]="$MBPP_DATASET"
)
declare -A LIMITS=([gsm8k]=1319 [math]=5000 [humaneval]=164 [mbpp]=500)

for task in gsm8k math humaneval mbpp; do
  for gen in 256 512; do
    run_one "$task" "$gen" llada none single "flash_native" "${DATASETS[$task]}" "${LIMITS[$task]}"
    run_one "$task" "$gen" cache prefix single "flash_native" "${DATASETS[$task]}" "${LIMITS[$task]}"
    if [[ "$task" == gsm8k && "$gen" == 256 && -n ${CURRENT_GSM256_RUN:-} ]]; then
      target="$RUN_ROOT/gsm8k_g256_parallel_ours"
      [[ -e "$target" || -L "$target" ]] || ln -s "$CURRENT_GSM256_RUN" "$target"
      echo "[$(date -Is)] REUSE $CURRENT_GSM256_RUN" | tee -a "$RUN_ROOT/progress.log"
    else
      run_one "$task" "$gen" parallel_ours none threshold \
        "flash_native flash_fastv_head" "${DATASETS[$task]}" "${LIMITS[$task]}"
    fi
    run_one "$task" "$gen" fastdllm_ours_cache prefix threshold \
      "flash_native flash_fastv_head" "${DATASETS[$task]}" "${LIMITS[$task]}"
  done
done

date -Is > "$RUN_ROOT/campaign_complete"
echo "Complete: $RUN_ROOT" | tee -a "$RUN_ROOT/progress.log"
