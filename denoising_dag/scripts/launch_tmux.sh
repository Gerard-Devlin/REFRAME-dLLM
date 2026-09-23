#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-probe}"
[[ "$MODE" =~ ^(probe|generate|download)$ ]] || { echo 'Mode: probe, generate, download'; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${RUN_DIR:-$REPO/denoising_dag/runs/${MODE}_$(date +%Y%m%d_%H%M%S)}"
if [[ "$MODE" != download ]]; then
    [[ "${GPU_IDS:-}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Set GPU_IDS explicitly'; exit 2; }
fi
[[ ! -e "$RUN_DIR" ]] || { echo 'Use a fresh RUN_DIR'; exit 2; }
mkdir -p "$RUN_DIR"
for name in REPO RUN_DIR MODE GPU_IDS MODEL_SIZES LIMIT BATCH_SIZE DEPTHS WIDTH REPEATS SNAPSHOTS MAX_NEW_TOKENS THRESHOLD PROMPTS MIN_FREE_GIB CONDA_ROOT CONDA_ENV HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE DOWNLOAD_RETRIES DENOISING_DAG_DOWNLOAD_WORKERS HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_ETAG_TIMEOUT; do
    if [[ -v "$name" ]]; then printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_DIR/job.env"; fi
done
printf '#!/usr/bin/env bash\nsource %q\nbash %q > %q 2>&1\nrc=$?\nprintf "%%s\\n" "$rc" > %q\nexit "$rc"\n' \
    "$RUN_DIR/job.env" "$REPO/denoising_dag/scripts/job.sh" "$RUN_DIR/job.log" "$RUN_DIR/exit_code" > "$RUN_DIR/entry.sh"
session="state-dag-${MODE}-$(date +%Y%m%d-%H%M%S)-$$"
tmux new-session -d -s "$session" "bash $(printf '%q' "$RUN_DIR/entry.sh")"
printf 'Session: %s\nLog: %s/job.log\nAttach: tmux attach -t %s\n' "$session" "$RUN_DIR" "$session"
