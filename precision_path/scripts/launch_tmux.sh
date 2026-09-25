#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-}"
[[ "$MODE" == cost || "$MODE" == audit || "$MODE" == calibrate ]] || { echo 'Usage: launch_tmux.sh cost|audit|calibrate'; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$MODE" != calibrate ]]; then
    GPU_IDS="${GPU_IDS:?Set physical GPU_IDS explicitly, e.g. 3 or 0,2,3}"
    [[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU_IDS'; exit 2; }
    DATA_DIR="${DATA_DIR:?Set existing prepared DATA_DIR}"
    DATA_DIR="$(realpath -e "$DATA_DIR")"
    [[ -s "$DATA_DIR/manifest.json" ]] || { echo 'Prepared manifest missing'; exit 2; }
fi
if [[ "$MODE" == audit ]]; then
    COST_REPORT="$(realpath -e "${COST_REPORT:?Set passed cost summary.json}")"
fi
if [[ "$MODE" == calibrate ]]; then
    CALIBRATION_REPORT="$(realpath -e "${CALIBRATION_REPORT:?Set calibration-role audit summary.json}")"
    EVALUATION_REPORT="$(realpath -e "${EVALUATION_REPORT:?Set independent evaluation-role audit summary.json}")"
fi
RUN_DIR="${RUN_DIR:-$REPO/precision_path/runs/${MODE}_$(date +%Y%m%d_%H%M%S)_$$}"
RUN_DIR="$(realpath -m "$RUN_DIR")"
[[ ! -e "$RUN_DIR" ]] || { echo 'RUN_DIR exists; choose a new directory'; exit 2; }
mkdir -p "$RUN_DIR"
for name in REPO MODE RUN_DIR GPU_IDS DATA_DIR COST_REPORT CALIBRATION_REPORT EVALUATION_REPORT ROLE PROMPTS MAX_NEW_TOKENS THRESHOLD REPEATS MAX_STATES SEED ALPHA CONDA_ROOT CONDA_ENV HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE; do
    if [[ -v "$name" ]]; then printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_DIR/job.env"; fi
done
printf '#!/usr/bin/env bash\nsource %q\ndate -Is > %q\nbash %q > %q 2>&1\nrc=$?\nprintf "%%s\\n" "$rc" > %q\ndate -Is > %q\nexit "$rc"\n' \
    "$RUN_DIR/job.env" "$RUN_DIR/started_at" "$REPO/precision_path/scripts/job.sh" \
    "$RUN_DIR/job.log" "$RUN_DIR/exit_code" "$RUN_DIR/finished_at" > "$RUN_DIR/entry.sh"
session="pp-${MODE}-$(date +%Y%m%d-%H%M%S)-$$"
tmux new-session -d -s "$session" "bash $(printf '%q' "$RUN_DIR/entry.sh")"
printf '%s\n' "$session" > "$RUN_DIR/tmux_session"
printf 'Session: %s\nLog: %s/job.log\nAttach: tmux attach -t %s\n' "$session" "$RUN_DIR" "$session"
