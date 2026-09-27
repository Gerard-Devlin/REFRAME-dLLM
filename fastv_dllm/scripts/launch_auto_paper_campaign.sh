#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for name in GSM8K_DATASET MATH_DATASET HUMANEVAL_DATASET MBPP_DATASET; do
  [[ -v "$name" ]] || { echo "$name is required"; exit 2; }
  printf -v "$name" '%s' "$(realpath -e "${!name}")"
done
DYNAMIC_GPU_COUNT="${DYNAMIC_GPU_COUNT:-6}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0,1,2,3,4,5,6,7}"
GPU_IDS="waiting-for-${DYNAMIC_GPU_COUNT}"
RUN_ROOT="$(realpath -m "${RUN_ROOT:-$REPO/fastv_dllm/runs/paper_auto_$(date +%Y%m%d_%H%M%S)_$$}")"
[[ ! -e "$RUN_ROOT" ]] || { echo 'Use a new RUN_ROOT'; exit 2; }
mkdir -p "$RUN_ROOT"
for name in REPO RUN_ROOT GPU_IDS DYNAMIC_GPU_COUNT GPU_CANDIDATES GPU_POLL_SECONDS GPU_STABLE_CHECKS GPU_MAX_MEMORY_MIB GPU_MAX_UTILIZATION GSM8K_DATASET MATH_DATASET HUMANEVAL_DATASET MBPP_DATASET CURRENT_GSM256_RUN CONDA_ROOT CONDA_ENV HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE; do
  if [[ -v "$name" ]]; then printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_ROOT/campaign.env"; fi
done
cat > "$RUN_ROOT/entry.sh" <<EOF
#!/usr/bin/env bash
source $(printf '%q' "$RUN_ROOT/campaign.env")
exec 9>$(printf '%q' "$REPO/fastv_dllm/runs/paper_auto.lock")
if ! flock -n 9; then
  echo 'Another automatic paper campaign already owns the lock.'
  exit 73
fi
date -Is > $(printf '%q' "$RUN_ROOT/started_at")
bash $(printf '%q' "$REPO/fastv_dllm/scripts/paper_campaign.sh") > $(printf '%q' "$RUN_ROOT/campaign.log") 2>&1
rc=\$?
printf '%s\n' "\$rc" > $(printf '%q' "$RUN_ROOT/exit_code")
date -Is > $(printf '%q' "$RUN_ROOT/finished_at")
exit "\$rc"
EOF
session="fv-paper-auto-$(date +%Y%m%d-%H%M%S)-$$"
tmux new-session -d -s "$session" "bash $(printf '%q' "$RUN_ROOT/entry.sh")"
printf '%s\n' "$session" > "$RUN_ROOT/tmux_session"
printf 'Session: %s\nRoot: %s\nLog: %s/campaign.log\n' "$session" "$RUN_ROOT" "$RUN_ROOT"
