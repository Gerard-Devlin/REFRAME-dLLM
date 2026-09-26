#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_IDS="${GPU_IDS:?Explicit physical GPU_IDS required}"
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU_IDS'; exit 2; }
for name in GSM8K_DATASET MATH_DATASET HUMANEVAL_DATASET MBPP_DATASET; do
  [[ -v "$name" ]] || { echo "$name is required"; exit 2; }
  printf -v "$name" '%s' "$(realpath -e "${!name}")"
done
RUN_ROOT="$(realpath -m "${RUN_ROOT:-$REPO/fastv_dllm/runs/paper_campaign_$(date +%Y%m%d_%H%M%S)_$$}")"
[[ ! -e "$RUN_ROOT" ]] || { echo 'Use a new RUN_ROOT'; exit 2; }
mkdir -p "$RUN_ROOT"
for name in REPO RUN_ROOT GPU_IDS GSM8K_DATASET MATH_DATASET HUMANEVAL_DATASET MBPP_DATASET CURRENT_GSM256_RUN CONDA_ROOT CONDA_ENV HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE; do
  if [[ -v "$name" ]]; then printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_ROOT/campaign.env"; fi
done
cat > "$RUN_ROOT/entry.sh" <<EOF
#!/usr/bin/env bash
source $(printf '%q' "$RUN_ROOT/campaign.env")
date -Is > $(printf '%q' "$RUN_ROOT/started_at")
bash $(printf '%q' "$REPO/fastv_dllm/scripts/paper_campaign.sh") > $(printf '%q' "$RUN_ROOT/campaign.log") 2>&1
rc=\$?
printf '%s\n' "\$rc" > $(printf '%q' "$RUN_ROOT/exit_code")
date -Is > $(printf '%q' "$RUN_ROOT/finished_at")
exit "\$rc"
EOF
session="fv-paper-$(date +%Y%m%d-%H%M%S)-$$"
tmux new-session -d -s "$session" "bash $(printf '%q' "$RUN_ROOT/entry.sh")"
printf '%s\n' "$session" > "$RUN_ROOT/tmux_session"
printf 'Session: %s\nRoot: %s\nLog: %s/campaign.log\n' "$session" "$RUN_ROOT" "$RUN_ROOT"
