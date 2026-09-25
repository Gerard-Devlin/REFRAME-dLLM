#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: run.sh MODE}
GPU_IDS=${GPU_IDS:-0}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
IFS=',' read -r -a _GPUS <<< "$GPU_IDS"
NPROC=${#_GPUS[@]}

distributed() {
  torchrun --standalone --nproc_per_node="$NPROC" -m llada_step_distill "$@"
}

train_extras() {
  EXTRA_ARGS=()
  [[ -n "${INIT_ADAPTER:-}" ]] && EXTRA_ARGS+=(--init-adapter "$INIT_ADAPTER")
  [[ -n "${RESUME:-}" ]] && EXTRA_ARGS+=(--resume "$RESUME")
  [[ -n "${DATASET:-}" ]] && EXTRA_ARGS+=(--dev-dataset "$DATASET")
  [[ -n "${OVERFIT_RECORDS:-}" ]] && EXTRA_ARGS+=(--overfit-records "$OVERFIT_RECORDS")
}

case "$MODE" in
  prepare)
    python -u -m llada_step_distill prepare \
      --nemotron-root "${NEMOTRON_ROOT:?set NEMOTRON_ROOT}" \
      --gsm8k-json "${DATASET:?set DATASET}" \
      --output "${PREPARED:?set PREPARED}" \
      --train-size "${TRAIN_SIZE:-10000000}" --validation-size "${VALIDATION_SIZE:-20000}" \
      --shard-size "${SHARD_SIZE:-2048}" --oversample "${OVERSAMPLE:-1.05}"
    ;;
  collect)
    distributed collect --prepared "${PREPARED:?set PREPARED}" \
      --output "${SUPERVISION:?set SUPERVISION}" --split "${SPLIT:-all}" --attempts "${ATTEMPTS:-8}"
    ;;
  audit)
    python -u -m llada_step_distill audit --dataset "${DATASET:?set DATASET}" \
      --output "$RUN_DIR/audit" --limit "${LIMIT:-256}"
    ;;
  smoke-a|smoke-b)
    STAGE=${MODE#smoke-}
    train_extras
    distributed smoke --prepared "${PREPARED:?set PREPARED}" --supervision "${SUPERVISION:?set SUPERVISION}" \
      --output "$RUN_DIR" --stage "$STAGE" --updates "${UPDATES:-200}" \
      "${EXTRA_ARGS[@]}"
    ;;
  train-a|train-b)
    STAGE=${MODE#train-}
    train_extras
    distributed train --prepared "${PREPARED:?set PREPARED}" --supervision "${SUPERVISION:?set SUPERVISION}" \
      --output "$RUN_DIR" --stage "$STAGE" \
      "${EXTRA_ARGS[@]}"
    ;;
  evaluate)
    EVAL_ARGS=()
    [[ -n "${ADAPTER:-}" ]] && EVAL_ARGS+=(--adapter "$ADAPTER")
    [[ -n "${LIMIT:-}" ]] && EVAL_ARGS+=(--limit "$LIMIT")
    read -r -a STEP_ARGS <<< "${STEPS:-8 16 32}"
    distributed evaluate --dataset "${DATASET:?set DATASET}" --output "$RUN_DIR" \
      --split "${SPLIT:-dev}" --steps "${STEP_ARGS[@]}" "${EVAL_ARGS[@]}"
    ;;
  export)
    python -u -m llada_step_distill export --checkpoint "${CHECKPOINT:?set CHECKPOINT}" --output "$RUN_DIR/exported"
    ;;
  *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac
