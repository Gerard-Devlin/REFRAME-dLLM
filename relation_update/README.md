# Conditional relation updates

The 119k-parameter residual-head pilot is paused: its held-out action predictions
did not improve over reuse. Existing training commands and results remain for
reproduction. Do not scale this head or inject it into online generation on the
basis of a small KL improvement.

## Full-trajectory action-replay oracle

`bash relation_update/scripts/launch_tmux.sh oracle` runs a separate **offline
cost-space diagnostic**. Set `GPU_IDS`, `DATA_DIR`, a new `TRACE_DIR`, and optionally
`ORACLE_PROMPTS=32`, `TIMING_REPEATS=2`, `MAX_NEW_TOKENS=512`, `THRESHOLD=0.90`.
All selected GPUs process disjoint held-out prompts. No head is trained or used.

For every prompt it runs warmed, unwrapped native generation for timing, a
CUDA-event forward timing pass, and an action observation pass. It checks identical
outputs and forward inputs between the latter passes, and verifies the recorded
commit action against the next actual native input. These repeated runs are
offline measurement overhead, not a proposed online decoding procedure.

Every call is retained; the old 32-transition reservoir cannot supply this bound.
The supported policy is deliberately narrow: same subblock, no EOS/MASK proposals,
no cache writes or prefill, no consecutive skipped calls, and no unverified terminal
action. A dynamic program finds the maximum forward time saved by a perfect
future-aware oracle under these constraints. Subblock crossings are excluded,
not silently counted as failures. This is a ceiling for this scope only.

`TRACE_DIR/summary.json` contains aggregate modeled speedup, native latency,
selected calls, timing validity, and the 1.5x cost target. Each `prompts/*.json`
retains the complete actions, old-probability replay actions, exclusion reasons,
call times, and selected indices, plus gate-cost sensitivity at 0.1/0.5/0.85 ms.
Only full-model forward cost is removed; sampler/commit work remains. Gate and
replay costs are omitted from the zero-overhead bound. If separately measured
forward costs exceed native latency, the latency estimate is invalidated.
The separate `forward_only_modeled_ceiling` uses one timing pass's forward costs
for both numerator and denominator and sets all non-forward work to zero. This
more optimistic cost ceiling remains interpretable when cross-pass timing is
inconsistent, but is still conditional on measured call weights, not a measured
end-to-end improvement or a noise-free hardware guarantee.

A high oracle bound says nothing about whether a cheap gate can identify those
states. No local risk guarantee, end-to-end quality, or realized speedup is claimed.
This stage does not implement a gate or online skipping.

This experiment replaces relation-token recoding with a small output-side model
of how a frozen Fast-dLLM v2 prediction changes after an actual native commit.
The tokenizer, embeddings, Transformer, LM head and native generation policy
remain unchanged. This is an **offline diagnosis**, not an online acceleration
implementation or a claim that quality is preserved.

## What is learned

For a native state `s` and its next state `s'`, the head learns a residual on
the current log probabilities using current hidden states and newly committed
token features. Teacher predictions at `s'` are targets only. They cannot supply
head input features or add future candidates.

The output partition is the current top-K candidates plus an **OTHER** bucket
for all remaining probability mass. Candidate coverage and teacher argmax
changes matter: high agreement on unchanged positions alone does not establish
that the head can substitute for another forward. OTHER is not a generated
token and does not identify a missing candidate.

Controls include unchanged probabilities, a calibration head, a head blind to
the committed tokens and their locations, and the conditional relation head.
Blind and conditional heads have matching parameter counts. Only these small
heads are trained; there is no relation-code relabeling, backbone full tuning,
or hidden LoRA update.

## Execution

Use the existing environment and prepared data. The launcher does not install
packages or download weights. The pinned original 1.5B model must already be in
the Hugging Face cache. `GPU_IDS` names physical GPUs and is mandatory; a single
GPU or any selected number of GPUs is supported. All selected GPUs participate
in collection or distributed head training.

```bash
export GPU_IDS=0,1,2,3
export DATA_DIR=/path/to/prepared/data
export TRACE_DIR=/path/to/new/conditional_traces
bash relation_update/scripts/launch_tmux.sh collect

# Run only after collection has completed successfully and written manifest.json.
bash relation_update/scripts/launch_tmux.sh train
```

Each command starts an independent tmux session and prints its log path. Runs
write `job.log`, `exit_code`, timestamps and the exact exported job settings into
a fresh `relation_update/runs/` directory. `RUN_DIR` can override that directory;
existing directories are rejected to protect prior results, except explicit
training resumes. Collection requires
a new `TRACE_DIR`; training requires its completed manifest. Wait for collection
to finish before launching training.

To resume interrupted head training, set `RESUME=1` and reuse its `RUN_DIR`,
`TRACE_DIR` and GPU configuration, then launch `train` again. The launcher rejects
a still-active session, appends logs, and retains the previous environment and
exit status. The training program validates checkpoint compatibility.
Checkpoints are written at epoch boundaries; interruption within an epoch
replays that epoch from its last saved boundary. No mid-epoch resume is claimed.

Collection defaults: `TRAIN_PROMPTS=256`, `HELDOUT_PROMPTS=64`,
`MAX_PAIRS_PER_PROMPT=32`, `TOP_K=16`, `FEATURE_SIZE=64`,
`MAX_NEW_TOKENS=512`, `THRESHOLD=0.90`, `SEED=1234`.
Training defaults: `EPOCHS=3`, `BATCH_PER_GPU=32`, `WIDTH=64`, `LR=0.001`.
The global training batch depends on the number of selected GPUs.
`CONDA_ROOT`, `CONDA_ENV`, `HF_HOME`, `HF_HUB_CACHE` and
`HF_DATASETS_CACHE` can override the environment and cache paths.
Memory checks require 8 GiB free per GPU for collection and 2 GiB for head
training by default; `MIN_FREE_GIB` overrides the check, not actual memory use.

## Interpretation

First check held-out conditional KL, teacher changes, candidate coverage and
the blind/calibration controls. A falling head loss only establishes learning.
The frozen teacher can itself be wrong, and accurately reproducing its next
prediction is not the same as answering a task correctly.

No trained head is automatically injected into generation. An online experiment
would still need to demonstrate fewer actual model calls, include head and cache
costs, and measure task quality and end-to-end latency against native decoding.
The existing relation-code experiments remain separate historical results.

The nearest methodological reference is [ADJUST](https://arxiv.org/abs/2509.22738),
which trains a lightweight conditional sampler over a frozen diffusion model.
This prototype does not claim that adding a conditional head is a new idea;
the measured native next-call residual and its controls are a feasibility study.

After training, inspect `RUN_DIR/train/summary.json` or run
`python -m relation_update.report /path/to/run/train/summary.json`.
TensorBoard logs are in `RUN_DIR/train/tensorboard`; JSONL training metrics are
always written even when TensorBoard is unavailable.
