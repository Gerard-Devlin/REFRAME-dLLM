# REFRAME-dLLM experimental prototype

This directory implements **reference KV + query-side transport** for LLaDA.
It is a research prototype. The first real LLaDA-8B check on an RTX 5090 found
the default pair variant about **3.7 times slower** than native v1 DualCache
on two repeated smoke prompts and four development prompts. See
[the feasibility report](FEASIBILITY_5090.md) before scheduling a full campaign.
No benchmark-wide speedup or accuracy preservation has been established.
All new code lives here, alongside `v1/` and `v2/`. Neither upstream directory
is modified. The runner imports the original model and native decoding code
from `../v1/llada/` at runtime.

## Implemented

- Per-head complex scale/rotation + translation for **post-RoPE** keys. LLaDA
  pairs split halves of the head dimension, not adjacent channels.
- Per-channel affine value transport, FP32 fitting, ridge toward identity.
- Query/output transport, including the key-bias correction to group
  logsumexp. Prefix, masked suffix and freshly computed rows are disjoint.
- Current block + a small number of prefix/suffix pilots run through the
  existing model. True absolute positions support non-contiguous pilot rows.
- Reference cache reuse across multiple blocks. On the next block's first
  partial forward, the completed block is recomputed with its **final token
  identities**, then inverse-written into the prefix frame. Writes are committed
  only after all layers succeed. The final block needs no further cache write.
- Held-out validation pilots (one in four), invertibility/finite checks and
  full-forward fallback before token commitment. Periodic full refresh sets
  a hard maximum reference age. These checks are heuristics, not losslessness
  certificates; small pilot error does not bound unseen-token error.
- Controls: native PrefixCache/DualCache, stale cache with fewer refreshes,
  shift-only, scale-only, pair transport, and explicit KV materialization.
- Oracle probes on **native Fast-dLLM DualCache block-boundary states**.
  Fits use a pilot subset; the other eligible tokens are held out. A token is
  eligible only if its identity matches the old snapshot. All other tokens use
  current exact KV, so this experiment isolates one cache group's fit error.
- Optional full-state logit/top-1/commit audits, which never feed back into
  online generation. Oracle/audited runs cannot be used as speed results.
- A separate lm-eval entry point for task scoring with v1 prompt/scoring rules.

## Layout

```text
reframe/
  reframe_dllm/transport.py   # algebra, fitting, group normalization
  reframe_dllm/model.py       # reference cache + sparse token-row execution
  reframe_dllm/generate.py    # cross-block generation, accounting, audits
  reframe_dllm/oracle.py      # native trajectory holdout diagnostics
  run.py                     # tiny smoke and real-model comparison
  eval_reframe.py             # lm-eval model: reframe_llada
  summarize.py               # JSONL summaries
  scripts/                   # tmux jobs for the lab server
  tests/                     # algebra + real tiny LLaDA architecture tests
  VALIDATION.md              # measured local validation, limitations
```

## Local validation (no weights/downloads needed after dependencies)

Reuse a working PyTorch environment. `requirements.txt` intentionally does not
reinstall Torch or FlashAttention. Full lm-eval additionally needs
`v1/requirements.txt`. Run from the repository root:

```bash
python -m pip install -r reframe/requirements.txt
python -X utf8 -m pytest reframe/tests -q
python -X utf8 reframe/run.py --tiny --gen-length 16 --block-length 4 --threshold 1 --output reframe/results/tiny.jsonl
python reframe/summarize.py reframe/results/tiny.jsonl
```

On Windows use `-X utf8` for PyTorch's template source files. If the local CUDA
environment lacks Triton, set `TORCH_COMPILE_DISABLE=1` for tiny CUDA checks;
record this setting when comparing performance. The lab server should use its
existing, consistent compile settings for every method. `--tiny` uses random
weights and is **only a functional test**.

All runner output paths must be new, preventing accidental result mixing.
Warmup runs are discarded, but every timed request starts with an empty cache:
initial full computation, fit, pilots, completed-block writes and fallbacks
are included. CPU `torch` backend uses chunked FP32 attention; it is a reference
implementation, not a GPU performance claim.

## Lab server: tmux entry points

For the complete **eight-method RTX 5090 campaign**, including native v1
baselines, shared few-shot prompts, HumanEval postprocessing and common timing,
follow [EXPERIMENTS_5090.md](EXPERIMENTS_5090.md). The simple launchers below are
individual smoke jobs, not the complete comparison matrix.

Existing paths default to `/opt/miniconda3`, environment `fastdllm311`, model
cache `/home/xuyouwen/hf_hub_local`, dataset cache
`/home/xuyouwen/hf_home_local/datasets`. These scripts clear the old proxy and
read existing HF downloads offline. They do not download weights or install
packages. From the checkout root:

```bash
# Three-repeat single-prompt comparison, one GPU. Not a task accuracy test.
CUDA_VISIBLE_DEVICES=0 bash reframe/scripts/run_tmux.sh compare

# Held-out diagnostics; 8 blocks allow reference ages 1, 2 and 4.
CUDA_VISIBLE_DEVICES=0 GEN_LENGTH=256 bash reframe/scripts/run_tmux.sh oracle

# Two GSM8K examples, 5-shot, to validate the evaluation integration.
CUDA_VISIBLE_DEVICES=0 bash reframe/scripts/run_tmux.sh gsm8k
```

These launchers require tmux with `new-session -e` support. Each command
prints its `tmux attach -t ...` command. Logs and results live
under `reframe/logs/` and `reframe/results/`, both ignored by Git. Successful
jobs exit; if the session has already ended, read its log. Leave a live session
using Ctrl-b then d.

After a future variant passes both feasibility and quality checks, the same
GSM8K entry can run all 1319 test questions. The current pair variant has not
passed that gate:

```bash
CUDA_VISIBLE_DEVICES=0 GEN_LENGTH=256 LIMIT=1319 \
  REFRAME_KIND=pair PILOTS=16 REFRESH_BLOCKS=2 \
  bash reframe/scripts/run_tmux.sh gsm8k
```

Use distinct single-GPU jobs for the other variants. A six-GPU aggregate
throughput is not a single-GPU speedup. Final comparisons must also rerun the
original v1 method under the same benchmark prompts, model, dtype, backend,
length, hardware, stop extraction and timing scope.

## Real-model experiments

`run.py` accepts a shared `--prompts file.jsonl`, with one object per line:

```json
{"id":"example_0","prompt":"Your full prompt, including fixed demonstrations","until":["<|eot_id|>"]}
```

By default it adds the Instruct chat template. Use `--no-chat-template` if the
prompt is already formatted. It does **not** compute benchmark accuracy; use
`eval_reframe.py` and the identical native v1 evaluator for that.

```bash
python reframe/run.py --model-path GSAI-ML/LLaDA-8B-Instruct \
  --device cuda --dtype bfloat16 --backend flash --prompts prompts.jsonl \
  --gen-length 256 --block-length 32 --pilots 16 --refresh-blocks 2 \
  --warmup 1 --repeats 3 --output reframe/results/compare.jsonl

# Read-only diagnostics; deliberately separate from the timing experiment.
python reframe/run.py --model-path GSAI-ML/LLaDA-8B-Instruct \
  --device cuda --dtype bfloat16 --backend flash --prompts prompts.jsonl \
  --gen-length 256 --block-length 32 --oracle --output reframe/results/oracle.jsonl

python reframe/run.py --model-path GSAI-ML/LLaDA-8B-Instruct \
  --device cuda --dtype bfloat16 --backend flash --methods pair \
  --gen-length 128 --block-length 32 --audit-every 4 --output reframe/results/audit.jsonl
```

The principal test is **pair versus stale/shift at the same refresh interval**,
then accuracy versus actual latency relative to native Fast-dLLM. Otherwise,
merely refreshing less often can be mistaken for a transport contribution.
Try pilots 8/16/32 per side and refresh intervals 1/2/4 blocks on development
examples; freeze settings before final test evaluation. Do not pick thresholds
using the final test score.

## Metrics and current limitations

The runner reports generation wall time, actual attempted NFE, full forward
count, partial attempts/layer calls, fallback causes, pilot row count, commit
row count, fits, maximum observed held-out pilot error, allocated peak memory,
generated slots and useful output tokens. NFE includes an aborted partial
attempt and the succeeding full fallback. Processed-row counts include failed
attempts as scheduled work; `partial_layer_calls` records executed layers.

`slots_per_second` uses fixed generation slots. `useful_tokens_per_second`
stops at EOS/request stops. They are intentionally separate. The native v1
lm-eval timer includes CPU formatting/printing as well, so do not directly mix
its printed TPS with the generation-only metric in the new evaluator. Use the
shared standalone comparison for a common timing boundary.

The first version supports unpadded batch=1, RoPE LLaDA llama blocks, greedy
threshold/factor selection, and one request at a time per model. It does not
support Dream, model training, padded batches, shared-model concurrent threads,
custom masks, tensor parallelism, or CUDA graphs. The torch backend and safety
checks launch many kernels and synchronize; group `index_select` also copies
reference rows. This can be slower than native caching. No full corrected KV
is materialized in the transport path, but gather copies and attention still
read/cache-size data. Fusion/less-copy group handling is future optimization,
conditional on actual held-out and closed-loop results.

FA2 uses its public `flash_attn_func(..., return_attn_probs=True)` to obtain LSE;
with dropout=0, version 2.8.3 does not allocate the probability map. The CUDA
FlashAttention test must pass on the target server before performance claims.
BF16 transport and BF16 explicit materialization have different rounding
locations, so expect small numerical differences, not bitwise equality.

Upstream: [Fast-dLLM v1](https://github.com/NVlabs/Fast-dLLM/tree/main/v1).
Weights remain the unchanged LLaDA weights. Neither weights, datasets, local
IDE configuration, nor generated experiment logs are included in this repo's
new contribution.
