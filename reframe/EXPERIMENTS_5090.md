# Single-RTX-5090 comparison protocol

All commands below run inside the **same tmux session** on the lab server.
The campaign invokes unmodified v1 code. No new weights are needed.

## Fixed first-round settings

LLaDA-8B-Instruct, BF16, FlashAttention, batch=1, one GPU, generation length
256, block length 32, greedy decoding, seed `0,1234,1234,1234`. GSM8K uses
5-shot; HumanEval uses 0-shot. Parallel methods use confidence threshold 0.9.
REFRAME uses 16 pilots per side, reference refresh every two blocks and the
default residual threshold 0.25. These are initial settings, not tuned results.

| Method name | Purpose |
| --- | --- |
| native-serial | v1 full-forward baseline, fixed one-token quota per step |
| native-prefix-serial | v1 prefix caching with the same fixed quota |
| native-full | v1 parallel selection, no KV cache |
| native-prefix | v1 prefix cache + parallel selection |
| native-dual | v1 dual cache + parallel selection; primary speed reference |
| stale | Fewer reference refreshes, no transport |
| shift | Translation-only transport |
| pair | REFRAME complex key transform and affine values |

The serial rows deliberately change token selection and are context baselines.
The main cache comparison fixes threshold 0.9. NFE is an observed result, not
something artificially matched. `steps=gen_length` gives the current v1
DualCache enough iterations to finish each block; passing `steps=8` for a
256-token, 32-token-block run would allow just one iteration per block in this
checkout. The threshold decoder can still stop early.

The factor decoder is a separate sampling-policy sweep. Fast-dLLM v2 uses a
different setup and is not part of this first controlled LLaDA/v1 experiment.
Optional ablations `scale` and `materialize` can be selected with `--methods`.

## Setup, once

Start a session outside tmux (or attach to the existing one):

```bash
tmux new-session -A -s reframe5090
```

Then run inside that session:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastdllm311
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd /home/xuyouwen
if [ ! -d REFRAME-dLLM/.git ]; then
  git clone --depth 1 https://github.com/Gerard-Devlin/REFRAME-dLLM.git
fi
cd /home/xuyouwen/REFRAME-dLLM
git pull --ff-only

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/xuyouwen/hf_home_local
export HF_HUB_CACHE=/home/xuyouwen/hf_hub_local
export HF_DATASETS_CACHE=/home/xuyouwen/hf_home_local/datasets
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
unset TORCH_COMPILE_DISABLE

# HumanEval needs a small scoring module in addition to the dataset.
# Only this preparation step needs HF mirror access.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_EVALUATE_OFFLINE
python -c "import evaluate; evaluate.load('code_eval'); print('code_eval cached')"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 'pytest>=8,<10'
python -c "import torch, flash_attn, transformers, accelerate, lm_eval; from importlib.metadata import version; print('torch',torch.__version__,'cuda',torch.version.cuda,'GPU',torch.cuda.get_device_name(0)); print('flash_attn',flash_attn.__version__,'transformers',transformers.__version__,'accelerate',accelerate.__version__,'lm_eval',version('lm_eval'))"
python -X utf8 -m pytest reframe/tests -q
nvidia-smi

export RUN_ROOT="reframe/results/5090_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
```

Expected application versions: Transformers 4.49.0, lm-eval 0.4.8,
Accelerate 0.34.2. Preserve the server's working Torch/CUDA/FlashAttention.
The FlashAttention test must actually pass on this server, not be skipped.
If GPU 0 is occupied, select an idle physical GPU and keep it fixed for all
timing runs. Do not overlap jobs on that GPU.

The GSM8K task is copied from installed lm-eval with only its name and dataset
ID changed (`gsm8k_local`, `openai/gsm8k`), to use the existing offline cache.
Both native and REFRAME use this identical task file. It is saved with results.

## Smoke and mechanism checks

```bash
python -u reframe/benchmark.py smoke --task gsm8k --output "$RUN_ROOT/smoke_gsm8k"
python -u reframe/benchmark.py smoke --task humaneval --output "$RUN_ROOT/smoke_humaneval"

python -u reframe/benchmark.py oracle --task gsm8k \
  --source "$RUN_ROOT/smoke_gsm8k/native-dual" --limit 2 \
  --output "$RUN_ROOT/oracle_gsm8k"

python -u reframe/benchmark.py audit --task gsm8k \
  --source "$RUN_ROOT/smoke_gsm8k/native-dual" --limit 2 \
  --output "$RUN_ROOT/audit_gsm8k"
```

Each smoke evaluates two examples for all eight methods, at the real 256-slot
length. It is not an accuracy estimate. Oracle/audit runs are diagnostics;
their times include teacher/probe computations and must not enter speed tables.
Inspect errors/fallbacks before committing to all full runs. Do not tune using
final test scores; use separate development examples for any parameter search.

## Accuracy and common-boundary timing

Once the smoke and diagnostics are acceptable, execute sequentially:

```bash
python -u reframe/benchmark.py accuracy --task gsm8k \
  --output "$RUN_ROOT/acc_gsm8k" && \
python -u reframe/benchmark.py accuracy --task humaneval \
  --output "$RUN_ROOT/acc_humaneval"

python -u reframe/benchmark.py timing --task gsm8k \
  --source "$RUN_ROOT/acc_gsm8k/native-dual" --limit 16 \
  --output "$RUN_ROOT/time_gsm8k" && \
python -u reframe/benchmark.py timing --task humaneval \
  --source "$RUN_ROOT/acc_humaneval/native-dual" --limit 16 \
  --output "$RUN_ROOT/time_humaneval"
```

Accuracy defaults to the full test split: GSM8K 1319, HumanEval 164. HumanEval
automatically applies the original `postprocess_code.py` to **every method**;
report its cleaned pass@1 consistently. GSM8K saves both strict-match and
flexible-extract metrics. Full scores are saved in each campaign's `scores.json`.

Timing replays the exact logged raw prompts (including few-shot demonstrations)
from native DualCache, then applies the same model chat template to every
method. It discards one warmup per method/prompt and measures three repeats;
every timed generation starts with an empty cache. Method order alternates.
Warmup, weights loading, tokenization and task scoring are outside the timing
boundary; cache initialization, pilots, fitting, writes and fallback are inside.
The first 16 questions are a preliminary performance sample, not a full-suite
speed or tail-latency result. Increase the sample count/repeats before final
claims, and repeat at length 512 only after the 256-token campaign is sound.

Prompt replay accepts both the in-memory request list and lm-eval 0.4.8's saved
`arguments.gen_args_0.arg_0` / `arg_1` mapping. If an older checkout fails with
`KeyError: 0` during `export_prompts`, update the checkout and reuse the existing
smoke/accuracy samples with a **new timing/audit output directory**. The failure
occurs before launching generation, so the completed accuracy run is unaffected.

Never compare native evaluator printed TPS directly with REFRAME evaluator
printed TPS. Use the timing campaign's common measurement boundary. Serial
versus parallel speedups include the sampler change; the primary cache speedup
is native-dual mean latency divided by pair mean latency on the same prompts.

Each campaign records package versions, commit, GPU status, commands and logs.
On failure it stops without overwriting any previous results. Use a fresh output
directory or `--methods` for remaining methods on retry; never delete successful
results just to restart. `run.py` rejects leftover MASK tokens in timing runs.

Detach tmux with Ctrl-b then d. Reattach with `tmux attach -t reframe5090`.
For another terminal, use `tail -f` on the per-method log under `RUN_ROOT`.
Send `scores.json`, timing `summary.log`, and diagnostics/manifest files for
analysis. Keep all raw samples to permit paired comparisons later.
