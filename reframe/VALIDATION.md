# Local validation — 2026-09-12

This is a functional prototype. **No LLaDA-8B accuracy or speed result has been
measured yet.** The original `v1/` and `v2/` source trees are unchanged relative
to upstream commit `a9b81e4`.

## Environment and checks

- Windows, Python 3.12.13, PyTorch 2.13.0+cu130, Transformers 4.49.0.
- lm-eval 0.4.8, Accelerate 0.34.2; the evaluation entry point loads successfully.
- Local GPU: NVIDIA RTX 4060 Laptop, 8 GB. This is not the lab's RTX 5090.
- `python -X utf8 -m pytest reframe/tests -q`: **28 passed, 1 skipped**.
  The skipped test requires FlashAttention, which is not installed locally.
  CUDA BF16 transport versus materialization passed with the torch backend.
- Both server shell scripts pass `bash -n` syntax checking. Server jobs have
  not been executed; the scripts target the existing `fastdllm311` environment.

Tests cover group softmax normalization with key translation; split-half RoPE
transforms and inverse writes; grouped-query heads; FP64 algebra and BF16
tolerances; non-contiguous pilot positions; full-forward agreement with the
original tiny LLaDA architecture; completed-token writes; transactional
fallback; original DualCache agreement in the stale/per-block-refresh control;
oracle probes that do not change the baseline; read-only logit audits; and
the lm-eval adapter's stop handling and repeated-call logging.
Campaign tests additionally cover method arguments, replaying logged few-shot
prompts, and preserving GSM8K scoring when switching to the cached dataset ID.
All five native comparison modes also complete random-tiny generation. The
full real-model campaign itself remains to be run on the lab server.

## End-to-end smoke checks

Random tiny LLaDA uses two layers, hidden size 32, four heads, vocabulary 128.
These weights carry no task knowledge. CPU and CUDA BF16 generation completed
for native DualCache, stale cache, shift, pair transport and materialization.

The CUDA check used the torch attention backend and disabled compilation,
because this Windows environment has no Triton:

```bash
TORCH_COMPILE_DISABLE=1 python -X utf8 reframe/run.py --tiny \
  --device cuda --dtype bfloat16 --backend torch \
  --gen-length 16 --block-length 4 --threshold 1 \
  --warmup 1 --repeats 1 --output reframe/results/local_cuda_smoke.jsonl
```

| Method | Attempted NFE | Full forwards | Fallbacks |
| --- | ---: | ---: | ---: |
| Native DualCache | 16 | 4 | 0 |
| Stale, refresh every 2 blocks | 16 | 2 | 0 |
| Shift, refresh every 2 blocks | 16 | 2 | 0 |
| Pair, refresh every 2 blocks | 16 | 2 | 0 |
| Materialized pair control | 16 | 2 | 0 |

**All new paths were slower than native DualCache on this tiny CUDA test.**
Python dispatch, many small kernels, checks/synchronizations and cache gathers
dominate at this scale. Fewer full forwards are not proof of lower latency.
One tiny-model repeat is not evidence about LLaDA-8B performance.

Native-trajectory oracle diagnostics also completed. A separate CUDA BF16 pair
run with 8 generation slots and `--audit-every 2` produced four finite logit
comparisons, without feeding the full-model outputs into the trajectory. Those
probe forwards are logged separately and their time remains included; this
audited run must not be treated as a timing comparison.

## Outstanding validation

1. Run the FlashAttention test on the lab server before using its backend.
2. Run LLaDA-8B oracle diagnostics to test held-out prefix/suffix fit errors
   over reference ages. Use at least five blocks to observe age four.
3. Compare closed-loop outputs and task accuracy at matched prompts, few-shot
   examples, length, threshold, dtype and hardware.
4. Compare common-boundary wall time, fallback rate and peak memory, including
   first full forward, pilot fitting and completed-block writes. Only optimize
   kernels further if the approximation maintains useful quality.

Raw local smoke files stay in ignored `reframe/results/`; they are not benchmark
results. No model weights, datasets or fabricated task scores are committed.
