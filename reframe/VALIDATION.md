# Validation status — updated 2026-09-13

The real LLaDA-8B server validation is now complete for a bounded feasibility
check. **The default pair variant did not accelerate inference:** about 3.7
times native v1 DualCache latency on both two repeated smoke prompts and four
separate development prompts. See [FEASIBILITY_5090.md](FEASIBILITY_5090.md)
for the protocol, oracle/audit findings, timing and limitations. These are not
full benchmark accuracy results. The original `v1/` and `v2/` source trees are
unchanged relative to upstream commit `a9b81e4`.

## Server validation — 2026-09-13

- RTX 5090 32 GB, physical GPU 3, existing `fastdllm311` environment.
- PyTorch 2.7.1+cu128, FlashAttention 2.8.3.post1, Transformers 4.49.0,
  Accelerate 0.34.2, lm-eval 0.4.8.
- `python -X utf8 -m pytest reframe/tests -q`: **43 passed in 24.98 seconds**,
  including FlashAttention and saved-checkpoint loading tests.
- Fixed serialized lm-eval generation argument replay and the real-checkpoint
  config-class mismatch. Real 8B oracle, audit and timing paths all completed.
- Oracle: 2160 correlated observations from two prompts; audit: 42 decisions;
  timing: two prompts with three repeats per method, plus four development
  prompts with one measured repeat per method. Full refresh cost was measured
  separately with CUDA events and unchanged native outputs.
- GPU jobs ran under tmux, exited successfully, and released GPU 3.

The remainder of this file records the earlier local prototype checks.

## Historical local environment and checks — 2026-09-12

- Windows, Python 3.12.13, PyTorch 2.13.0+cu130, Transformers 4.49.0.
- lm-eval 0.4.8, Accelerate 0.34.2; the evaluation entry point loads successfully.
- Local GPU: NVIDIA RTX 4060 Laptop, 8 GB. This is not the lab's RTX 5090.
- `python -X utf8 -m pytest reframe/tests -q`: **28 passed, 1 skipped**.
  The skipped test requires FlashAttention, which is not installed locally.
  CUDA BF16 transport versus materialization passed with the torch backend.
- Both server shell scripts passed `bash -n` syntax checking. At that time,
  server jobs had not yet been executed.

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
full task-accuracy campaign remains unrun; the feasibility check above now
provides a reason to pause the default variant before that campaign.

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

## Requirements before promoting a future variant

1. Establish improved held-out approximation on more development prompts,
   especially for prefix states. The current pair fit fails this first gate.
2. Demonstrate actual common-boundary latency gains with all fit/write/fallback
   work included. The current adapter fails this gate even in stale mode.
3. Compare closed-loop outputs and task accuracy at matched prompts, few-shot
   examples, length, threshold, dtype and hardware.
4. Expand to full benchmark evaluations only after those checks succeed.

Raw local and server files stay in ignored `reframe/results/`; measured server
summaries are linked from the feasibility report. No model weights, datasets
or fabricated task scores are committed.
