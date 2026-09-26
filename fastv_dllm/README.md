# FastV-style support-token pruning for original LLaDA

This directory is an independent, training-free experiment on the pinned
`GSAI-ML/LLaDA-8B-Instruct` base checkpoint. It does not use Fast-dLLM v2
weights, edit upstream `v1/` or `v2/`, or alter the paused `step_distill/`
experiment.
It adapts the physical token-dropping mechanism from the official
[FastV repository](https://github.com/pkunlp-icler/FastV) to diffusion-block
support positions; it does not copy FastV source code.

The original LLaDA parallel sampler evaluates the prompt, generated tokens and
remaining mask canvas on every denoising call. The experiment keeps every
prompt token, every revealed language token and every masked position eligible
for the current block's commit decision. Early-layer attention from those
target queries ranks only the untouched future MASK canvas; deep layers retain
all language/target positions plus a configured ratio of those future MASKs.
Physical pruning preserves each retained token's original RoPE coordinate.

The workflow is deliberately gated:

1. `audit` checks parity with the original LLaDA `generate()`, all-support
   parity, and proves that requested FlashAttention calls actually execute.
2. `probe` leaves generation unchanged and measures retained attention mass,
   cross-layer ranking stability, and scoring overhead.
3. `smoke` and `evaluate` compare torch-SDPA/native, flash-attn/native,
   torch-SDPA/pruned and flash-attn/pruned.

`--cache-mode` provides three execution controls under the same evaluator:

- `none`: confidence-aware parallel decoding without a KV cache;
- `prefix`: the official Fast-dLLM PrefixCache, optionally followed by FastV
  pruning in ordinary suffix-refinement calls;
- `dual`: the official Fast-dLLM DualCache execution path.

`--decoding-mode single` supplies the paper's one-token-per-step LLaDA and
cache-only controls. The default `threshold` mode uses the confidence-aware
parallel decoder at the configured threshold. Together these switches cover
the paper's `LLaDA / +Cache / +Parallel / +Cache+Parallel` component matrix.

DualCache already evaluates only the active block after its warm-up. Under the
safe pruning policy there is therefore no untouched future MASK canvas left to
remove; `dual + FastV` intentionally takes the exact DualCache forward path.
This executable no-op is kept as an overlap control rather than advertised as
a composed acceleration method.

Reported attribution separates:

- Flash engineering speedup: `torch-SDPA native / flash-attn native`.
- Method speedup: `flash-attn native / flash-attn FastV` (same backend).

Cache composition is attributed against the corresponding cache baseline, not
against uncached LLaDA. In particular, the reported PrefixCache composition is
`Flash PrefixCache / Flash PrefixCache+FastV`; the much larger uncached-to-cache
gain belongs to Fast-dLLM.

The Flash backend is not assumed to be faster for these short query lengths;
it is measured. Mask-only pruning is not reported as acceleration: the FastV
path physically shortens hidden states after the selected layer.
