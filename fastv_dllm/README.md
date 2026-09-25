# FastV-style intra-block pruning for Fast-dLLM v2

This directory is an independent, training-free experiment. It does not edit
the upstream `v2/` implementation or the paused `step_distill/` experiment.
It adapts the physical token-dropping mechanism from the official
[FastV repository](https://github.com/pkunlp-icler/FastV) to diffusion-block
support positions; it does not copy FastV source code.

The pinned v2 decoder evaluates a complete 32-token diffusion block while its
commit rule consumes one 8-token sub-block. The experiment runs early layers
on all 32 current-block tokens, derives target-to-support relevance from the
same attention projections, keeps every hidden position required by v2's
shifted-logit semantics, and physically drops low-ranked support positions in
later layers. Prefix KV remains exact and cache-write forwards remain native.

The workflow is deliberately gated:

1. `audit` checks copied-decoder parity, all-support parity, and proves that a
   requested FlashAttention path actually issued FlashAttention calls.
2. `probe` leaves generation unchanged and measures retained attention mass,
   cross-layer ranking stability, and scoring overhead.
3. `smoke` and `evaluate` compare SDPA/native, Flash/native, SDPA/pruned,
   Flash/pruned, and the official sub-block cache under both backends.

Reported attribution separates:

- Flash engineering speedup: `SDPA native / Flash native`.
- Method speedup: `Flash native / Flash FastV` (same backend).
- Official cache speedup: `Flash native / Flash cache`.

The Flash backend is not assumed to be faster for these short query lengths;
it is measured. Mask-only pruning is not reported as acceleration: the FastV
path physically shortens hidden states after the selected layer.
