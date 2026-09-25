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
masked position eligible for the current block's commit decision. Early-layer
attention from those target queries ranks all remaining support positions;
deep layers retain every target plus a configured ratio of support positions.
Physical pruning preserves each retained token's original RoPE coordinate.

The workflow is deliberately gated:

1. `audit` checks parity with the original LLaDA `generate()`, all-support
   parity, and proves that requested FlashAttention calls actually execute.
2. `probe` leaves generation unchanged and measures retained attention mass,
   cross-layer ranking stability, and scoring overhead.
3. `smoke` and `evaluate` compare torch-SDPA/native, flash-attn/native,
   torch-SDPA/pruned and flash-attn/pruned.

Reported attribution separates:

- Flash engineering speedup: `torch-SDPA native / flash-attn native`.
- Method speedup: `flash-attn native / flash-attn FastV` (same backend).

The Flash backend is not assumed to be faster for these short query lengths;
it is measured. Mask-only pruning is not reported as acceleration: the FastV
path physically shortens hidden states after the selected layer.
