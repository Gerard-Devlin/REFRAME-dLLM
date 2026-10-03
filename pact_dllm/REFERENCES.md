# Implementation sources and attribution

Reviewed on2026-10-03; checked code and paper claims are distinguished.

- **Flash-dLLM**: `VILA-Lab/Flash-dLLM`, pinned commit
  `7437a550fd3d1a0752edcbf58bd015ad69083068`. Inspected
  `llada/generate.py`, `llada/flash_cache_triton.py`,
  `llada/model/modeling_llada.py`. Reuse its fused QKV/RoPE operator through the
  existing private-buffer adapter: accumulation/rotation precede BF16 storage.
  No third-party file is modified. Its cache table construction, tracked queries
  and probability/readout conventions inform integration; PACT does **not**
  reuse its speculative information-flow mask or prefix acceptance rule.
  [Official code](https://github.com/VILA-Lab/Flash-dLLM),
  [paper](https://arxiv.org/html/2609.26796v1).
- Existing repository `focus_dllm/tuning/firebreak/attention.py`, `kernels.py`
  and `probe.native_projection` supply independently audited conventional
  version-selection attention and private projection infrastructure. PACT's
  arbitrary DAG, maximum closure, identity ledger, clean transactions and online
  decoder live in this separate folder. This infrastructure is not a novelty
  claim or a revival of FIREBREAK's cross-view veto algorithm.
- **PUNT**: checked conditional-influence testing and implementation discussion;
  [paper](https://arxiv.org/html/2510.21961v1),
  [author publication](https://www.microsoft.com/en-us/research/publication/parallel-sampling-from-masked-diffusion-models-via-conditional-independence-testing/).
  A usable official repository was not located in the checked paper/publication
  links and exact-title search. No PUNT code was imported or claimed reproduced.
- **DiCo**: checked adaptive parallel-decoding paper,
  [paper](https://arxiv.org/html/2602.23792v1). The checked paper/search did not
  expose a verified official implementation. A different visual-diffusion
  ConvNet project also called DiCo must not be mistaken for this work.
- **COVER**: checked dual attention views/cache override and diagonal correction,
  [paper](https://arxiv.org/html/2602.06161v1). Linked WINO/Saber repositories are
  baselines, not verified COVER source. No COVER implementation was imported.
  A direct diagonal correction alone is not used as an all-layer leakage proof.

The last three availability statements describe this bounded inspection, not
proof that no public code exists. Their reported hardware/model speedups are
not used as measurements on our GPU. PACT's mechanism must be assessed against
these near neighbors before making any research originality claim.
