# Novelty boundary

FOCUS-v5 Relay is currently a research hypothesis, not a novelty claim.  The
closest primary sources found in the initial search are:

- [Flash-dLLM](https://arxiv.org/abs/2609.26796) combines an I/O-aware cache
  kernel with self draft-and-verify for dLLMs.  Relay keeps that useful
  cache/verify framing but changes the state transition: verification produces
  a rollback-safe next cache and the next proposal in the same pass.
- [Trajectory-Level Speculative Decoding for Diffusion Language Models](https://arxiv.org/abs/2608.27514)
  verifies explored denoising trajectories using bidirectional masks.  Relay
  does not search a trajectory tree; it pipelines one current proposal with one
  clean next-proposal/cache branch.
- [SpecLA](https://arxiv.org/abs/2607.16673) and
  [TreeWY](https://arxiv.org/abs/2608.20961) recover the accepted recurrent
  state produced during speculative verification for linear-attention hybrid
  models.  They establish that verification-carried state is a real nearby
  systems idea.  Relay's prospective difference is a bidirectional masked
  Transformer construction with clean/draft/verify views and token-indexed KV
  rollback, rather than recurrent-state algebra.
- [VeriCache](https://arxiv.org/abs/2605.17613) verifies lossy AR KV decoding
  against a full cache.  Its objective and execution path differ, but it is a
  required comparison for any broad cache-verification claim.

The defensible candidate contribution is therefore narrow: a constant-width,
one-way information-flow factorization that jointly (1) verifies current dLLM
drafts, (2) refreshes a non-speculative cache version, and (3) produces the next
draft in one Transformer pass.  If implementation reduces to ordinary
speculative decoding plus cache rollback, or if the clean one-step-lag branch
destroys acceptance, the novelty/performance hypothesis is rejected.
