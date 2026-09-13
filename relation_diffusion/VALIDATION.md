# Validation status — 2026-09-13

This is a from-scratch byte-model experiment runner, not a validated speedup,
an LLaDA-8B training implementation, or an already trained language model.

Checks performed locally:

- Existing finite-state mechanism tests plus neural pipeline tests.
- Byte codec round-trip, protected prefix/boundary and NumPy/PyTorch agreement,
  including local CUDA execution.
- Exhaustive normalization of small fixed-path model distributions and checks
  that current/future target groups remain masked during likelihood evaluation.
- Real forward/backward, global loss normalization, connected zero-mask loss,
  atomic checkpoint save and exact single-process resume.
- Real checkpoint evaluation, inverse-transform error propagation and untrained
  cost preflight on separate synthetic text split files.
- Local CUDA BF16 training for four updates and 1/2/4/8/16-step evaluation on
  a 135,361-parameter test model completed successfully. These outputs are
  functional evidence only, not language-model quality or speed claims.

`RUN_DDP_TESTS=1 python -m pytest relation_diffusion/tests -q` passed all
18 tests in 25.44 seconds, including preparation with the train-only
independent categorical diagnostic and held-out scoring.

The opt-in `RUN_DDP_TESTS=1` test compares two CPU/Gloo workers against a
single worker using the same global batch, samples, noise and four optimizer
updates. Windows uses two env:// workers because this local PyTorch's
`torchrun` static rendezvous requests libuv even though its build lacks it.
The test is set up to use the ordinary torchrun entry point on Linux; that
branch has not been run locally. CPU/Gloo is not a substitute for measuring
six-GPU NCCL.

Default model: 10,894,721 parameters, 6 bidirectional blocks, width 384,
6 heads, sequence length 128, 257 output symbols and a separate MASK input.

No server training was launched in this turn. WikiText downloads, full data
preparation, RTX 5090 throughput, multi-GPU NCCL and actual few-step language
quality remain to be measured by the user. Entry points and explicit GPU
selection are documented in [TRAINING_5090.md](TRAINING_5090.md).
