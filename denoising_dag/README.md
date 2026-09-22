# Denoising state confluence

Frozen **Fast-dLLM-v2 1.5B / 7B**, official pinned Hugging Face modeling code.
No training, no AR surrogate, no approximate feature reuse.

This implements the proposal's bounded deterministic search: branch on the
highest-confidence remaining positions, propose their greedy token, and score
each path by summed selected log probabilities plus terminal negative mean
entropy. Every logical path and its score remain separate. Prediction statistics
are reused only within an immutable model/prefix-cache/position/mask context.
The search is an experimental policy, **not a reproduction of LoPA or MCTS**.
Any future stochastic policy needs full distributions and independent RNG draws;
the current greedy summaries are sufficient only for the current policy.

`probe` observes real states from unmodified native v2 generation, then compares
the same bounded search with and without memoization. Both batch frontiers with
the same maximum batch size. Both skip model calls at completely filled leaves
whose terminal entropy is already zero. Depth 1 is the expected low/zero-hit control.
The audit records all logical states, predictions used by the policy, every
expansion decision and the selected path. A separate full-vocabulary check
compares repeated singleton and batched native inputs. Shape-related BF16 drift
is reported; changed decisions invalidate the quality-preserving speedup field.

Timing uses alternating execution order and median repeated wall time, including
key construction, GPU prediction, policy, and result dispatch. CUDA model time,
logical rows, physically evaluated rows, actual forward calls and peak memory
are separate. A count ratio is never presented as measured latency speedup.

`generate` additionally compares unmodified native threshold generation with
full bounded-search generation, before/after memoization. It commits the selected
path and clears the table when the completed-prefix cache advances. Native and
search have different policies, so inspect GSM8K correctness, EOS/truncation and
actual output length as well as time. Native's stock block-count length cap can
stop below the requested token cap; this is disclosed in its records.

Multiple GPUs run independent prompt shards, one complete BF16 model per GPU.
Both sizes run sequentially. Reported latency is per request, not tensor-parallel
or aggregate-throughput speedup. A small development set tests the mechanism;
it is not sufficient for a task-quality or novelty claim.

Entry points: `python -m denoising_dag.benchmark --help`,
`python -m denoising_dag.campaign --help`, and `scripts/launch_tmux.sh`.
Inference is offline. `download` explicitly fills the existing Hugging Face
cache with the pinned checkpoints; it is not invoked by inference.

Upstream: [Fast-dLLM v2](https://github.com/NVlabs/Fast-dLLM/tree/main/v2),
[1.5B](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B),
[7B](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B).
