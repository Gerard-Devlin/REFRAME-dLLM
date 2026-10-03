# WAVE-Verify feasibility audit

The user's proposal changes evaluation order while retaining Flash's exact
reference information flow. Do not replace it with FIREBREAK's different mask.
No GPU decoder or DP scheduler is deployed here.

First inspect the pinned T/D/M attention matrix, per-layer residual edges and
cache side effects. Reverse dependencies of even the FIRST MASK output include
ALL later MASK rows immediately, and all except the last draft within two layers.
For block32/search16/depth32, the prefix-one output needs63/64 input rows.
Layers1..30 still need63 output rows, layer31 needs48, and layer32 needs1.
The matched full MASK target needs63 rows in layers1..31, then16 in layer32.
The optimistic output-row work proxy falls only1969->1939 (1.52%). This is NOT
actual FLOPs or time: final-layer K/V still need all necessary source rows, and
head/launch/query metadata costs are additional. Ordinary Flash verification is
read-only with respect to public KV, but that does not remove these dependencies.

This rejects a deep prefix-only lazy implementation for this unchanged graph;
there can still be small terminal output/MLP savings. No batch-cost profiler or
full executor is justified until a different graph is explicitly proposed and
its changed semantics acknowledged. Bellman DP is tested only on given toy
cost/survival tables; toy gains are not measured dLLM speedups.

Sources: pinned `7437a550fd3d1a0752edcbf58bd015ad69083068`
`flash_dllm/llada/generate.py` causal_mask m11/m12/m21/m22 and
`flash_dllm/llada/flash_cache_triton.py` private verification projections.
Related DP reference: [Sequoia](https://arxiv.org/abs/2402.12374).
The specific matrix audit, not DP novelty, determines feasibility here.
