# FOCUS-v5 Relay

FOCUS-v5 Relay replaces the abandoned PACT branch.  It starts from the two
expensive phases in Flash-dLLM:

1. a cache pass proposes tokens and refreshes selected state;
2. a private verification pass checks a proposal chain, then discards all of
   its K/V state.

Relay replaces the two serial calls with one three-view state transition after
bootstrap:

- a **clean view** refreshes tracked cache rows and all 32 active MASK rows,
  while producing the next proposal without reading the current draft;
- a **draft view** carries the current proposed token identities;
- a **verify view** checks each draft under a prefix-only dependency graph.

The query width is always 96 rows for block size 32.  Clean K/V is valid for
every outcome.  Commit first installs the clean cache, then overlays draft K/V
only for the verified prefix; rejected positions retain their clean MASK K/V.
Thus verification produces the next cache state rather than discarding it.

The official complementary two-view mask cannot be used for this promotion:
over multiple layers it admits paths from later proposals into earlier rows.
Relay therefore uses a one-way three-view mask with three invariants:

- clean cache and next-proposal rows never ingest speculative labels;
- draft cache row `i` depends only on proposals `0..i`;
- verifier row `i` depends only on proposals `0..i-1`.

These are all-layer information-flow guarantees for the proposed labels.  They
do **not** prove task accuracy or equivalence to bidirectional LLaDA.  The real
model must establish whether the cleaner dependency graph preserves enough
acceptance and whether saved cache work exceeds transaction overhead.

## Experimental gates

1. **Dependency gate:** exhaustive reachability at 32 layers must pass.
2. **Ceiling gate:** trace the pinned Flash runtime and compare its regular+64
   verification query rows with Relay's fixed 96-row transition.  Stop if the
   removable fraction is too small.
3. **State gate:** on fixed real states, compare staged K/V with the next regular
   recomputation and measure acceptance, action changes, and transaction cost.
4. **Generation gate:** only after the first three gates, run the same held-out
   prompts with official scoring and report end-to-end latency, accuracy, NFE,
   acceptance, rollback rate, and memory.

No gold answer, unit test, or solution text may enter model input.  Original
LLaDA/v1 results are reused rather than rerun.

## Measured decision (2026-10-03)

The row-count screen looked favorable (47,744 official regular+verify query
rows versus 13,248 Relay rows across 138 cycles, an optimistic 3.60x ratio),
but the real joint-forward gate rejected this version:

- 77 real Flash cycles on three reused development prompts;
- official regular+verify median: 57.80 ms;
- padded 96-live/128-kernel-row Relay median: 56.07 ms;
- aggregate measured state-level speedup: only 1.064x;
- official acceptance: 202 of 1,106 proposals;
- Relay acceptance: 77 of 1,106 proposals;
- equal accepted-prefix length in only 40/77 cycles.

The public cache was bitwise unchanged by the shadow pass and the official
Flash output hashes matched the earlier trace, so this is not a cache-mutation
artifact.  Two assumptions failed.  First, the existing Triton kernel rounds
96 live rows to a 128-row tile, making the joint pass approximately as costly
as the two original calls.  Second, the dependency-safe mask removes useful
bidirectional conditioning and sharply reduces acceptance, especially on MATH.

There is also a causal scheduling flaw in the original Relay story: a clean
branch evaluated before the current verification outcome cannot produce the
true next proposal conditioned on the newly accepted prefix.  Treating those
logits as the next cycle would introduce a one-step stale state.  Therefore
this Relay design stops at the state gate; it must not be promoted to a full
generator or reported as a 3.60x method.  The preserved diagnostic is
`results/relay_joint_20261003/diagnostic.json` (SHA256
`ffac8d2140f37cf90feb89175ee291b4b69e3d919c17896d651b0e09036c9380`).
