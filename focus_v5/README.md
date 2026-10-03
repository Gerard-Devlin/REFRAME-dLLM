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
