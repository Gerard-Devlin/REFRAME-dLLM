# FOCUS-v4 execution work

FOCUS-v4 is an isolated runtime experiment. Native LLaDA and Fast-dLLM v1's
completed 24-cell table stays frozen; neither baseline is generated again.
The main-table entry, decoder, scoring files, model weights and README stay
unchanged. Research runs only on physical GPU 1 under the exclusive lock.

The first screen holds the physical pruning rule fixed: PrefixCache, layer 4,
future support ratio 0.3125, block 32 and native >=0.90/argmax decisions. It tests
device selection maps, private prefix RoPE reuse, full-vocabulary FP64 fused
statistics, and fixed-shape shallow/deep CUDA Graph segments. The last block
keeps the original exact-forward branch. Already accepted prefix KV is never
modified. No quantization, stale future cache, speculative state, early EOS,
new sampler or reduced generation budget is included.

CUDA Graph ownership and shape restrictions follow the [PyTorch CUDA Graph
documentation](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).
Stable inputs and dynamic position maps are copied on device. The two graph
segments leave variable active-query saliency and variable active-row LM head
outside capture, preserving the original BF16 GEMM and relevance shapes.

Cost accounting includes every graph capture, its setup/warm-up model work,
private cache copy and RoPE preparation in clean per-request wall time. NFE
counts committed decoding calls; extra setup forwards are identified separately.
Tracing/logged timings never become the clean latency baseline. A captured
segment alone being faster is not sufficient to enable it by default.

Quality gates are CPU cache/position/exception tests, real same-state BF16
logit/action comparisons, and complete output/commit/NFE regression. These are
finite tests, not a theorem of floating-point equivalence on all inputs. Fused
FP64 statistics regroup the summation and therefore need decision regression.
Once one common configuration is fixed, validate 256/512 on all four tasks with
shuffle seed 51713 and IDs at offset >=64; final-expression-v3 and official code
execution apply uniformly. Answers and tests are consumed only by the scorer.

The first component screen is four tasks, two already used development prompts
per task, length 256. Its pass rates are not a benchmark accuracy claim. Later
versions must retain failed records and snapshots and report additional cost
or numerical changes explicitly. Frozen baseline timings reused across runs
must be labelled historical; current FOCUS component controls give a paired
measurement of the engineering change under the current server conditions.

Two completed development screens selected one common default: on-device maps,
read-only borrowed formal prefix KV, private prefix RoPE preparation and fused
FP64 readout statistics. Per-request graph capture lost in both screens,
including delayed capture; it remains an opt-in ablation rather than a default.
The second screen's paired speedups on two prompts per task were 1.073x
HumanEval, 1.082x MBPP, 1.106x MATH and 1.098x GSM8K. Complete commit sequences,
tokens and NFE agreed with the current FOCUS control on all eight prompts.
These are small development results, not a claim of improved benchmark accuracy.

Two historical text/NFE mismatches also reproduced with the unchanged original
FOCUS entry on the current server, independently of v4. Their cause remains
unresolved. Historical table agreement is reported separately from same-process
FOCUS parity, so historical timing or numerical drift is not attributed to v4.
The fixed validation uses seed 51713, offset 64, 16 prompts per task at each of
256/512; only FOCUS controls and this chosen v4 config run. Native LLaDA/v1 stay
frozen. Scorer exceptions remain explicitly unresolved instead of counting as
incorrect answers. Every traced action and every clean token/NFE is checked.

## Research objective clarified by the user

The v4 method is allowed to change the token sequence, commit order, NFE,
pruning and decoding decisions. The target is genuinely higher task accuracy
and lower end-to-end decoding latency, not reproduction of old FOCUS outputs.
Exact trajectory checks apply only to the current execution-only control;
they are not an acceptance gate for subsequent quality-improving candidates.

Keep the ongoing immutable runtime validation as a component result. A quality
candidate uses a separate run identity after that job releases GPU 1; do not
edit its source, configuration, or interpretation halfway through the run.
Choose candidates on the existing development prompts, then fix one common
configuration and reserve new IDs for paired quality validation. A different
answer is evaluated by final-expression-v3 or official code execution, rather
than by equality with the previous answer. Native LLaDA/v1 results stay frozen.

Report accuracy gains and regressions, paired uncertainty, request latency,
NFE, truncation and setup/correction costs together. Retain better quality/speed
tradeoffs as experimental candidates, and prefer candidates improving both.
Do not claim higher accuracy from format-only score changes or from a few
development examples. The current 7–11% execution gains improve cost only;
an actual quality gain remains unproven.
