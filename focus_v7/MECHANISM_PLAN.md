# Implementation and admission audit after the128quality failure

The user asks to preserve the promising architecture and examine implementation
and parameters. V7's frozen128HumanEval screen scored17/128 versus FOCUS43/128.
Do not relabel that result or rerun LLaDA/v1. All current GPU tasks have ended;
the research lock, physical GPU1 and protected main hashes were freshly checked.

First audit the actual v7 trajectory, without changing its proposals, commits,
cache or stopping. Reuse the same six development prompts as the earlier small
generation diagnosis (two each HumanEval/MBPP/MATH). Log audit probability,
clean confidence, proposal ordering and every actual commit. Compare final raw
tokens, packet decisions, rendered text and NFE to the saved original v7 runs.
No gold/tests/solutions reach the generation or observer.

At predeclared packet indices0,1,4,8, run two paid private shadows:

1. A full current-canvas bidirectional forward in a separate cache bank. All
   current inputs and positions are legitimate; no later teacher tokens are
   used. It returns current selected-position predictions and fresh external KV.
2. The exact same v7 packet, including its candidate order and mask, against
   that fresh bank. Holding conditioning fixed isolates cached external KV.

Also repeat the old-bank packet at the first selected window per prompt and
require exact logits/actions/private-source KV. Always restore public cache
objects and adapter state on normal or exceptional exit. Confirm public cache
versions/content are unchanged and each physical key appears exactly once.
If controls fail, stop and diagnose implementation before interpreting results.

Fresh full computation is explicitly paid diagnostic information, NOT a free
refresh algorithm. All shadow calls/time are separate from online v7 NFE and
cannot be reported as a speed baseline. Reference top1 agreement is not task
accuracy; altered sequential conditioning can legitimately differ from a full
MASK prediction. First compare same conditioning old/fresh, then compare the
clean view (same canvas, no proposals) to full prediction.

Report accepted drafts versus first-mismatch corrections at fixed probability
bands .5/.9; clean/full prediction agreement; old/fresh audit decisions and
confidence drift; and the original six task scores. These are mechanism and
parameter diagnostics, not a parameter sweep or independent accuracy test.
Any candidate policy needs separate actual generation, equal scoring and total
latency. A high oracle agreement alone licenses no free cache update or success.
