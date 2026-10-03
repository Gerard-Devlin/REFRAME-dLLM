"""Query-specific future-context budgets, an approximate v4 research control.

Use only the current layer-4 Q/K already computed by the shallow segment.
Bound discarded *mean-over-head attention mass for each active query* at that
observed layer. This is not a bound on later layers, logits or task accuracy.
No stale future KV, teacher oracle, new stopping rule or answer information.
"""
import math
import time

import torch

from ..llada_common import MASK_ID
from ..llada_decode import Result
from .focus_v4_runtime import BlockEngine, DEFAULT_OPTIONS, commit


def budget_order(future, budget):
    """Shortest prefix of a shared saliency ordering satisfying every query.

    ``future`` is the nonnegative mean-over-head softmax attention probability,
    with shape [active queries, future positions]. Its row sum can be below one
    because the accepted prefix/current block are protected separately. The
    budget is absolute mass, not a fraction of the future-only normalization.
    Minimality holds along this ordering, not among all possible position sets.
    """
    if future.ndim != 2 or not future.shape[0] or not future.shape[1]:
        raise ValueError('Nonempty query and future axes required')
    if not math.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError('Finite discarded attention mass budget in [0,1] required')
    order = future.mean(0).argsort(descending=True, stable=True)
    ordered = future.index_select(1, order)
    # Direct sums of remaining entries avoid cancellation at the all-kept end.
    remaining = torch.cat((ordered.flip(-1).cumsum(-1).flip(-1),
                           torch.zeros_like(ordered[:, :1])), -1)
    required = (remaining > budget).sum(-1).amax()
    return order, required, remaining


def select_budget(q, k, target, suffix, past, budget):
    if target.ndim != 1 or not target.numel() or suffix <= 32:
        raise ValueError('Nonempty active block and a future suffix required')
    scores = q.index_select(-2, target) @ k.transpose(-2, -1)
    probabilities = (scores / math.sqrt(q.shape[-1])).float().softmax(-1)
    future = probabilities.mean((0, 1))[:, past+32:]
    if future.shape[1] != suffix-32:
        raise ValueError('Absolute prefix/suffix geometry differs from projections')
    order, required, remaining = budget_order(future, budget)
    # This dynamic-size sync is real algorithm cost and is included in timing.
    count = int(required.item())
    kept = torch.cat((torch.arange(32, device=q.device), order[:count]+32)).sort().values
    return kept, remaining[:, count].amax()


class BudgetEngine(BlockEngine):
    def __init__(self, model, ids, past, budget):
        super().__init__(model, ids, past, DEFAULT_OPTIONS)
        self.budget = budget
        self.future_counts = []
        self.future_sizes = []
        self.dropped = []

    def forward(self, ids, target):
        if ids.shape != self.ids.shape or target.ndim != 1 or not target.numel():
            raise ValueError('Changed block geometry or empty active positions')
        self.ids.copy_(ids)
        hidden, q, k = self.pre.call()
        kept, dropped = select_budget(q, k, target, self.length, self.past_length, self.budget)
        self.kept = kept
        self.qpositions = kept+self.past_length
        self.kpositions = torch.cat((torch.arange(self.past_length, device=ids.device), self.qpositions))
        self.hidden = hidden.index_select(1, kept)
        for rotary in self.deep_rope:
            rotary.query_positions = self.qpositions
            rotary.key_positions = self.kpositions
        # Original-position RoPE and all formal-prefix KV are still used.
        normalized = self.post.call().index_select(1, target)
        core = self.core
        logits = (torch.nn.functional.linear(normalized, core.transformer.wte.weight)
                  if core.config.weight_tying else core.transformer.ff_out(normalized))
        if core.config.scale_logits:
            logits.mul_(1/math.sqrt(core.config.d_model))
        self.calls += 1
        self.future_counts.append(kept.numel()-32)
        self.future_sizes.append(self.length-32)
        self.dropped.append(dropped.detach())
        return logits


@torch.no_grad()
def generate_budget(model, prompt, *, gen_length, budget, trace=False):
    if prompt.ndim != 2 or prompt.shape[0] != 1 or gen_length not in (256, 512):
        raise ValueError('Batch one and fixed 256/512 generation budgets required')
    if not math.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError('Invalid future attention mass budget')
    DEFAULT_OPTIONS.validate(model.model.config.n_layers)
    torch.cuda.synchronize() if prompt.is_cuda else None
    began = time.perf_counter()
    x = torch.full((1, prompt.shape[1]+gen_length), MASK_ID, device=prompt.device, dtype=torch.long)
    x[:, :prompt.shape[1]] = prompt
    actions, dropped, counts, sizes = [], [], [], []
    telemetry = dict(warm_calls=0, refine_calls=0, block_setup_seconds=0.,
        budget=budget, observed_layer=4, mass_aggregation='mean heads separately for each active query',
        scope='Approximate physical pruning. Local layer4 mass is not a later-layer or accuracy certificate. '
              'Every selection sync, dynamic shape allocation, prefix preparation and full warm is charged.')
    nfe = 0
    for offset in range(0, gen_length, 32):
        start = prompt.shape[1]+offset
        output = model(x, use_cache=True)
        positions, values = commit(x, torch.arange(start, start+32, device=x.device),
                                  output.logits[:, start:start+32], .90, True)
        if trace:
            actions.append((positions.tolist(), values.tolist()))
        nfe += 1
        telemetry['warm_calls'] += 1
        past = [tuple(t[:, :, :start] for t in pair) for pair in output.past_key_values]
        versions = [(a._version, b._version) for a,b in past]
        del output
        engine = None
        block_calls = 1
        try:
            while True:
                target = (x[0, start:start+32] == MASK_ID).nonzero().flatten()
                if not target.numel():
                    break
                if block_calls >= 32:
                    raise RuntimeError('Native argmax fallback failed to make progress; do not silently retry')
                suffix = x[:, start:]
                if suffix.shape[1] == 32:
                    logits = model(suffix, past_key_values=past, use_cache=True).logits.index_select(1, target)
                else:
                    if engine is None:
                        torch.cuda.synchronize() if x.is_cuda else None
                        setup = time.perf_counter()
                        engine = BudgetEngine(model, suffix, past, budget)
                        torch.cuda.synchronize() if x.is_cuda else None
                        telemetry['block_setup_seconds'] += time.perf_counter()-setup
                    logits = engine.forward(suffix, target)
                positions, values = commit(x, start+target, logits, .90, True)
                if trace:
                    actions.append((positions.tolist(), values.tolist()))
                nfe += 1
                block_calls += 1
                telemetry['refine_calls'] += 1
                del logits
        finally:
            if engine is not None:
                counts.extend(engine.future_counts)
                sizes.extend(engine.future_sizes)
                dropped.extend(engine.dropped)
                engine.close()
            if versions != [(a._version,b._version) for a,b in past]:
                raise RuntimeError('Formal prefix cache was mutated')
    maximum = float(torch.stack(dropped).amax().item()) if dropped else 0.
    if maximum > budget+1e-6:
        raise RuntimeError('Observed layer4 budget violated')
    telemetry.update(max_observed_dropped_mass=maximum, future_counts=counts, future_sizes=sizes,
                     future_keep_fraction=sum(counts)/sum(sizes) if sizes else None,
                     formal_prefix_read_only=True)
    torch.cuda.synchronize() if x.is_cuda else None
    return Result(x, nfe, time.perf_counter()-began,
                  torch.cuda.max_memory_allocated()/2**30 if x.is_cuda else 0.), actions, telemetry
