"""FOCUS-v2 experiment: fresh future-MASK pooling with proportional attention.

The shallow full suffix is computed on *this* canvas, never a teacher canvas or
old warm cache. A fixed deep-token budget is split between salient singletons
and ordered pools. Pool hidden states are means; their RoPE position is their
middle member (an approximation). Deep attention adds log(pool cardinality).

Proportional attention is an existing ToMe idea, not a novelty claim:
https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py
Only ordinary suffix calls are eligible. Formal prefix KV, warm calls, release
thresholds and the native sampler remain unchanged. This is not lossless.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ..llada_common import MASK_ID
from ..llada_pruning import LLaDABlockForward, PositionedRotary
from .backend import selected_forward


@dataclass(frozen=True)
class ProxyConfig:
    layer: int = 4
    keep_ratio: float = .3125
    exact_fraction: float = .25
    weighted: bool = True
    mass_implementation: str = 'feature'
    block_length: int = 32
    # Numerical control only: singleton full attention with padded head shape.
    force_mass_kernel: bool = False

    def validate(self, layers):
        if not 1 <= self.layer < layers:
            raise ValueError('The pooling point must leave deep layers')
        if not 0 < self.keep_ratio <= 1 or not 0 <= self.exact_fraction <= 1:
            raise ValueError('Invalid pool budget or exact fraction')
        if self.block_length < 1:
            raise ValueError('Invalid block length')
        if self.mass_implementation not in ('feature', 'repeat'):
            raise ValueError('Unknown mass implementation')


@dataclass(frozen=True)
class Partition:
    positions: tuple
    groups: tuple
    future_count: int
    exact_count: int
    pool_count: int

    @property
    def masses(self):
        return tuple(len(group) for group in self.groups)


def make_partition(length, future, relevance, keep_ratio, exact_fraction):
    """Cover every input position exactly once; only future MASKs can merge.

    Pools are consecutive slices of the ordered remaining future positions,
    after removing dominant singletons. They can straddle a retained singleton;
    membership, rather than the representative's position, defines their mass.
    """
    future = sorted(set(int(p) for p in future))
    if length < 1 or any(p < 0 or p >= length for p in future):
        raise ValueError('Invalid future positions')
    if not 0 < keep_ratio <= 1 or not 0 <= exact_fraction <= 1:
        raise ValueError('Invalid partition budget')
    budget = min(len(future), math.ceil(len(future) * keep_ratio))
    exact_count = min(budget, math.floor(budget * exact_fraction))
    # With no pooling budget left, retain all positions rather than silently
    # throwing away members. exact_fraction=1 is the exact identity control.
    if budget == len(future) or exact_count == budget:
        return Partition(tuple(range(length)), tuple((p,) for p in range(length)),
                         len(future), len(future), 0)
    idx = torch.as_tensor(future, device=relevance.device, dtype=torch.long)
    exact = (idx[torch.topk(relevance.index_select(0, idx), exact_count,
                          sorted=False).indices].tolist() if exact_count else [])
    exact_set, future_set = set(exact), set(future)
    remaining = [p for p in future if p not in exact_set]
    pool_count = min(budget - exact_count, len(remaining))
    entries = [(p, (p,)) for p in range(length)
               if p not in future_set or p in exact_set]
    # Integer boundaries match tensor_split: first groups receive the remainder.
    width, extra = divmod(len(remaining), pool_count)
    start = 0
    for number in range(pool_count):
        end = start + width + (number < extra)
        members = tuple(remaining[start:end])
        entries.append((members[len(members) // 2], members))
        start = end
    entries.sort(key=lambda pair: pair[0])
    return Partition(tuple(p for p, _ in entries), tuple(g for _, g in entries),
                     len(future), exact_count, pool_count)


def pool_hidden(hidden, partition):
    """One FP32 scatter reduction; no per-group CUDA launch loop."""
    if hidden.shape[0] != 1 or sum(partition.masses) != hidden.shape[1]:
        raise ValueError('Partition must cover the batch=1 suffix')
    if len(partition.positions) == hidden.shape[1]:
        return hidden
    membership = [None] * hidden.shape[1]
    for number, group in enumerate(partition.groups):
        for position in group:
            membership[position] = number
    index = torch.tensor(membership, device=hidden.device, dtype=torch.long)
    sums = torch.zeros((len(partition.groups), hidden.shape[-1]),
                       device=hidden.device, dtype=torch.float32)
    sums.index_add_(0, index, hidden[0].float())
    mass = torch.tensor(partition.masses, device=hidden.device, dtype=torch.float32)
    return (sums / mass[:, None]).to(hidden.dtype).unsqueeze(0)


def mass_attention(flash, q, k, v, key_log_mass, **kwargs):
    """Encode an additive key bias in one padded feature, after RoPE.

    Inputs follow FlashAttention's B,T,H,D order. Keep the *original* scale;
    default 1/sqrt(padded_dim) would change the model. BF16 quantization and a
    different Flash kernel shape can still change actions and require controls.
    No dense attention matrix, SDPA fallback, or online teacher is used.
    """
    dim = q.shape[-1]
    if dim != k.shape[-1] or dim != v.shape[-1] or dim >= 256:
        raise ValueError('Expected equal Flash head dimensions below 256')
    if key_log_mass.numel() != k.shape[1] or key_log_mass.ndim != 1:
        raise ValueError('Key mass does not cover prefix plus compact suffix')
    if kwargs.get('causal', False) or kwargs.get('dropout_p', 0.) != 0.:
        raise ValueError('Only noncausal inference attention is supported')
    if kwargs.get('softmax_scale') is not None:
        raise ValueError('The original model scale is set by this wrapper')
    scale = 1 / math.sqrt(dim)
    padding = 8 - dim % 8
    qa, ka, va = (F.pad(t, (0, padding)) for t in (q, k, v))
    qa[..., dim] = 1.
    ka[..., dim] = (key_log_mass / scale).to(k.dtype)[None, :, None]
    return flash(qa, ka, va, softmax_scale=scale, **kwargs)[..., :dim]


def repeated_attention(flash, q, k, v, key_expansion, **kwargs):
    """Same proportional attention by duplicate proxy K/V, with head_dim=128.

    This pays full logical key length, two gathers, and attention over expanded
    keys. Only deep projections/FFN/queries are compressed. It is an execution
    alternative to the feature bias, not a claim that KV/attention is removed.
    """
    return flash(q, k.index_select(1, key_expansion),
                 v.index_select(1, key_expansion), **kwargs)


@contextmanager
def proportional_flash(block, key_log_mass, key_expansion=None):
    original = block.flash_attn_func
    if original is None:
        raise RuntimeError('FOCUS-v2 requires the shared FlashAttention backend')
    def wrapped(q, k, v, **kwargs):
        if key_expansion is not None:
            return repeated_attention(original, q, k, v, key_expansion, **kwargs)
        return mass_attention(original, q, k, v, key_log_mass, **kwargs)
    block.flash_attn_func = wrapped
    try:
        yield
    finally:
        block.flash_attn_func = original


class FocusV2Forward:
    """Recompute every live pool each call, without changing formal KV."""
    def __init__(self, model, config=ProxyConfig(), rotary_factory=None):
        config.validate(model.model.config.n_layers)
        self.model, self.config = model, config
        self.rotary_factory = rotary_factory
        self.records = []

    @torch.no_grad()
    def __call__(self, ids, targets, *, past_key_values=None):
        config = self.config
        if ids.shape[0] != 1 or not targets or len(set(targets)) != len(targets):
            raise ValueError('Expected batch=1 and unique active positions')
        if any(p < 0 or p >= min(config.block_length, ids.shape[1]) for p in targets):
            raise ValueError('Targets must belong to the current suffix block')
        tokens = ids[0].tolist()
        if any(tokens[p] != MASK_ID for p in targets):
            raise ValueError('Targets must be current MASK positions')
        if any(tokens[p] == MASK_ID and p not in set(targets)
               for p in range(min(config.block_length, len(tokens)))):
            raise ValueError('Every current block MASK must be active')
        future = [p for p in range(config.block_length, len(tokens)) if tokens[p] == MASK_ID]
        if (not future or config.keep_ratio == 1) and not config.force_mass_kernel:
            self.records.append(dict(identity=True, original_tokens=len(tokens),
                                     deep_tokens=len(tokens), future=len(future)))
            return selected_forward(self.model, ids, torch.tensor(targets, device=ids.device),
                                    past_key_values=past_key_values, use_cache=False).logits
        core = self.model.model
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        hidden = core.transformer.wte(ids)
        if core.config.input_emb_norm:
            hidden = hidden * math.sqrt(core.config.d_model)
        hidden = core.transformer.emb_drop(hidden)
        partition, positions, log_mass, expansion = None, None, None, None
        for layer, block in enumerate(core.transformer.blocks, start=1):
            past = None if past_key_values is None else past_key_values[layer - 1]
            if layer <= config.layer:
                if layer == config.layer and config.keep_ratio < 1:
                    hidden, capture, _ = LLaDABlockForward._captured_block(
                        block, hidden, layer_past=past, use_cache=False)
                    relevance = LLaDABlockForward._relevance(capture, targets)[past_length:]
                else:
                    hidden, _ = block(hidden, attention_bias=None, layer_past=past, use_cache=False)
                if layer == config.layer:
                    if config.keep_ratio == 1:
                        relevance = torch.zeros(len(tokens), device=ids.device)
                    partition = make_partition(len(tokens), future, relevance,
                                               config.keep_ratio, config.exact_fraction)
                    hidden = pool_hidden(hidden, partition)
                    positions = torch.tensor(partition.positions, device=ids.device)
                    masses = torch.tensor((1,) * past_length + partition.masses,
                                          device=ids.device, dtype=torch.float32)
                    log_mass = masses.log()
                    if config.mass_implementation == 'repeat':
                        expansion = torch.tensor(
                            list(range(past_length)) + [past_length + group
                            for group, count in enumerate(partition.masses)
                            for _ in range(count)], device=ids.device, dtype=torch.long)
                continue
            rotary = block.rotary_emb
            block.rotary_emb = (PositionedRotary(rotary, positions, len(tokens), past_length)
                               if self.rotary_factory is None else
                               self.rotary_factory.positioned(rotary, positions, len(tokens), past[0]))
            try:
                if config.weighted and (partition.pool_count or config.force_mass_kernel):
                    with proportional_flash(block, log_mass, expansion):
                        hidden, _ = block(hidden, attention_bias=None, layer_past=past, use_cache=False)
                else:
                    hidden, _ = block(hidden, attention_bias=None, layer_past=past, use_cache=False)
            finally:
                block.rotary_emb = rotary
        compact = {p: i for i, p in enumerate(partition.positions)}
        gather = torch.tensor([compact[p] for p in targets], device=ids.device)
        hidden = core.transformer.ln_f(hidden).index_select(1, gather)
        logits = (F.linear(hidden, core.transformer.wte.weight) if core.config.weight_tying
                  else core.transformer.ff_out(hidden))
        if core.config.scale_logits:
            logits = logits * (1 / math.sqrt(core.config.d_model))
        self.records.append(dict(identity=len(partition.positions) == len(tokens),
                                 original_tokens=len(tokens), deep_tokens=len(partition.positions),
                                 future=len(future), exact=partition.exact_count,
                                 pools=partition.pool_count, future_mass=len(future),
                                 weighted=config.weighted, rope_policy='middle_member',
                                 mass_implementation=config.mass_implementation,
                                 deep_key_tokens=(past_length + len(tokens) if expansion is not None
                                                  else past_length + len(partition.positions))))
        return logits
