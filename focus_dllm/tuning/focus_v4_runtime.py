"""FOCUS-v4 execution ablations, isolated from the frozen paper methods.

The physical pruning rule, BF16 operator shapes, target-only head and native
>= threshold / argmax fallback are preserved. Fixed-shape CUDA Graph segments
surround dynamic support selection. Graph creation and prefix preparation are
charged to each request. No trajectory batching, stale KV or early stopping.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import time

import torch
from torch import nn

from ..llada_common import MASK_ID
from ..llada_decode import Result, _selected_positions
from ..llada_pruning import LLaDABlockForward


@dataclass(frozen=True)
class Options:
    layer: int = 4
    keep: float = .3125
    graph: bool = False
    prepared_rope: bool = False
    fused_statistics: bool = False
    borrow_prefix: bool = False
    graph_start: int = 0
    graph_min_remaining: int = 0
    graph_warmups: int = 2

    def validate(self, layers):
        if not 1 <= self.layer < layers or not 0 < self.keep < 1:
            raise ValueError('A real physical pruning point and keep ratio required')
        if min(self.graph_start, self.graph_min_remaining, self.graph_warmups) < 0:
            raise ValueError('Nonnegative graph dispatch settings required')
        if self.graph and self.graph_warmups == 0 and self.graph_start < 1:
            raise ValueError('Zero-warmup capture requires previously executed identical segments')


# Fixed after the two development screens. CUDA Graph remains an opt-in
# ablation because per-request capture cost outweighed its replay savings.
DEFAULT_OPTIONS = Options(prepared_rope=True, fused_statistics=True, borrow_prefix=True)


def select_support(q, k, target, suffix_length, past_length, keep, block=32):
    """Identical score/reduction shapes; selection maps never leave the GPU."""
    if target.ndim != 1 or target.numel() == 0 or suffix_length <= block:
        raise ValueError('Nonempty active block and future suffix required')
    scores = torch.matmul(q.index_select(-2, target), k.transpose(-2, -1))
    scores = scores / math.sqrt(q.shape[-1])
    relevance = scores.float().softmax(-1).mean(dim=(0, 1, 2))[past_length:]
    count = math.ceil((suffix_length-block)*keep)
    future = torch.topk(relevance[block:], count, sorted=False).indices + block
    return torch.cat((torch.arange(block, device=q.device), future)).sort().values


def commit(x, target, logits, threshold, fused=False):
    if fused:
        from .flash_statistics import statistics
        confidence, tokens = statistics(logits[0])
        tokens, confidence = tokens[None], confidence[None]
    else:
        tokens = logits.argmax(-1)
        confidence = logits.double().softmax(-1).gather(-1, tokens[..., None]).squeeze(-1)
    take = _selected_positions(confidence, threshold)
    positions, values = target[take], tokens[0, take]
    x[0, positions] = values
    return positions, values


class DeviceRotary(nn.Module):
    """One shared map, original phases; optional once-per-block prefix RoPE.

    Prepared prefix values are private, immutable for this block. The formal
    raw cache is never modified. Prefix rotation is still paid in setup.
    """
    def __init__(self, base, query_positions, key_positions, length, past, prepared):
        super().__init__()
        object.__setattr__(self, 'base', base)
        self.query_positions = query_positions
        self.key_positions = key_positions
        self.length, self.past_length = length, past.shape[-2]
        self.register_buffer('rotated_prefix', None, persistent=False)
        if prepared:
            sin, cos = base.get_rotary_embedding(length, past.device)
            value = past.float() if base.config.rope_full_precision else past
            with torch.autocast(past.device.type, enabled=False):
                value = base.apply_rotary_pos_emb(sin[:, :, :self.past_length].type_as(value),
                                                cos[:, :, :self.past_length].type_as(value), value)
            self.rotated_prefix = value.type_as(past)

    def forward(self, q, k, block_end_index=None):
        base = object.__getattribute__(self, 'base')
        qf = q.float() if base.config.rope_full_precision else q
        prefix = self.past_length if self.rotated_prefix is not None else 0
        raw = k[:, :, prefix:]
        kf = raw.float() if base.config.rope_full_precision else raw
        with torch.autocast(q.device.type, enabled=False):
            sin, cos = base.get_rotary_embedding(self.length, q.device)
            qr = base.apply_rotary_pos_emb(sin.index_select(2, self.query_positions).type_as(qf),
                                          cos.index_select(2, self.query_positions).type_as(qf), qf)
            keys = self.key_positions[prefix:]
            kr = base.apply_rotary_pos_emb(sin.index_select(2, keys).type_as(kf),
                                          cos.index_select(2, keys).type_as(kf), kf).type_as(k)
        if prefix:
            kr = torch.cat((self.rotated_prefix, kr), dim=-2)
        return qr.type_as(q), kr


class Segment:
    """Capture only fixed geometry. Dynamic maps live in owned stable buffers."""
    def __init__(self, function, graph):
        self.function, self.graph_mode = function, graph
        self.graph = None
        self.output = None
        self.setup_seconds = 0.
        self.replays = 0

    def prepare(self, warmups=2):
        if not self.graph_mode:
            return
        torch.cuda.synchronize()
        started = time.perf_counter()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmups):
                self.function()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = self.function()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter()-started

    def call(self):
        if self.graph is None:
            return self.function()
        self.graph.replay()
        self.replays += 1
        return self.output

    def close(self):
        if self.graph is not None:
            self.graph.reset()
        self.output = self.function = self.graph = None


class BlockEngine:
    """Two graph segments per block; same active-row head stays outside graphs."""
    def __init__(self, model, ids, past, options):
        core = model.model
        options.validate(core.config.n_layers)
        self.model, self.core, self.options = model, core, options
        self.length = ids.shape[1]
        if self.length <= 32:
            raise ValueError('Final block uses the original exact-forward branch')
        self.ids = ids.detach().clone()
        # Both segments only read layer_past, never use replace_position and
        # return no cache. Borrowing keeps the formal tensors alive throughout
        # the block; graphs are destroyed before the next warm-up. This avoids
        # copying a complete read-only prefix without reusing stale future KV.
        self.past = ([tuple(t for t in pair) for pair in past] if options.borrow_prefix else
                     [tuple(t.detach().clone() for t in pair) for pair in past])
        self.past_length = self.past[0][0].shape[-2]
        self.compact_length = 32+math.ceil((self.length-32)*options.keep)
        self.hidden = torch.empty((1, self.compact_length, core.config.d_model),
                                  device=ids.device, dtype=next(core.parameters()).dtype)
        self.kept = torch.arange(self.compact_length, device=ids.device)
        self.qpositions = self.kept+self.past_length
        self.kpositions = torch.cat((torch.arange(self.past_length, device=ids.device), self.qpositions))
        fullq = torch.arange(self.past_length, self.past_length+self.length, device=ids.device)
        fullk = torch.arange(self.past_length+self.length, device=ids.device)
        self.shallow_rope, self.deep_rope = [], []
        for index, block in enumerate(core.transformer.blocks):
            if index < options.layer:
                self.shallow_rope.append(DeviceRotary(block.rotary_emb, fullq, fullk,
                    self.past_length+self.length, self.past[index][0], options.prepared_rope))
            else:
                self.deep_rope.append(DeviceRotary(block.rotary_emb, self.qpositions, self.kpositions,
                    self.past_length+self.length, self.past[index][0], options.prepared_rope))
        immediate = options.graph and options.graph_start == 0
        self.pre = Segment(self.shallow, immediate)
        self.post = Segment(self.deep, immediate)
        self.calls = 0
        if immediate:
            self.pre.prepare(options.graph_warmups)
            hidden, _, _ = self.pre.call()
            self.hidden.copy_(hidden[:, :self.compact_length])
            self.post.prepare(options.graph_warmups)

    @contextmanager
    def rotary_scope(self, blocks, ropes):
        saved = [(b, b.rotary_emb) for b in blocks]
        try:
            for block, rotary in zip(blocks, ropes):
                block.rotary_emb = rotary
            yield
        finally:
            for block, rotary in saved:
                block.rotary_emb = rotary

    def shallow(self):
        core = self.core
        hidden = core.transformer.wte(self.ids)
        if core.config.input_emb_norm:
            hidden = hidden * core.config.d_model**.5
        hidden = core.transformer.emb_drop(hidden)
        blocks = list(core.transformer.blocks[:self.options.layer])
        with self.rotary_scope(blocks, self.shallow_rope):
            for index, block in enumerate(blocks):
                if index == self.options.layer-1:
                    hidden, capture, _ = LLaDABlockForward._captured_block(
                        block, hidden, self.past[index], False)
                else:
                    hidden, _ = block(hidden, attention_bias=None,
                                      layer_past=self.past[index], use_cache=False)
        return hidden, capture['q'], capture['k']

    def deep(self):
        hidden = self.hidden
        blocks = list(self.core.transformer.blocks[self.options.layer:])
        with self.rotary_scope(blocks, self.deep_rope):
            for index, block in enumerate(blocks, self.options.layer):
                hidden, _ = block(hidden, attention_bias=None,
                                  layer_past=self.past[index], use_cache=False)
        # Match the main method's full retained-row final normalization.
        return self.core.transformer.ln_f(hidden)

    def forward(self, ids, target):
        if ids.shape != self.ids.shape or target.ndim != 1 or not target.numel():
            raise ValueError('Changed block geometry or empty target')
        self.ids.copy_(ids)
        if (self.options.graph and self.pre.graph is None and
                self.calls >= self.options.graph_start and
                target.numel() >= self.options.graph_min_remaining):
            # No future NFE is consulted. Before a zero-warmup capture, both
            # fixed-shape segments have already executed graph_start times.
            self.pre.graph_mode = self.post.graph_mode = True
            self.pre.prepare(self.options.graph_warmups)
            self.post.prepare(self.options.graph_warmups)
        hidden, q, k = self.pre.call()
        kept = select_support(q, k, target, self.length, self.past_length, self.options.keep)
        self.kept.copy_(kept)
        self.qpositions.copy_(kept+self.past_length)
        self.kpositions[self.past_length:].copy_(self.qpositions)
        self.hidden.copy_(hidden.index_select(1, kept))
        normalized = self.post.call().index_select(1, target)
        core = self.core
        logits = (torch.nn.functional.linear(normalized, core.transformer.wte.weight)
                  if core.config.weight_tying else core.transformer.ff_out(normalized))
        if core.config.scale_logits:
            logits.mul_(1/math.sqrt(core.config.d_model))
        self.calls += 1
        return logits

    def close(self):
        self.pre.close()
        self.post.close()


@torch.no_grad()
def generate_v4(model, prompt, *, gen_length=256, threshold=.90,
                options=DEFAULT_OPTIONS, trace=False):
    if prompt.shape[0] != 1 or gen_length not in (256, 512):
        raise ValueError('Fixed batch-one 256/512 protocol required')
    options.validate(model.model.config.n_layers)
    if options.graph and not prompt.is_cuda:
        raise ValueError('CUDA Graph requires CUDA')
    torch.cuda.synchronize() if prompt.is_cuda else None
    started = time.perf_counter()
    x = torch.full((1, prompt.shape[1]+gen_length), MASK_ID, device=prompt.device, dtype=torch.long)
    x[:, :prompt.shape[1]] = prompt
    nfe, actions = 0, []
    telemetry = dict(graph_setup_seconds=0., block_setup_seconds=0., graph_replays=0,
                     refine_calls=0, warm_calls=0, projected_active_rows=0,
                     scope='Per-request wall time includes graph capture, cache copies and RoPE preparation. '
                           'Graph warm-up/capture compute is setup, not committed decoding NFE.')
    for offset in range(0, gen_length, 32):
        start = prompt.shape[1]+offset
        output = model(x, use_cache=True)
        target = torch.arange(start, start+32, device=x.device)
        positions, values = commit(x, target, output.logits[:, start:start+32], threshold,
                                  options.fused_statistics)
        if trace:
            actions.append((positions.tolist(), values.tolist()))
        nfe += 1
        telemetry['warm_calls'] += 1
        telemetry['projected_active_rows'] += 32
        past = [tuple(t[:, :, :start] for t in pair) for pair in output.past_key_values]
        del output
        engine = None
        initial_graph_setup = 0.
        try:
            # One nonzero shape sync per refine; no separate bool, tolist,
            # support round-trip or scalar saliency logging in the hot path.
            while True:
                target = (x[0, start:start+32] == MASK_ID).nonzero().flatten()
                if not target.numel():
                    break
                suffix = x[:, start:]
                if suffix.shape[1] <= 32:
                    logits = model(suffix, past_key_values=past, use_cache=True).logits.index_select(1, target)
                else:
                    if engine is None:
                        torch.cuda.synchronize() if x.is_cuda else None
                        setup_start = time.perf_counter()
                        engine = BlockEngine(model, suffix, past, options)
                        torch.cuda.synchronize() if x.is_cuda else None
                        telemetry['block_setup_seconds'] += time.perf_counter()-setup_start
                        initial_graph_setup = engine.pre.setup_seconds+engine.post.setup_seconds
                    logits = engine.forward(suffix, target)
                positions, values = commit(x, start+target, logits, threshold, options.fused_statistics)
                if trace:
                    actions.append((positions.tolist(), values.tolist()))
                nfe += 1
                telemetry['refine_calls'] += 1
                telemetry['projected_active_rows'] += target.numel()
                del logits
        finally:
            if engine is not None:
                telemetry['graph_replays'] += engine.pre.replays+engine.post.replays
                graph_setup = engine.pre.setup_seconds+engine.post.setup_seconds
                telemetry['graph_setup_seconds'] += graph_setup
                telemetry['block_setup_seconds'] += graph_setup-initial_graph_setup
                engine.close()
                del engine
    torch.cuda.synchronize() if x.is_cuda else None
    elapsed = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated()/2**30 if x.is_cuda else 0.
    return Result(x, nfe, elapsed, peak), actions, telemetry
