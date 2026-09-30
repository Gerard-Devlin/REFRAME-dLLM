"""GPU-resident support selection with an optional retained-KV ablation."""
import math

import torch
from torch import nn

from ..llada_pruning import LLaDABlockForward, PositionedRotary
from .backend import selected_forward


class MappedRotary(PositionedRotary):
    def __init__(self, base, positions, original_length, past_length, keys):
        super().__init__(base, positions, original_length, past_length)
        self.key_positions = keys


class TensorForward(LLaDABlockForward):
    """Only for the prefix decoder's 32-token block + untouched MASK suffix.

    Selection and all positional maps remain on device. In the retained-KV
    ablation, removed future queries keep their per-layer block-warmup K/V;
    no cached value is mutated or installed into the formal prefix cache.
    """
    retain_reference = False

    @torch.no_grad()
    def __call__(self, ids, positions, prune=True, past_key_values=None,
                 use_cache=False, **kwargs):
        target = torch.as_tensor(positions, device=ids.device)
        length = ids.shape[1]
        if length <= 32 or self.config.support_keep_ratio == 1:
            return selected_forward(self.model, ids, target,
                                    past_key_values=past_key_values, use_cache=False).logits
        core = self.model.model
        hidden = core.transformer.wte(ids)
        if core.config.input_emb_norm:
            hidden = hidden * core.config.d_model**.5
        hidden = core.transformer.emb_drop(hidden)
        past_length = past_key_values[0][0].shape[-2]
        kept, dropped = None, None
        for number, block in enumerate(core.transformer.blocks, 1):
            layer_past = past_key_values[number-1]
            if number == self.config.prune_after_layer:
                hidden, capture, _ = self._captured_block(block, hidden, layer_past, False)
                relevance = self._relevance(capture, target)[past_length:]
                count = math.ceil((length-32)*self.config.support_keep_ratio)
                support = relevance[32:].topk(count, sorted=False).indices + 32
                kept = torch.cat((torch.arange(32, device=ids.device), support)).sort().values
                hidden = hidden.index_select(1, kept)
                if self.retain_reference:
                    remove = torch.ones(length, device=ids.device, dtype=torch.bool)
                    remove[kept] = False
                    dropped = torch.arange(length, device=ids.device)[remove]
                continue
            original = block.rotary_emb
            if kept is not None:
                if self.retain_reference:
                    full_past = self.reference[number-1]
                    old = tuple(t.index_select(-2, dropped + past_length) for t in full_past)
                    layer_past = tuple(torch.cat((p, q), dim=-2) for p, q in zip(layer_past, old))
                    keys = torch.cat((torch.arange(past_length, device=ids.device),
                                      dropped + past_length, kept + past_length))
                    block.rotary_emb = MappedRotary(original, kept, length, past_length, keys)
                else:
                    block.rotary_emb = PositionedRotary(original, kept, length, past_length)
            try:
                hidden, _ = block(hidden, attention_bias=None,
                                  layer_past=layer_past, use_cache=False)
            finally:
                block.rotary_emb = original
        hidden = core.transformer.ln_f(hidden[:, target])
        logits = (torch.nn.functional.linear(hidden, core.transformer.wte.weight)
                  if core.config.weight_tying else core.transformer.ff_out(hidden))
        if core.config.scale_logits:
            logits.mul_(1/math.sqrt(core.config.d_model))
        return logits


class ReferenceForward(TensorForward):
    retain_reference = True


def generate_tensor(model, prompt, *, reference=False, **kwargs):
    # Reuse the identical canvas/commit/timer implementation in a scoped patch.
    # This runner is single-threaded and isolated from the paper workers.
    from . import backend
    base_type = backend.ActiveForward
    base_forward = backend.selected_forward
    memo = {}

    def create(model, config):
        memo['forward'] = (ReferenceForward if reference else TensorForward)(model, config)
        return memo['forward']

    def observe(model, ids, target, **options):
        output = base_forward(model, ids, target, **options)
        if options.get('use_cache') and options.get('past_key_values') is None:
            memo['forward'].reference = output.past_key_values
        return output

    backend.ActiveForward, backend.selected_forward = create, observe
    try:
        return backend.generate_active_prefix(model, prompt, **kwargs)
    finally:
        backend.ActiveForward, backend.selected_forward = base_type, base_forward
