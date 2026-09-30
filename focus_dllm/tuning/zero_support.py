"""Avoid repeated selection and cache concatenation when all future queries wait."""
import math

import torch

from .rotation import RotatedForward
from .tensor_pruning import MappedRotary


class ZeroForward(RotatedForward):
    @torch.no_grad()
    def body(self, ids, positions, past_key_values=None, **kwargs):
        if self.config.support_keep_ratio != 0:
            raise ValueError('This specialized path requires keep=0')
        core = self.model.model
        past_length = past_key_values[0][0].shape[-2]
        if getattr(self, 'zero_source', None) is not self.reference:
            self.zero_source = self.reference
            kept = torch.arange(32, device=ids.device)
            keys = torch.cat((torch.arange(past_length, device=ids.device),
                              torch.arange(past_length+32, self.reference[0][0].shape[-2], device=ids.device),
                              kept+past_length))
            self.zero_past, self.zero_rotaries = [], []
            for index, (block, pair) in enumerate(zip(core.transformer.blocks, self.reference)):
                self.zero_past.append(tuple(torch.cat((t[:, :, :past_length], t[:, :, past_length+32:]), -2) for t in pair))
                rotary = MappedRotary(block.rotary_emb, kept, ids.shape[1], past_length, keys)
                rotary.register_buffer('frozen_rotated_key',
                    self.rotated_reference[index].index_select(-2, keys[:-32]), persistent=False)
                self.zero_rotaries.append(rotary)
        hidden = core.transformer.wte(ids)
        if core.config.input_emb_norm:
            hidden = hidden * core.config.d_model**.5
        hidden = core.transformer.emb_drop(hidden)
        for number, block in enumerate(core.transformer.blocks, 1):
            original = block.rotary_emb
            deep = number > self.config.prune_after_layer
            if deep:
                block.rotary_emb = self.zero_rotaries[number-1]
            try:
                hidden, _ = block(hidden, attention_bias=None,
                    layer_past=(self.zero_past[number-1] if deep else past_key_values[number-1]), use_cache=False)
            finally:
                block.rotary_emb = original
            if number == self.config.prune_after_layer:
                hidden = hidden[:, :32]
        target = torch.as_tensor(positions, device=ids.device)
        hidden = core.transformer.ln_f(hidden.index_select(1, target))
        logits = (torch.nn.functional.linear(hidden, core.transformer.wte.weight)
                  if core.config.weight_tying else core.transformer.ff_out(hidden))
        if core.config.scale_logits:
            logits.mul_(1/math.sqrt(core.config.d_model))
        return logits


def generate_zero(model, prompt, **kwargs):
    from . import tensor_pruning
    original = tensor_pruning.ReferenceForward
    tensor_pruning.ReferenceForward = ZeroForward
    try:
        return tensor_pruning.generate_tensor(model, prompt, reference=True, **kwargs)
    finally:
        tensor_pruning.ReferenceForward = original
