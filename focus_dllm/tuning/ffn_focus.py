"""Component ablation: fresh suffix attention, cached future MASK FFN outputs.

This is an approximation, not an exact pruning transformation. Only the ordinary
prefix-cache refinement path is changed; formal prefix K/V is never modified.
"""
from contextlib import contextmanager

import torch

from ..llada_common import MASK_ID
from .backend import selected_forward


@contextmanager
def capture_ffn(model, after_layer):
    """Capture FFN residuals during an already-required full block warm-up."""
    layers = model.model.transformer.blocks
    values = [None]*len(layers)
    handles = []
    try:
        for index, block in enumerate(layers):
            if index+1 > after_layer:
                def save(_module, _args, output, i=index):
                    values[i] = output.detach()
                handles.append(block.ff_out.register_forward_hook(save))
        yield values
    finally:
        for handle in handles:
            handle.remove()


class FFNFocusForward:
    def __init__(self, model, config):
        self.model, self.after_layer = model, config.prune_after_layer
        layers = model.model.transformer.blocks
        if not 0 <= self.after_layer <= len(layers):
            raise ValueError('FFN layer boundary outside the model')
        if model.training or any(block.dropout.p != 0 for block in layers):
            raise ValueError('FFN reuse requires evaluation without dropout')
        self.reference, self.prefix_length = None, None
        self.checked_suffix = False

    def install_reference(self, values, prefix_length):
        self.reference, self.prefix_length = values, prefix_length
        self.checked_suffix = False

    @torch.no_grad()
    def __call__(self, ids, positions, past_key_values=None, **kwargs):
        target = torch.as_tensor(positions, device=ids.device)
        layers = self.model.model.transformer.blocks
        if ids.shape[1] <= 32 or self.after_layer == len(layers):
            return selected_forward(self.model, ids, target,
                past_key_values=past_key_values, use_cache=False).logits
        if kwargs.get('use_cache') or kwargs.get('replace_position') is not None:
            raise ValueError('FFN focus only supports read-only prefix refinement')
        if self.reference is None or len(self.reference) != len(layers):
            raise ValueError('Missing block warm-up FFN residuals')
        if past_key_values is None or past_key_values[0][0].shape[-2] != self.prefix_length:
            raise ValueError('Prefix and FFN reference disagree')
        if not self.checked_suffix:
            if not torch.all(ids[:, 32:] == MASK_ID):
                raise ValueError('Only untouched future MASK positions may reuse FFN')
            self.checked_suffix = True
        handles = []
        try:
            for index, block in enumerate(layers):
                if index+1 <= self.after_layer:
                    continue
                cached = self.reference[index][:, self.prefix_length:]
                if cached.shape[:2] != ids.shape:
                    raise ValueError('FFN reference has the wrong suffix length')
                def select(_module, _args, output):
                    return output[:, :32]
                def restore(_module, _args, output, base=cached):
                    residual = base.clone()
                    residual[:, :32] = output
                    return residual
                handles.append(block.ff_norm.register_forward_hook(select))
                handles.append(block.ff_out.register_forward_hook(restore))
            return selected_forward(self.model, ids, target,
                past_key_values=past_key_values, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()


def generate_ffn_focus(model, prompt, **kwargs):
    """Use the unchanged active-prefix canvas and commit rules."""
    from . import backend
    previous_type, previous_forward = backend.ActiveForward, backend.selected_forward
    memo = {}
    def create(model, config):
        memo['forward'] = FFNFocusForward(model, config)
        return memo['forward']
    def observe(model, ids, target, **options):
        if options.get('use_cache') and options.get('past_key_values') is None:
            forward = memo['forward']
            with capture_ffn(model, forward.after_layer) as values:
                output = previous_forward(model, ids, target, **options)
            forward.install_reference(values, int(target[0]))
            return output
        return previous_forward(model, ids, target, **options)
    backend.ActiveForward, backend.selected_forward = create, observe
    try:
        return backend.generate_active_prefix(model, prompt, **kwargs)
    finally:
        backend.ActiveForward, backend.selected_forward = previous_type, previous_forward
