"""Offline mechanism control, requiring the current FULL teacher computation.

This is not an executable acceleration method. It replaces only the frozen
future K/V in an already selected support layout, temporarily, with values
computed by the full teacher at the SAME state. Formal prefix and selection
are held fixed. No input from a later teacher state is used.
"""
from contextlib import contextmanager

import torch

from .tensor_pruning import MappedRotary


@contextmanager
def fresh_frozen_future(forward, fresh, past_length):
    """Temporarily install same-state oracle future KV, never formal prefix KV."""
    if fresh is None or len(fresh) != len(forward.reference):
        raise ValueError('Expected all current teacher layers')
    if past_length < 0:
        raise ValueError('Invalid prefix length')
    for old_pair,new_pair in zip(forward.reference,fresh):
        for old,new in zip(old_pair,new_pair):
            if old.shape != new.shape or old.dtype != new.dtype or old.device != new.device:
                raise ValueError('Oracle must be aligned with the same canvas and prefix')
            if not torch.equal(old[:,:,:past_length],new[:,:,:past_length]):
                raise ValueError('Oracle is forbidden to change the formal prefix')
    length = forward.reference[0][0].shape[-2]-past_length
    if length <= 32:
        yield
        return
    if getattr(forward,'support_source',None) is not forward.reference:
        raise ValueError('Initialize the original support selection before the oracle control')
    kept = forward.kept
    if not torch.equal(kept[:32],torch.arange(32,device=kept.device)):
        raise ValueError('The current block must stay live')
    removed = torch.ones(length,dtype=torch.bool,device=kept.device)
    removed[kept] = False
    dropped = torch.arange(length,device=kept.device)[removed]+past_length
    if not dropped.numel():
        yield
        return
    frozen = torch.cat((torch.arange(past_length,device=kept.device),dropped))
    keys = torch.cat((frozen,kept+past_length))
    saved_past,saved_rotaries = forward.support_past,forward.support_rotaries
    replacement_past,replacement_rotaries = list(saved_past),list(saved_rotaries)
    for index,(block,pair) in enumerate(zip(forward.model.model.transformer.blocks,fresh)):
        if index+1 <= forward.config.prune_after_layer:
            continue
        if not torch.equal(saved_rotaries[index].key_positions,keys):
            raise ValueError('Original selected support layout changed')
        replacement_past[index] = tuple(t.index_select(-2,frozen) for t in pair)
        base = block.rotary_emb
        _,rotated = base(pair[0][:,:,:1],pair[0])
        rotary = MappedRotary(base,kept,length,past_length,keys)
        rotary.register_buffer('frozen_rotated_key',rotated.index_select(-2,frozen),persistent=False)
        replacement_rotaries[index] = rotary
    forward.support_past,forward.support_rotaries = replacement_past,replacement_rotaries
    try:
        yield
    finally:
        forward.support_past,forward.support_rotaries = saved_past,saved_rotaries


@contextmanager
def bf16_reduction(enabled):
    """Isolated numeric control; always restore the process-local setting."""
    settings = torch.backends.cuda.matmul
    previous = settings.allow_bf16_reduced_precision_reduction
    settings.allow_bf16_reduced_precision_reduction = enabled
    try:
        yield
    finally:
        settings.allow_bf16_reduced_precision_reduction = previous
