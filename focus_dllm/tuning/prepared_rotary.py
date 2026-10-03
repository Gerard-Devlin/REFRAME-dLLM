"""Shared execution control: rotate immutable formal prefix K once per block.

This is an engineering optimization for both native and compressed forwards,
not FOCUS's algorithmic contribution. Prefix values are never changed, and
position phases are the original model's phases. All cache creation must be
charged to end-to-end generation, including the first ordinary suffix call.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PrefixEntry:
    source: torch.Tensor
    version: int
    rotated: torch.Tensor


class PreparedRotary(nn.Module):
    def __init__(self, base, query_sin, query_cos, prefix, full_length):
        super().__init__()
        # Keep upstream nn.Module ownership unchanged while the scope is active.
        object.__setattr__(self, 'base', base)
        self.config = base.config
        self.query_sin, self.query_cos = query_sin, query_cos
        self.prefix = prefix
        self.full_length = full_length

    def get_rotary_embedding(self, length, device):
        return object.__getattribute__(self, 'base').get_rotary_embedding(length, device)

    def apply_rotary_pos_emb(self, sin, cos, tensor):
        return object.__getattribute__(self, 'base').apply_rotary_pos_emb(sin, cos, tensor)

    def forward(self, q, k, block_end_index=None):
        if block_end_index is not None:
            raise ValueError('Prepared prefix rotation does not support KV replacement')
        plen = self.prefix.rotated.shape[-2]
        if q.shape[-2] != self.query_sin.shape[-2] or k.shape[-2] != plen + q.shape[-2]:
            raise ValueError('Query/key lengths do not match prepared positions')
        qf = q.float() if self.config.rope_full_precision else q
        suffix = k[:, :, plen:]
        kf = suffix.float() if self.config.rope_full_precision else suffix
        with torch.autocast(q.device.type, enabled=False):
            sin, cos = self.query_sin.type_as(qf), self.query_cos.type_as(qf)
            qr = self.apply_rotary_pos_emb(sin, cos, qf).type_as(q)
            kr = self.apply_rotary_pos_emb(sin, cos, kf).type_as(k)
        return qr, torch.cat((self.prefix.rotated, kr), dim=-2)


class RotatedPrefixCache:
    """One model, one prompt; discard entries when a new warm prefix arrives."""
    def __init__(self):
        self.entries = {}
        self.geometry = {}
        self.cache_prepares = 0
        self.phase_gathers = 0
        self.saved = []

    def clear(self):
        self.entries.clear()
        self.geometry.clear()

    @staticmethod
    def original(base):
        return object.__getattribute__(base, 'base') if isinstance(base, PreparedRotary) else base

    def positioned(self, base, positions, original_length, past_key):
        base = self.original(base)
        past_length = past_key.shape[-2]
        full_length = past_length + original_length
        sin, cos = base.get_rotary_embedding(full_length, past_key.device)
        signature = (past_key.device, past_key.dtype, sin.data_ptr(), cos.data_ptr(),
                     sin.stride(), cos.stride(), bool(base.config.rope_full_precision))
        key = id(base)
        entry = self.entries.get(key)
        if entry is None or entry.source is not past_key or entry.version != past_key._version:
            kf = past_key.float() if base.config.rope_full_precision else past_key
            with torch.autocast(past_key.device.type, enabled=False):
                rotated = base.apply_rotary_pos_emb(sin[:, :, :past_length].type_as(kf),
                                                   cos[:, :, :past_length].type_as(kf), kf).type_as(past_key)
            entry = PrefixEntry(past_key, past_key._version, rotated)
            self.entries[key] = entry
            self.cache_prepares += 1
        if positions is None:
            phase_key = (signature, past_length, original_length, None)
        else:
            # Identity and tensor-version distinguish current selection maps.
            phase_key = (signature, past_length, original_length, id(positions), positions._version)
        phases = self.geometry.get(phase_key)
        if phases is None:
            if positions is None:
                phases = sin[:, :, past_length:full_length], cos[:, :, past_length:full_length]
            else:
                idx = positions + past_length
                phases = sin.index_select(2, idx), cos.index_select(2, idx)
                self.phase_gathers += 1
            # Hold the map and source buffers to prevent allocator/id reuse.
            self.geometry[phase_key] = (*phases, positions, sin, cos)
        return PreparedRotary(base, phases[0], phases[1], entry, full_length)

    @contextmanager
    def suffix_scope(self, model, past_key_values, original_length):
        if self.saved:
            raise RuntimeError('Nested prepared-rotary scopes are unsupported')
        self.geometry.clear()
        try:
            for block, pair in zip(model.model.transformer.blocks, past_key_values):
                original = block.rotary_emb
                self.saved.append((block, original))
                block.rotary_emb = self.positioned(original, None, original_length, pair[0])
            yield
        finally:
            for block, original in self.saved:
                block.rotary_emb = original
            self.saved.clear()
            self.geometry.clear()
