"""Reuse frozen keys' RoPE results instead of rotating them every refinement."""
from contextlib import contextmanager

import torch

from .tensor_pruning import TensorForward


class RotatedForward(TensorForward):
    retain_reference = True

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        # The final-block exact fallback goes through the official model,
        # including its internally constructed (unused) attention bias.
        # Leave that path completely outside our attention replacement.
        if args[0].shape[1] <= 32 or self.config.support_keep_ratio == 1:
            return super().__call__(*args, **kwargs)
        if getattr(self, 'rotated_reference_source', None) is not self.reference:
            self.rotated_reference_source = self.reference
            self.rotated_reference = []
            for block, (key, _) in zip(self.model.model.transformer.blocks, self.reference):
                _, rotated = block.rotary_emb(key[:, :, :1], key)
                self.rotated_reference.append(rotated)
        saved = []
        for number, block in enumerate(self.model.model.transformer.blocks):
            previous = block.attention
            def attention(q, k, v, attention_bias=None, layer_past=None,
                          use_cache=False, _block=block, _number=number, **options):
                if use_cache or attention_bias is not None:
                    raise ValueError('Rotation experiment only supports cache-read-only Flash refinement')
                dtype = k.dtype
                if _block.q_norm is not None and _block.k_norm is not None:
                    q, k = _block.q_norm(q).to(dtype), _block.k_norm(k).to(dtype)
                batch, length, width = q.shape
                head_dim = width // _block.config.n_heads
                q = q.view(batch, length, _block.config.n_heads, head_dim).transpose(1, 2)
                k = k.view(batch, length, _block.config.effective_n_kv_heads, head_dim).transpose(1, 2)
                v = v.view(batch, length, _block.config.effective_n_kv_heads, head_dim).transpose(1, 2)
                rotary = _block.rotary_emb
                total_length = self.reference[_number][0].shape[-2]
                base = getattr(rotary, 'base', rotary)
                if hasattr(rotary, 'query_positions'):
                    positions = rotary.query_positions
                    frozen_positions = rotary.key_positions[:-length]
                else:
                    past_length = layer_past[0].shape[-2]
                    positions = torch.arange(past_length, total_length, device=q.device)
                    frozen_positions = None
                qf, kf = (q.float(), k.float()) if base.config.rope_full_precision else (q, k)
                with torch.autocast(q.device.type, enabled=False):
                    sin, cos = base.get_rotary_embedding(total_length, q.device)
                    qs, qc = sin.index_select(2, positions), cos.index_select(2, positions)
                    qf = base.apply_rotary_pos_emb(qs.type_as(qf), qc.type_as(qf), qf)
                    kf = base.apply_rotary_pos_emb(qs.type_as(kf), qc.type_as(kf), kf)
                q, k = qf.type_as(q), kf.type_as(k)
                if hasattr(rotary, 'frozen_rotated_key'):
                    frozen = rotary.frozen_rotated_key
                else:
                    frozen = self.rotated_reference[_number]
                    frozen = (frozen[:, :, :layer_past[0].shape[-2]] if frozen_positions is None
                              else frozen.index_select(-2, frozen_positions))
                k = torch.cat((frozen, k), -2)
                v = torch.cat((layer_past[1], v), -2)
                attended = _block._scaled_dot_product_attention(q, k, v,
                              attn_mask=None, dropout_p=0., is_causal=False)
                attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
                return _block.attn_out(attended), None
            block.attention = attention
            saved.append((block, previous))
        try:
            return self.body(*args, **kwargs)
        finally:
            for block, previous in saved:
                block.attention = previous

    def body(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


def generate_rotated(model, prompt, **kwargs):
    from . import tensor_pruning
    original = tensor_pruning.ReferenceForward
    tensor_pruning.ReferenceForward = RotatedForward
    try:
        return tensor_pruning.generate_tensor(model, prompt, reference=True, **kwargs)
    finally:
        tensor_pruning.ReferenceForward = original
