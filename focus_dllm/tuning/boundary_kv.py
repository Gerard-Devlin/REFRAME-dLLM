"""Use the current cut-layer features for dropped positions' next-layer K/V.

Only the first pruned layer receives these fresh projections. Deeper
layers retain the original reference policy; no hidden-state extrapolation,
extra model forward or persistent-prefix write is introduced.
"""
import torch

from .static_support import StaticSupportForward


class FreshBoundaryForward(StaticSupportForward):
    def __init__(self, model, config, support_count=32):
        super().__init__(model, config, support_count=support_count)
        if not 1 <= config.prune_after_layer < len(model.model.transformer.blocks):
            raise ValueError('Boundary K/V needs at least one layer after the cut')

    @torch.no_grad()
    def __call__(self, ids, positions, past_key_values=None, **kwargs):
        if ids.shape[1] <= 32 or self.config.support_keep_ratio == 1:
            return super().__call__(ids, positions, past_key_values=past_key_values, **kwargs)
        index = self.config.prune_after_layer
        cut = self.model.model.transformer.blocks[index-1]
        boundary = self.model.model.transformer.blocks[index]
        if not all(hasattr(boundary, name) for name in ('attn_norm', 'k_proj', 'v_proj')):
            raise ValueError('Boundary K/V supports LLaDA Llama blocks only')

        def capture(_block, _args, output):
            self.cut_hidden = output[0]

        def refresh(block, args, options):
            prefix_key, prefix_value = past_key_values[index]
            rotary = block.rotary_emb
            # The existing fixed map already lists prefix, dropped, live keys.
            # Slice it directly to avoid a per-step GPU nonzero/host sync.
            removed = rotary.key_positions[prefix_key.shape[-2]:-len(self.kept)]-prefix_key.shape[-2]
            if len(removed) == 0:
                return args, options
            hidden = block.attn_norm(self.cut_hidden.index_select(1, removed))
            key, value = block.k_proj(hidden), block.v_proj(hidden)
            dtype = key.dtype
            if block.q_norm is not None and block.k_norm is not None:
                key = block.k_norm(key).to(dtype)
            batch, length, _ = key.shape
            head_dim = block.config.d_model // block.config.n_heads
            heads = block.config.effective_n_kv_heads
            key = key.view(batch, length, heads, head_dim).transpose(1, 2)
            value = value.view(batch, length, heads, head_dim).transpose(1, 2)
            # Prefix is the original clean cache, never a temporary support cache.
            fresh = (torch.cat((prefix_key, key), -2), torch.cat((prefix_value, value), -2))
            self.support_past[index] = fresh
            options = dict(options, layer_past=fresh)
            base = rotary.base
            original_length = self.reference[index][0].shape[-2]
            locations = removed + prefix_key.shape[-2]
            rotated = key.float() if base.config.rope_full_precision else key
            with torch.autocast(key.device.type, enabled=False):
                sin, cos = base.get_rotary_embedding(original_length, key.device)
                sin, cos = sin.index_select(2, locations), cos.index_select(2, locations)
                rotated = base.apply_rotary_pos_emb(sin.type_as(rotated), cos.type_as(rotated), rotated)
            rotary.frozen_rotated_key = torch.cat((
                self.rotated_reference[index][:, :, :prefix_key.shape[-2]], rotated.type_as(key)), -2)
            return args, options

        captured = cut.register_forward_hook(capture)
        refreshed = boundary.register_forward_pre_hook(refresh, with_kwargs=True)
        try:
            return super().__call__(ids, positions, past_key_values=past_key_values, **kwargs)
        finally:
            captured.remove()
            refreshed.remove()
            if hasattr(self, 'cut_hidden'):
                del self.cut_hidden
