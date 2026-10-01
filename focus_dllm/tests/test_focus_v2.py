import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from focus_dllm.llada_common import MASK_ID
from focus_dllm.tuning.focus_v2 import (
    FocusV2Forward, ProxyConfig, make_partition, mass_attention, pool_hidden,
    proportional_flash, repeated_attention,
)


def dense_flash(q, k, v, *, softmax_scale=None, **kwargs):
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if q.shape[1] != k.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scale = 1 / math.sqrt(q.shape[-1]) if softmax_scale is None else softmax_scale
    return ((q @ k.transpose(-2, -1) * scale).softmax(-1) @ v).transpose(1, 2)


def test_partition_protects_current_and_real_tokens_and_conserves_mass():
    plan = make_partition(12, [4, 5, 7, 8, 10, 11], torch.arange(12.), .5, .34)
    assert plan.exact_count == 1 and plan.pool_count == 2
    assert sorted(p for group in plan.groups for p in group) == list(range(12))
    assert sum(plan.masses) == 12
    for p in [0, 1, 2, 3, 6, 9, 11]:
        assert (p,) in plan.groups
    assert plan.positions == tuple(sorted(plan.positions))


def test_pool_mean_is_current_state_not_representative_or_future_canvas():
    plan = make_partition(8, range(4, 8), torch.zeros(8), .5, 0.)
    hidden = torch.arange(16.).reshape(1, 8, 2)
    version = hidden._version
    result = pool_hidden(hidden, plan)
    assert torch.equal(result[:, :4], hidden[:, :4])
    assert torch.equal(result[0, 4], hidden[0, 4:6].mean(0))
    assert torch.equal(result[0, 5], hidden[0, 6:8].mean(0))
    assert hidden._version == version


@pytest.mark.parametrize('future,ratio,fraction', [([], .3, .2), ([4, 5], 1., .25), ([4, 5], .5, 1.)])
def test_exact_partition_controls(future, ratio, fraction):
    plan = make_partition(6, future, torch.zeros(6), ratio, fraction)
    hidden = torch.randn(1, 6, 8)
    assert plan.positions == tuple(range(6)) and plan.masses == (1,) * 6
    assert pool_hidden(hidden, plan) is hidden


def test_proportional_attention_equals_identical_key_value_duplicates():
    torch.manual_seed(12)
    q = torch.randn(1, 5, 4, 8, dtype=torch.float64)
    k, v = (torch.randn(1, 3, 2, 8, dtype=torch.float64) for _ in range(2))
    mass = torch.tensor([1, 3, 4])
    expanded_k = k.repeat_interleave(mass, dim=1)
    expanded_v = v.repeat_interleave(mass, dim=1)
    expected = dense_flash(q, expanded_k, expanded_v)
    actual = mass_attention(dense_flash, q, k, v, mass.double().log(), causal=False, dropout_p=0.)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(dense_flash(q, k, v), expected)


def test_mass_feature_does_not_change_original_scale_or_input():
    q, k, v = (torch.randn(1, 4, 2, 128) for _ in range(3))
    copies = [t.clone() for t in (q, k, v)]
    scales = []
    def observe(a, b, c, **kwargs):
        assert a.shape[-1] == 136
        scales.append(kwargs['softmax_scale'])
        return dense_flash(a, b, c, **kwargs)
    actual = mass_attention(observe, q, k, v, torch.zeros(4))
    assert scales == [1 / math.sqrt(128)]
    assert torch.allclose(actual, dense_flash(q, k, v), atol=1e-6, rtol=1e-6)
    assert all(torch.equal(a, b) for a, b in zip((q, k, v), copies))


def test_repeated_proxy_keeps_original_head_dimension_and_pays_logical_keys():
    torch.manual_seed(81)
    q, k, v = (torch.randn(1, 3, 2, 128, dtype=torch.float64) for _ in range(3))
    masses = torch.tensor([1, 2, 4])
    index = torch.arange(3).repeat_interleave(masses)
    observed = []
    def observe(a, b, c, **kwargs):
        observed.append((a.shape[-1], b.shape[1]))
        return dense_flash(a, b, c, **kwargs)
    expanded = repeated_attention(observe, q, k, v, index)
    augmented = mass_attention(dense_flash, q, k, v, masses.double().log())
    assert observed == [(128, 7)]
    assert torch.allclose(expanded, augmented, atol=1e-12, rtol=1e-12)


def test_flash_hook_restores_after_exception_and_rejects_fallback():
    def broken(*args, **kwargs):
        raise RuntimeError('expected failure')
    block = SimpleNamespace(flash_attn_func=broken)
    with pytest.raises(RuntimeError, match='expected failure'):
        with proportional_flash(block, torch.zeros(2)):
            block.flash_attn_func(*(torch.ones(1, 2, 1, 8) for _ in range(3)))
    assert block.flash_attn_func is broken
    with pytest.raises(RuntimeError, match='FlashAttention'):
        with proportional_flash(SimpleNamespace(flash_attn_func=None), torch.zeros(2)):
            pass


class TinyRotary(nn.Module):
    config = SimpleNamespace(rope_full_precision=True)
    def get_rotary_embedding(self, length, device):
        phase = torch.arange(length, device=device).float()[:, None] * torch.tensor([.17, .31, .17, .31], device=device)
        return phase.sin()[None, None], phase.cos()[None, None]

    @staticmethod
    def apply_rotary_pos_emb(sin, cos, x):
        half = x.shape[-1] // 2
        return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin

    def forward(self, q, k):
        sin, cos = self.get_rotary_embedding(k.shape[-2], q.device)
        return (self.apply_rotary_pos_emb(sin[:, :, -q.shape[-2]:], cos[:, :, -q.shape[-2]:], q),
                self.apply_rotary_pos_emb(sin, cos, k))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 24, bias=False)
        self.out = nn.Linear(8, 8, bias=False)
        self.rotary_emb = TinyRotary()
        self.flash_attn_func = dense_flash

    def _scaled_dot_product_attention(self, q, k, v, **kwargs):
        return self.flash_attn_func(*(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)

    def forward(self, hidden, attention_bias=None, layer_past=None, use_cache=False):
        q, k, v = (t.reshape(1, -1, 2, 4).transpose(1, 2) for t in self.projection(hidden).chunk(3, -1))
        if layer_past is not None:
            k, v = (torch.cat((a, b), -2) for a, b in zip(layer_past, (k, v)))
        present = (k, v) if use_cache else None
        q, kr = self.rotary_emb(q, k)
        att = self._scaled_dot_product_attention(q, kr, v).transpose(1, 2).reshape(1, -1, 8)
        return hidden + self.out(att), present


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        core = nn.Module()
        core.config = SimpleNamespace(n_layers=3, d_model=8, input_emb_norm=False,
                                      weight_tying=False, scale_logits=False)
        tr = nn.Module()
        tr.wte = nn.Embedding(MASK_ID + 1, 8)
        tr.emb_drop = nn.Identity()
        tr.blocks = nn.ModuleList([TinyBlock() for _ in range(3)])
        tr.ln_f, tr.ff_out = nn.LayerNorm(8), nn.Linear(8, 17, bias=False)
        core.transformer = tr
        self.model = core

    def forward(self, ids, past_key_values=None, use_cache=False):
        h, caches = self.model.transformer.wte(ids), []
        for i, b in enumerate(self.model.transformer.blocks):
            h, p = b(h, layer_past=None if past_key_values is None else past_key_values[i], use_cache=use_cache)
            caches.append(p)
        tr = self.model.transformer
        return SimpleNamespace(logits=tr.ff_out(tr.ln_f(h)), past_key_values=caches)


@pytest.mark.parametrize('cached', [False, True])
def test_full_and_singleton_paths_preserve_positions_cache_and_logits(cached):
    torch.manual_seed(33)
    model = TinyModel()
    ids = torch.tensor([[MASK_ID, 1, MASK_ID, 2, MASK_ID, MASK_ID, MASK_ID, MASK_ID]])
    past = [tuple(torch.randn(1, 2, 3, 4) for _ in range(2)) for _ in range(3)] if cached else None
    copies = [t.clone() for pair in (past or []) for t in pair]
    targets = [0, 2]
    reference = model(ids, past_key_values=past).logits[:, targets]
    for config in [ProxyConfig(layer=1, block_length=4, keep_ratio=1),
                   ProxyConfig(layer=1, block_length=4, exact_fraction=1),
                   ProxyConfig(layer=1, block_length=4, keep_ratio=1, force_mass_kernel=True)]:
        result = FocusV2Forward(model, config)(ids, targets, past_key_values=past)
        assert torch.allclose(result, reference, atol=1e-6, rtol=1e-6)
    assert all(torch.equal(t, copy) for t, copy in zip([t for pair in (past or []) for t in pair], copies))


def test_current_mask_coverage_and_outside_block_targets_rejected():
    model = TinyModel()
    forward = FocusV2Forward(model, ProxyConfig(layer=1, block_length=4))
    ids = torch.tensor([[MASK_ID] * 8])
    for targets in ([0, 1], [0, 1, 2, 3, 4], [0, 0, 1, 2, 3]):
        with pytest.raises(ValueError):
            forward(ids, targets)


def test_weighted_forward_restores_rotary_and_flash_on_failure():
    model = TinyModel()
    block = model.model.transformer.blocks[1]
    rotary, flash = block.rotary_emb, block.flash_attn_func
    def broken(*args, **kwargs):
        raise RuntimeError('deep failed')
    block.flash_attn_func = broken
    forward = FocusV2Forward(model, ProxyConfig(layer=1, block_length=4))
    with pytest.raises(RuntimeError, match='deep failed'):
        forward(torch.tensor([[MASK_ID] * 12]), [0, 1, 2, 3])
    assert block.rotary_emb is rotary and block.flash_attn_func is broken
    block.flash_attn_func = flash
