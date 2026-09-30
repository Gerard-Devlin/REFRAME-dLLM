import pytest
import torch

from focus_dllm.llada_common import MASK_ID
from focus_dllm.llada_pruning import Config
from focus_dllm.tests.test_tuning import tiny_model
from focus_dllm.tuning.boundary_kv import FreshBoundaryForward
from focus_dllm.tuning.static_support import StaticSupportForward


@torch.no_grad()
def test_boundary_raw_kv_matches_fresh_full_state_with_original_positions():
    torch.set_num_threads(1)
    model = tiny_model()
    ids = torch.full((1, 64), MASK_ID)
    reference = model(torch.cat((torch.tensor([[10, 21, 31]]), ids), 1), use_cache=True).past_key_values
    before = [tuple(t.clone() for t in pair) for pair in reference]
    past = [tuple(t[:, :, :3] for t in pair) for pair in reference]
    ids[:, 2] = 18
    full = model(ids, past_key_values=past, use_cache=True)
    forward = FreshBoundaryForward(model, Config(prune_after_layer=1,
        support_keep_ratio=0., target_only_head=True), support_count=16)
    forward.reference = reference
    forward(ids, [7, 15], past_key_values=past)
    rotary = forward.support_rotaries[1]
    frozen = rotary.key_positions[:-len(forward.kept)]
    expected = tuple(t.index_select(-2, frozen) for t in full.past_key_values[1])
    for a, b in zip(expected, forward.support_past[1]):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-7)
    key = full.past_key_values[1][0]
    _, rotated = model.model.transformer.blocks[1].rotary_emb(key[:, :, :1], key)
    torch.testing.assert_close(rotary.frozen_rotated_key, rotated.index_select(-2, frozen),
                               rtol=2e-5, atol=1e-7)
    for a, b in zip(before, reference):
        assert all(torch.equal(u, v) for u, v in zip(a, b))
    assert not hasattr(forward, 'cut_hidden')


@torch.no_grad()
def test_only_last_layer_pruned_has_full_attention_semantics_after_fresh_kv():
    torch.set_num_threads(1)
    model = tiny_model()
    ids = torch.full((1, 64), MASK_ID)
    reference = model(torch.cat((torch.tensor([[10, 21, 31]]), ids), 1), use_cache=True).past_key_values
    past = [tuple(t[:, :, :3] for t in pair) for pair in reference]
    ids[:, 2] = 18
    full = model(ids, past_key_values=past, use_cache=False).logits[:, [7, 15]]
    forward = FreshBoundaryForward(model, Config(prune_after_layer=2,
        support_keep_ratio=0., target_only_head=True), support_count=16)
    forward.reference = reference
    updated = forward(ids, [7, 15], past_key_values=past)
    torch.testing.assert_close(updated, full, rtol=2e-5, atol=1e-7)


@torch.no_grad()
def test_boundary_hooks_and_attention_restore_on_failure(monkeypatch):
    torch.set_num_threads(1)
    model = tiny_model()
    ids = torch.full((1, 64), MASK_ID)
    reference = model(torch.cat((torch.tensor([[10, 21, 31]]), ids), 1), use_cache=True).past_key_values
    past = [tuple(t[:, :, :3] for t in pair) for pair in reference]
    forward = FreshBoundaryForward(model, Config(prune_after_layer=1,
        support_keep_ratio=0., target_only_head=True), support_count=16)
    forward.reference = reference
    original = [block.attention for block in model.model.transformer.blocks]
    def fail(*args, **kwargs):
        raise RuntimeError('simulated failure')
    monkeypatch.setattr(StaticSupportForward, 'body', fail)
    with pytest.raises(RuntimeError, match='simulated failure'):
        forward(ids, [7, 15], past_key_values=past)
    for block, attention in zip(model.model.transformer.blocks, original):
        assert not block._forward_hooks and not block._forward_pre_hooks
        assert block.attention == attention
