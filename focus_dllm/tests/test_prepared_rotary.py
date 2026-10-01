import pytest
import torch

from focus_dllm.llada_common import MASK_ID
from focus_dllm.tuning.focus_v2 import FocusV2Forward, ProxyConfig
from focus_dllm.tuning.prepared_rotary import RotatedPrefixCache
from focus_dllm.tests.test_focus_v2 import TinyModel, TinyRotary


def test_native_cache_reuse_exactly_matches_original_rotation():
    torch.manual_seed(35)
    base = TinyRotary()
    cache = RotatedPrefixCache()
    past = torch.randn(1, 2, 7, 4)
    copies = past.clone()
    for _ in range(3):
        q, suffix = (torch.randn(1, 2, 5, 4) for _ in range(2))
        k = torch.cat((past, suffix), -2)
        reference = base(q, k)
        prepared = cache.positioned(base, None, 5, past)
        actual = prepared(q, k)
        assert all(torch.equal(a, b) for a, b in zip(reference, actual))
    assert cache.cache_prepares == 1 and torch.equal(past, copies)


def test_noncontiguous_position_phases_and_formal_prefix_preserved():
    from focus_dllm.llada_pruning import PositionedRotary
    base, cache = TinyRotary(), RotatedPrefixCache()
    past = torch.randn(1, 2, 7, 4)
    positions = torch.tensor([0, 1, 4, 7])
    q, current = (torch.randn(1, 2, 4, 4) for _ in range(2))
    k = torch.cat((past, current), -2)
    expected = PositionedRotary(base, positions, 8, 7)(q, k)
    actual = cache.positioned(base, positions, 8, past)(q, k)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


def test_modified_or_replaced_prefix_invalidates_rotated_cache():
    base, cache = TinyRotary(), RotatedPrefixCache()
    past = torch.randn(1, 2, 2, 4)
    first = cache.positioned(base, None, 3, past).prefix.rotated.clone()
    past.add_(1.)
    second = cache.positioned(base, None, 3, past).prefix.rotated
    assert not torch.equal(first, second)
    cache.positioned(base, None, 3, past.clone())
    assert cache.cache_prepares == 3


def test_full_forward_and_proxy_logits_exact_with_shared_engine():
    model, cache = TinyModel(), RotatedPrefixCache()
    ids = torch.tensor([[MASK_ID, 1, MASK_ID, 2, MASK_ID, MASK_ID, MASK_ID, MASK_ID]])
    past = [tuple(torch.randn(1, 2, 3, 4) for _ in range(2)) for _ in range(3)]
    originals = [b.rotary_emb for b in model.model.transformer.blocks]
    native = model(ids, past_key_values=past).logits
    config = ProxyConfig(layer=1, block_length=4, mass_implementation='repeat')
    reference = FocusV2Forward(model, config)(ids, [0, 2], past_key_values=past)
    optimized = FocusV2Forward(model, config, rotary_factory=cache)
    with cache.suffix_scope(model, past, 8):
        other = model(ids, past_key_values=past).logits
        result = optimized(ids, [0, 2], past_key_values=past)
    assert torch.equal(other, native) and torch.equal(reference, result)
    assert all(b.rotary_emb is r for b, r in zip(model.model.transformer.blocks, originals))


def test_scope_restores_all_layers_after_failure():
    model, cache = TinyModel(), RotatedPrefixCache()
    past = [tuple(torch.randn(1, 2, 3, 4) for _ in range(2)) for _ in range(3)]
    originals = [b.rotary_emb for b in model.model.transformer.blocks]
    with pytest.raises(RuntimeError, match='expected'):
        with cache.suffix_scope(model, past, 8):
            raise RuntimeError('expected')
    assert not cache.saved and not cache.geometry
    assert all(b.rotary_emb is r for b, r in zip(model.model.transformer.blocks, originals))
