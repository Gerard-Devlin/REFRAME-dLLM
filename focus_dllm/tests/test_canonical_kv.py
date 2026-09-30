from types import SimpleNamespace

import pytest
import torch

from focus_dllm.llada_common import MASK_ID
from focus_dllm.llada_pruning import Config
from focus_dllm.tests.test_tuning import tiny_model
from focus_dllm.tuning.canonical_kv import CanonicalOrder, CanonicalZero, CanonicalSupport
from focus_dllm.tuning.zero_support import ZeroForward
from focus_dllm.tuning.static_support import StaticSupportForward


def test_attention_receives_original_token_order_and_unchanged_queries():
    seen = []

    def original(q, k, v, **kwargs):
        seen.append((q.clone(), k.clone(), v.clone()))
        return q

    block = SimpleNamespace(_scaled_dot_product_attention=original,
                            rotary_emb=SimpleNamespace(key_positions=torch.tensor([2, 0, 1])))

    class Caller:
        def __call__(self, q, k, v):
            return block._scaled_dot_product_attention(q, k, v)

    class Probe(CanonicalOrder, Caller):
        pass

    probe = Probe()
    probe.model = SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=[block])))
    q = torch.tensor([[[[5.]]]])
    k, v = torch.tensor([[[[30.], [10.], [20.]]]]), torch.tensor([[[[300.], [100.], [200.]]]])
    probe(q, k, v)
    assert torch.equal(seen[0][0], q)
    assert seen[0][1].flatten().tolist() == [10., 20., 30.]
    assert seen[0][2].flatten().tolist() == [100., 200., 300.]
    assert block._scaled_dot_product_attention is original


@torch.no_grad()
def test_canonical_order_preserves_values_positions_and_reference():
    torch.set_num_threads(1)
    model = tiny_model()
    ids = torch.full((1, 64), MASK_ID)
    ids[:, :32] = torch.arange(32) + 10
    ids[:, [2, 7, 15]] = MASK_ID
    reference = model(torch.cat((torch.tensor([[10, 21, 31]]), ids), 1),
                      use_cache=True).past_key_values
    past = [tuple(t[:, :, :3] for t in pair) for pair in reference]
    before = [tuple(t.clone() for t in pair) for pair in reference]
    config = Config(prune_after_layer=1, support_keep_ratio=0., target_only_head=True)
    for normal_type, canonical_type in ((ZeroForward, CanonicalZero),
                                       (StaticSupportForward, CanonicalSupport)):
        normal, canonical = normal_type(model, config), canonical_type(model, config)
        normal.reference = canonical.reference = reference
        methods = [b._scaled_dot_product_attention for b in model.model.transformer.blocks]
        a = normal(ids, [2, 7, 15], past_key_values=past)
        b = canonical(ids, [2, 7, 15], past_key_values=past)
        torch.testing.assert_close(a, b, rtol=3e-5, atol=1e-7)
        assert all(b._scaled_dot_product_attention == method for b, method in
                   zip(model.model.transformer.blocks, methods))
        for old, new in zip(before, reference):
            assert all(torch.equal(u, v) for u, v in zip(old, new))


def test_canonical_wrapper_restores_attention_on_failure():
    class Failure:
        def __call__(self, *args, **kwargs):
            raise RuntimeError('simulated failure')

    class Probe(CanonicalOrder, Failure):
        pass

    probe = Probe()
    probe.model = tiny_model()
    blocks = probe.model.model.transformer.blocks
    original = [b._scaled_dot_product_attention for b in blocks]
    with pytest.raises(RuntimeError, match='simulated failure'):
        probe(torch.tensor([[1, 2]]))
    assert all(b._scaled_dot_product_attention == method for b, method in zip(blocks, original))
