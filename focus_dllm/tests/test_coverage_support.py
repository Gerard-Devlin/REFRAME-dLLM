import torch

from focus_dllm.llada_common import MASK_ID
from focus_dllm.llada_pruning import Config
from focus_dllm.tests.test_tuning import tiny_model
from focus_dllm.tuning.coverage_support import CoverageSupportForward, choose_support
from focus_dllm.tuning.static_support import StaticSupportForward


def test_coverage_policies_cover_each_region_even_with_concentrated_scores():
    scores = torch.arange(224, 0, -1, dtype=torch.float)
    top = choose_support(scores, 32, 'attention')
    assert top.max() < 32
    for policy in ('uniform', 'stratified'):
        selected = choose_support(scores, 32, policy)
        assert torch.equal(selected * 32 // 224, torch.arange(32))
        assert len(selected.unique()) == 32
    # Ragged regions also select within their own, nonoverlapping intervals.
    scores = torch.arange(11, 0, -1, dtype=torch.float)
    selected = choose_support(scores, 4, 'stratified')
    assert selected.tolist() == [0, 2, 5, 8]
    for policy in ('attention', 'uniform', 'stratified'):
        assert choose_support(scores, 0, policy).numel() == 0
        assert choose_support(scores, 11, policy).sort().values.tolist() == list(range(11))


@torch.no_grad()
def test_attention_policy_matches_existing_live_support_and_cache_read_only():
    torch.set_num_threads(1)
    model = tiny_model()
    ids = torch.full((1, 64), MASK_ID)
    prefix = torch.tensor([[10, 21, 31]])
    reference = model(torch.cat((prefix, ids), 1), use_cache=True).past_key_values
    before = [tuple(t.clone() for t in pair) for pair in reference]
    past = [tuple(t[:, :, :3] for t in pair) for pair in reference]
    config = Config(prune_after_layer=1, support_keep_ratio=0., target_only_head=True)
    original = StaticSupportForward(model, config, support_count=16)
    updated = CoverageSupportForward(model, config, support_count=16, policy='attention')
    original.reference = updated.reference = reference
    a, b = original(ids, [2, 7, 15], past_key_values=past), updated(ids, [2, 7, 15], past_key_values=past)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(original.kept, updated.kept)
    layout = updated.support_past
    ids[:, 2] = 8
    a, b = original(ids, [7, 15], past_key_values=past), updated(ids, [7, 15], past_key_values=past)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert updated.support_past is layout
    for a, b in zip(before, reference):
        assert all(torch.equal(u, v) for u, v in zip(a, b))


@torch.no_grad()
def test_full_live_future_budget_makes_selection_policy_irrelevant():
    torch.set_num_threads(1)
    model = tiny_model()
    ids = torch.full((1, 64), MASK_ID)
    reference = model(torch.cat((torch.tensor([[10, 21, 31]]), ids), 1), use_cache=True).past_key_values
    past = [tuple(t[:, :, :3] for t in pair) for pair in reference]
    outputs = []
    for policy in ('attention', 'uniform', 'stratified'):
        forward = CoverageSupportForward(model, Config(prune_after_layer=1,
            support_keep_ratio=0., target_only_head=True), support_count=32, policy=policy)
        forward.reference = reference
        outputs.append(forward(ids, [2, 7, 15], past_key_values=past))
        assert torch.equal(forward.kept, torch.arange(64))
    for value in outputs[1:]:
        torch.testing.assert_close(value, outputs[0], rtol=0, atol=0)
