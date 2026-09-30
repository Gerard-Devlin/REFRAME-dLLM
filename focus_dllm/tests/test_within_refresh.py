import torch

from focus_dllm.llada_common import MASK_ID
from focus_dllm.llada_pruning import Config
from focus_dllm.tests.test_tuning import tiny_model
from focus_dllm.tuning.backend import generate_active_prefix, selected_forward
from focus_dllm.tuning.within_refresh import RefreshingSupport, generate_refresh


@torch.no_grad()
def test_refresh_every_call_matches_exact_prefix_actions_without_extra_forwards():
    torch.set_num_threads(1)
    model = tiny_model()
    model.model.transformer.ff_out.weight[MASK_ID].zero_()
    prompt = torch.tensor([[10, 21, 31]])
    expected, actions = generate_active_prefix(model, prompt, gen_length=64,
        layer=1, keep=1., pruning=False, trace=True)
    actual, refreshed_actions, counts = generate_refresh(model, prompt, gen_length=64,
        layer=1, keep=0., period=1, trace=True)
    assert torch.equal(actual.output, expected.output)
    assert actual.nfe == expected.nfe and actions == refreshed_actions
    assert counts['focused_calls'] == 0
    assert counts['within_refresh_calls'] + counts['tail_calls'] + 2 == actual.nfe


@torch.no_grad()
def test_stall_refresh_uses_current_state_and_preserves_clean_prefix():
    torch.set_num_threads(1)
    model = tiny_model()
    ids = torch.full((1, 64), MASK_ID)
    prefix = torch.tensor([[10, 21, 31]])
    reference = model(torch.cat((prefix, ids), 1), use_cache=True).past_key_values
    past = [tuple(t[:, :, :3] for t in pair) for pair in reference]
    before = [tuple(t.clone() for t in pair) for pair in past]
    forward = RefreshingSupport(model, Config(prune_after_layer=1,
        support_keep_ratio=0., target_only_head=True), support_count=16, stall_limit=2)
    forward.reference = reference
    forward(ids, [2, 7, 15], past_key_values=past)
    ids[:, 2] = 9
    forward(ids, [7, 15], past_key_values=past)
    ids[:, 7] = 10
    expected = selected_forward(model, ids, torch.tensor([15]),
                                past_key_values=past, use_cache=True).logits
    actual = forward(ids, [15], past_key_values=past)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert forward.refresh_calls == 1 and forward.focused_calls == 2
    assert forward.reference is not reference
    assert forward.reference is forward.refresh_source
    for a, b in zip(before, past):
        assert all(torch.equal(u, v) for u, v in zip(a, b))
    # A fresh block snapshot resets only the per-block policy state.
    forward.reference = reference
    forward(ids, [15], past_key_values=past)
    assert forward.block_calls == 1 and forward.stalled == 0
