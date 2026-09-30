import torch

from focus_dllm.llada_common import MASK_ID
from focus_dllm.llada_decode import generate_prefix_cache
from focus_dllm.tuning.backend import selected_forward, active_output, generate_active_prefix


def tiny_model(tied=False):
    from v1.llada.model.configuration_llada import LLaDAConfig
    from v1.llada.model.modeling_llada import LLaDAModelLM
    torch.manual_seed(1234)
    cfg = LLaDAConfig(d_model=16, n_heads=2, n_kv_heads=2, n_layers=3,
                      mlp_hidden_size=32, vocab_size=126464, embedding_size=126464,
                      max_sequence_length=64, block_type='llama', rope=True,
                      activation_type='silu', layer_norm_type='rms', weight_tying=tied,
                      attention_dropout=0., residual_dropout=0., embedding_dropout=0.,
                      flash_attention=False)
    return LLaDAModelLM(cfg, init_params=True).eval()


@torch.no_grad()
def test_selected_head_preserves_logits_and_all_cached_values():
    torch.set_num_threads(1)
    for tied in (False, True):
        model = tiny_model(tied)
        ids = torch.tensor([[1, 5, MASK_ID, MASK_ID, 8, MASK_ID]])
        target = torch.tensor([2, 5])
        full = model(ids, use_cache=True)
        short = selected_forward(model, ids, target, use_cache=True)
        # GEMM selects a different kernel for two rows than for the full canvas.
        # Require tight FP32 agreement and identical choices, not bit equality.
        torch.testing.assert_close(full.logits[:, target], short.logits, rtol=2e-5, atol=1e-7)
        assert torch.equal(full.logits[:, target].argmax(-1), short.logits.argmax(-1))
        for p, q in zip(full.past_key_values, short.past_key_values):
            for a, b in zip(p, q):
                assert torch.equal(a, b)
        assert len(model.model.transformer.ln_f._forward_pre_hooks) == 0


def test_output_hook_is_removed_on_failure():
    model = tiny_model()
    try:
        with active_output(model, torch.tensor([0])):
            raise RuntimeError('simulated failure')
    except RuntimeError:
        pass
    assert not model.model.transformer.ln_f._forward_pre_hooks


@torch.no_grad()
def test_exact_active_prefix_preserves_generation_and_nfe():
    torch.set_num_threads(1)
    model = tiny_model()
    prompt = torch.tensor([[10, 21, 31]])
    original = generate_prefix_cache(model, prompt, gen_length=8, block_length=4, threshold=.90)
    updated, actions = generate_active_prefix(model, prompt, gen_length=8, block_length=4,
                         threshold=.90, layer=1, keep=1., pruning=False, trace=True)
    assert torch.equal(original.output, updated.output)
    assert original.nfe == updated.nfe == len(actions)


@torch.no_grad()
def test_tensor_selection_matches_original_pruning():
    from focus_dllm.llada_pruning import LLaDABlockForward, Config
    from focus_dllm.tuning.tensor_pruning import TensorForward, ReferenceForward
    torch.set_num_threads(1)
    model = tiny_model()
    prefix = torch.tensor([[10, 21, 31]])
    ids = torch.full((1, 64), MASK_ID)
    ids[0, :32] = torch.arange(32) + 10
    ids[0, [2, 7, 15]] = MASK_ID
    full = model(torch.cat((prefix, ids), 1), use_cache=True)
    past = [tuple(t[:, :, :3] for t in pair) for pair in full.past_key_values]
    config = Config(prune_after_layer=1, support_keep_ratio=.5, target_only_head=True)
    original = LLaDABlockForward(model, config)(ids, [2, 7, 15], past_key_values=past)
    vector = TensorForward(model, config)(ids, [2, 7, 15], past_key_values=past)
    torch.testing.assert_close(original, vector, rtol=2e-5, atol=1e-7)
    reference = ReferenceForward(model, config)
    reference.reference = full.past_key_values
    before = [tuple(t.clone() for t in pair) for pair in full.past_key_values]
    result = reference(ids, [2, 7, 15], past_key_values=past)
    assert result.shape == original.shape and torch.isfinite(result).all()
    for a, b in zip(before, full.past_key_values):
        assert all(torch.equal(u, v) for u, v in zip(a, b))
    from focus_dllm.tuning.rotation import RotatedForward
    rotated = RotatedForward(model, config)
    rotated.reference = full.past_key_values
    rotated_result = rotated(ids, [2, 7, 15], past_key_values=past)
    torch.testing.assert_close(result, rotated_result, rtol=2e-5, atol=1e-7)
    last = ids[:, :32]
    expected_last = selected_forward(model, last, torch.tensor([2, 7, 15]),
                                      past_key_values=past, use_cache=False).logits
    actual_last = rotated(last, [2, 7, 15], past_key_values=past)
    torch.testing.assert_close(expected_last, actual_last, rtol=0, atol=0)
    from focus_dllm.tuning.zero_support import ZeroForward
    zero_config = Config(prune_after_layer=1, support_keep_ratio=0., target_only_head=True)
    normal_zero = RotatedForward(model, zero_config)
    fast_zero = ZeroForward(model, zero_config)
    normal_zero.reference = fast_zero.reference = full.past_key_values
    a = normal_zero(ids, [2, 7, 15], past_key_values=past)
    b = fast_zero(ids, [2, 7, 15], past_key_values=past)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-7)


@torch.no_grad()
def test_refresh_each_block_preserves_zero_policy_and_actions():
    from focus_dllm.tuning.zero_support import generate_zero
    from focus_dllm.tuning.streaming import generate_stream
    torch.set_num_threads(1)
    model = tiny_model()
    model.model.transformer.ff_out.weight[MASK_ID].zero_()
    prompt = torch.tensor([[10,21,31]])
    old, old_actions = generate_zero(model,prompt,gen_length=64,layer=1,keep=0.,trace=True)
    new, new_actions, queries = generate_stream(model,prompt,gen_length=64,layer=1,refresh_every=1,trace=True)
    assert torch.equal(old.output,new.output) and old.nfe == new.nfe
    assert old_actions == new_actions and queries == 2*(64+3)
    partial, _, partial_queries = generate_stream(model,prompt,gen_length=64,layer=1,refresh_every=2)
    assert partial.output.shape == old.output.shape
    assert partial_queries == 67+64
    exact, exact_actions = generate_active_prefix(model,prompt,gen_length=64,layer=1,
                                  keep=1.,pruning=False,trace=True)
    full, full_actions, _ = generate_stream(model,prompt,gen_length=64,layer=1,
                                           keep=1.,refresh_every=1,trace=True)
    assert torch.equal(exact.output, full.output) and exact_actions == full_actions
    assert exact.nfe == full.nfe


@torch.no_grad()
def test_streaming_refresh_boundaries_and_recent_history(monkeypatch):
    from focus_dllm.tuning import streaming
    torch.set_num_threads(1)
    model = tiny_model()
    model.model.transformer.ff_out.weight[MASK_ID].zero_()
    calls = []
    original = streaming.selected_forward

    def observe(model, ids, target, **kwargs):
        past = kwargs.get('past_key_values')
        calls.append((ids.clone(), target.clone(), 0 if past is None else past[0][0].shape[-2]))
        return original(model, ids, target, **kwargs)

    monkeypatch.setattr(streaming, 'selected_forward', observe)
    prompt = torch.tensor([[10,21,31]])
    result, actions, queries = streaming.generate_stream(model,prompt,gen_length=128,
                                      layer=1,refresh_every=2,trace=True)
    assert [ids.shape[1] for ids, _, _ in calls] == [131,128,131,64]
    assert [past for _, _, past in calls] == [0,3,0,67]
    assert [target.min().item() for _, target, _ in calls] == [3,32,67,32]
    assert queries == 131+128+131+64 and result.nfe == len(actions)
    # Partial refresh recomputes the preceding *clean* block before the MASK tail.
    for block in (1,3):
        ids, _, _ = calls[block]
        torch.testing.assert_close(ids[:, :32], result.output[:, 3+(block-1)*32:3+block*32],rtol=0,atol=0)
        assert (ids[:,32:] == MASK_ID).all()
    assert torch.equal(result.output[:, :3], prompt)


@torch.no_grad()
def test_static_support_matches_first_dynamic_selection_and_reuses_layout():
    from focus_dllm.llada_pruning import Config
    from focus_dllm.tuning.rotation import RotatedForward
    from focus_dllm.tuning.static_support import StaticSupportForward
    torch.set_num_threads(1)
    model = tiny_model()
    ids = torch.full((1,64),MASK_ID)
    ids[:, :32] = torch.arange(32)+10
    ids[:, [2,7,15]] = MASK_ID
    reference = model(torch.cat((torch.tensor([[10,21,31]]),ids),1),use_cache=True).past_key_values
    past = [tuple(t[:,:,:3] for t in pair) for pair in reference]
    dynamic = RotatedForward(model,Config(prune_after_layer=1,support_keep_ratio=.5,target_only_head=True))
    static = StaticSupportForward(model,Config(prune_after_layer=1,support_keep_ratio=0.,target_only_head=True),16)
    dynamic.reference = static.reference = reference
    a,b = dynamic(ids,[2,7,15],past_key_values=past), static(ids,[2,7,15],past_key_values=past)
    torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-7)
    assert len(static.kept) == 48
    layout = static.support_past
    kept = static.kept.clone()
    ids[:,2] = 18
    c = static(ids,[7,15],past_key_values=past)
    assert c.shape[1] == 2 and torch.isfinite(c).all()
    assert static.support_past is layout and torch.equal(static.kept,kept)
