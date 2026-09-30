import pytest
import torch

from focus_dllm.llada_common import MASK_ID
from focus_dllm.llada_pruning import Config
from focus_dllm.tests.test_tuning import tiny_model
from focus_dllm.tuning.backend import selected_forward, generate_active_prefix
from focus_dllm.tuning.ffn_focus import FFNFocusForward, capture_ffn, generate_ffn_focus


def warm(model, ids, after):
    with capture_ffn(model, after) as values:
        result = model(torch.cat((torch.tensor([[10,21,31]]),ids),1), use_cache=True)
    past = [tuple(t[:, :, :3] for t in pair) for pair in result.past_key_values]
    return values, past


@torch.no_grad()
def test_future_ffn_only_current_attention_and_prefix_preserved():
    torch.set_num_threads(1)
    model = tiny_model()
    ids = torch.full((1,64), MASK_ID)
    values, past = warm(model,ids,1)
    cached_before = [None if t is None else t.clone() for t in values]
    past_before = [tuple(t.clone() for t in pair) for pair in past]
    ids[:,2] = 18
    forward = FFNFocusForward(model,Config(prune_after_layer=1))
    forward.install_reference(values,3)
    shapes = {}
    handles = []
    for i, block in enumerate(model.model.transformer.blocks):
        handles.append(block.q_proj.register_forward_pre_hook(
            lambda _m,args,i=i: shapes.__setitem__((i,'q'),args[0].shape[1])))
        handles.append(block.ff_proj.register_forward_pre_hook(
            lambda _m,args,i=i: shapes.__setitem__((i,'ff'),args[0].shape[1])))
    try:
        logits = forward(ids,[7,15],past_key_values=past)
    finally:
        for h in handles: h.remove()
    assert logits.shape == (1,2,126464)
    assert shapes == {(0,'q'):64,(0,'ff'):64,(1,'q'):64,(1,'ff'):32,(2,'q'):64,(2,'ff'):32}
    assert all(a is None or torch.equal(a,b) for a,b in zip(cached_before,values))
    assert all(torch.equal(a,b) for old,new in zip(past_before,past) for a,b in zip(old,new))
    assert not any(b.ff_norm._forward_hooks or b.ff_out._forward_hooks for b in model.model.transformer.blocks)


@torch.no_grad()
def test_last_layer_future_ffn_cannot_affect_current_outputs():
    torch.set_num_threads(1)
    model = tiny_model()
    ids = torch.full((1,64),MASK_ID)
    values,past = warm(model,ids,2)
    ids[:,2] = 18
    forward = FFNFocusForward(model,Config(prune_after_layer=2))
    forward.install_reference(values,3)
    expected = selected_forward(model,ids,torch.tensor([7,15]),past_key_values=past,use_cache=False).logits
    torch.testing.assert_close(forward(ids,[7,15],past_key_values=past),expected,rtol=2e-5,atol=1e-7)


def test_hooks_restore_on_failure_and_non_mask_future_rejected(monkeypatch):
    model = tiny_model()
    ids = torch.full((1,64),MASK_ID)
    values,past = warm(model,ids,1)
    forward = FFNFocusForward(model,Config(prune_after_layer=1))
    forward.install_reference(values,3)
    ids[:,33] = 42
    with pytest.raises(ValueError,match='untouched'): forward(ids,[7],past_key_values=past)
    ids[:,33] = MASK_ID
    def fail(*args,**kwargs): raise RuntimeError('simulated')
    monkeypatch.setattr(model.model.transformer.blocks[2].ff_proj,'forward',fail)
    with pytest.raises(RuntimeError,match='simulated'): forward(ids,[7],past_key_values=past)
    assert not any(b.ff_norm._forward_hooks or b.ff_out._forward_hooks for b in model.model.transformer.blocks)
    assert not model.model.transformer.ln_f._forward_pre_hooks


@torch.no_grad()
def test_disabled_reuse_preserves_actions_and_generation_and_scoped_patches():
    torch.set_num_threads(1)
    from focus_dllm.tuning import backend
    model = tiny_model()
    model.model.transformer.ff_out.weight[MASK_ID].zero_()
    prompt = torch.tensor([[10,21,31]])
    reference,actions = generate_active_prefix(model,prompt,gen_length=64,layer=2,keep=1.,pruning=False,trace=True)
    original = backend.ActiveForward,backend.selected_forward
    actual,new_actions = generate_ffn_focus(model,prompt,gen_length=64,layer=3,keep=0.,pruning=True,trace=True)
    assert torch.equal(reference.output,actual.output) and reference.nfe == actual.nfe
    assert actions == new_actions
    assert (backend.ActiveForward,backend.selected_forward) == original


def test_capture_hooks_removed_on_failure():
    model = tiny_model()
    with pytest.raises(RuntimeError,match='simulated'):
        with capture_ffn(model,1):
            raise RuntimeError('simulated')
    assert not any(b.ff_out._forward_hooks for b in model.model.transformer.blocks)
