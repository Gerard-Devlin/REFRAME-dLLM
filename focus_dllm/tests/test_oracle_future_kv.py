import pytest
import torch

from focus_dllm.llada_common import MASK_ID
from focus_dllm.llada_pruning import Config
from focus_dllm.tests.test_tuning import tiny_model
from focus_dllm.tuning.backend import selected_forward
from focus_dllm.tuning.oracle_future_kv import fresh_frozen_future,bf16_reduction
from focus_dllm.tuning.static_support import StaticSupportForward


@torch.no_grad()
def fixture(support=16):
    torch.set_num_threads(1)
    model=tiny_model()
    prefix=torch.tensor([[10,21,31]])
    x=torch.full((1,64),MASK_ID)
    warm=selected_forward(model,torch.cat((prefix,x),1),torch.arange(3,35),use_cache=True)
    past=[tuple(t[:,:,:3] for t in pair) for pair in warm.past_key_values]
    x[0,:32]=torch.arange(32)+10;x[0,[2,7,15]]=MASK_ID
    target=torch.tensor([2,7,15])
    cached=selected_forward(model,x,target,past_key_values=past,use_cache=True)
    ordinary=selected_forward(model,x,target,past_key_values=past,use_cache=False)
    torch.testing.assert_close(cached.logits,ordinary.logits,rtol=0,atol=0)
    forward=StaticSupportForward(model,Config(prune_after_layer=1,support_keep_ratio=0.,target_only_head=True),support_count=support)
    forward.reference=warm.past_key_values
    old=forward(x,target,past_key_values=past)
    return model,x,target,past,cached,forward,old


@torch.no_grad()
def test_oracle_matches_full_fp32_state_without_changing_selection_or_prefix():
    model,x,target,past,cached,f,old=fixture()
    before=f.kept.clone();cache=f.support_past;rotaries=f.support_rotaries
    versions=[t._version for pair in f.reference+cached.past_key_values for t in pair]
    with fresh_frozen_future(f,cached.past_key_values,3):
        actual=f(x,target,past_key_values=past)
        for original,replacement in zip(cache,f.support_past):
            assert all(torch.equal(a[:,:,:3],b[:,:,:3]) for a,b in zip(original,replacement))
    torch.testing.assert_close(actual,cached.logits,rtol=3e-5,atol=3e-7)
    assert torch.equal(f.kept,before) and f.support_past is cache and f.support_rotaries is rotaries
    assert versions==[t._version for pair in f.reference+cached.past_key_values for t in pair]
    torch.testing.assert_close(f(x,target,past_key_values=past),old,rtol=0,atol=0)


@torch.no_grad()
def test_context_restores_layout_and_attention_after_failure():
    model,x,target,past,cached,f,old=fixture()
    cache=f.support_past;rotaries=f.support_rotaries
    attention=[block.attention for block in model.model.transformer.blocks]
    rope=[block.rotary_emb for block in model.model.transformer.blocks]
    with pytest.raises(RuntimeError):
        with fresh_frozen_future(f,cached.past_key_values,3):
            f(x,target,past_key_values=past)
            raise RuntimeError('expected')
    assert f.support_past is cache and f.support_rotaries is rotaries
    assert attention==[block.attention for block in model.model.transformer.blocks]
    assert rope==[block.rotary_emb for block in model.model.transformer.blocks]


@torch.no_grad()
def test_rejects_modified_formal_prefix_and_misaligned_canvas():
    _,_,_,_,cached,f,_=fixture()
    cache=f.support_past
    bad=[tuple(t.clone() for t in pair) for pair in cached.past_key_values]
    bad[0][0][:,:,:3].add_(1)
    with pytest.raises(ValueError):
        with fresh_frozen_future(f,bad,3):pass
    with pytest.raises(ValueError):
        with fresh_frozen_future(f,[tuple(t[:,:,:-1] for t in pair) for pair in cached.past_key_values],3):pass
    assert f.support_past is cache


@torch.no_grad()
def test_full_support_budget_needs_no_oracle_replacement():
    _,x,target,past,cached,f,old=fixture(support=32)
    cache=f.support_past
    with fresh_frozen_future(f,cached.past_key_values,3):
        assert f.support_past is cache
        torch.testing.assert_close(f(x,target,past_key_values=past),old,rtol=0,atol=0)


def test_precision_control_restores_on_exception():
    previous=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    with pytest.raises(RuntimeError):
        with bf16_reduction(not previous):
            assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is not previous
            raise RuntimeError('expected')
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is previous
