from types import SimpleNamespace

import pytest
import torch
from torch import nn

from focus_dllm.tuning.drift_support import capture_cut_hidden,support_priority


def test_attention_mirror_preserves_scores():
    score=torch.tensor([.8,.2])
    hidden=torch.randn(2,4)
    assert support_priority(score,hidden,hidden,'attention') is score


def test_relative_drift_distinguishes_change_from_scale():
    reference=torch.tensor([[1.,1.],[10.,10.]])
    current=torch.tensor([[2.,2.],[11.,11.]])
    score=torch.tensor([.1,.9])
    assert torch.allclose(support_priority(score,current,reference,'drift'),torch.tensor([1.,.1]))
    assert torch.allclose(support_priority(score,current,reference,'attention_drift'),torch.tensor([.1,.09]))


def test_no_change_is_finite_and_zero():
    hidden=torch.zeros(3,4)
    result=support_priority(torch.ones(3),hidden,hidden,'attention_drift')
    assert torch.isfinite(result).all() and not result.any()


def test_alignment_and_policy_validation():
    with pytest.raises(ValueError):support_priority(torch.ones(2),torch.ones(3,4),torch.ones(3,4),'drift')
    with pytest.raises(ValueError):support_priority(torch.ones(2),torch.ones(2,4),torch.ones(2,4),'unknown')


def test_capture_detaches_clones_and_removes_hook_on_exception():
    class Block(nn.Module):
        def forward(self,x):return x*2,None
    block=Block()
    model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=[block])))
    x=torch.ones(1,3,4,requires_grad=True)
    with pytest.raises(RuntimeError):
        with capture_cut_hidden(model,1) as captured:
            output,_=block(x)
            output.detach().add_(3)
            raise RuntimeError('expected')
    assert len(captured)==1 and not captured[0].requires_grad
    assert torch.equal(captured[0],torch.full_like(x,2.))
    assert len(block._forward_hooks)==0


@torch.no_grad()
def test_attention_mirror_and_reference_immutability():
    from focus_dllm.llada_common import MASK_ID
    from focus_dllm.llada_pruning import Config
    from focus_dllm.tests.test_tuning import tiny_model
    from focus_dllm.tuning.static_support import StaticSupportForward
    from focus_dllm.tuning.drift_support import DriftSupportForward
    torch.set_num_threads(1)
    model=tiny_model()
    ids=torch.full((1,64),MASK_ID)
    prefix=torch.tensor([[10,21,31]])
    with capture_cut_hidden(model,1) as cut:
        reference=model(torch.cat((prefix,ids),1),use_cache=True).past_key_values
    versions=[t._version for pair in reference for t in pair]
    cut_version=cut[0]._version
    past=[tuple(t[:,:,:3] for t in pair) for pair in reference]
    config=Config(prune_after_layer=1,support_keep_ratio=0.,target_only_head=True)
    old=StaticSupportForward(model,config,support_count=16)
    new=DriftSupportForward(model,config,support_count=16,policy='attention')
    old.reference=new.reference=reference
    new.cut_reference=cut[0]
    a,b=old(ids,[2,7,15],past_key_values=past),new(ids,[2,7,15],past_key_values=past)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.equal(old.kept,new.kept)
    layout=new.support_past
    ids[:,2]=8
    a,b=old(ids,[7,15],past_key_values=past),new(ids,[7,15],past_key_values=past)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert new.support_past is layout
    assert versions==[t._version for pair in reference for t in pair]
    assert cut[0]._version==cut_version


@torch.no_grad()
def test_full_budget_policy_invariance_and_missing_reference_failure():
    from focus_dllm.llada_common import MASK_ID
    from focus_dllm.llada_pruning import Config
    from focus_dllm.tests.test_tuning import tiny_model
    from focus_dllm.tuning.drift_support import DriftSupportForward
    model=tiny_model()
    ids=torch.full((1,64),MASK_ID)
    with capture_cut_hidden(model,1) as cut:
        reference=model(torch.cat((torch.tensor([[10,21,31]]),ids),1),use_cache=True).past_key_values
    past=[tuple(t[:,:,:3] for t in pair) for pair in reference]
    outputs=[]
    config=Config(prune_after_layer=1,support_keep_ratio=0.,target_only_head=True)
    for policy in ('attention','drift','attention_drift'):
        forward=DriftSupportForward(model,config,support_count=32,policy=policy)
        forward.reference=reference;forward.cut_reference=cut[0]
        outputs.append(forward(ids,[2,7],past_key_values=past))
        assert torch.equal(forward.kept,torch.arange(64))
    for value in outputs[1:]:torch.testing.assert_close(value,outputs[0],rtol=0,atol=0)
    failing=DriftSupportForward(model,config,support_count=16)
    failing.reference=reference
    attention_before=[block.attention for block in model.model.transformer.blocks]
    rotary_before=[block.rotary_emb for block in model.model.transformer.blocks]
    with pytest.raises(ValueError):failing(ids,[2,7],past_key_values=past)
    assert attention_before==[block.attention for block in model.model.transformer.blocks]
    assert rotary_before==[block.rotary_emb for block in model.model.transformer.blocks]
