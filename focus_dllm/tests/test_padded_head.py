import pytest
import torch

from focus_dllm.tests.test_focus_v2 import TinyModel
from focus_dllm.tuning.padded_head import selected_forward,project,pad_rows


@pytest.mark.parametrize('cached',[False,True])
def test_padded_head_preserves_full_normalization_and_formal_cache(cached):
    torch.manual_seed(11)
    model=TinyModel()
    ids=torch.tensor([[1,2,3,4,5]])
    past=[tuple(torch.randn(1,2,3,4) for _ in range(2)) for _ in range(3)] if cached else None
    copies=[t.clone() for pair in (past or []) for t in pair]
    expected=model(ids,past_key_values=past,use_cache=True)
    rows=[]
    hook=model.model.transformer.ff_out.register_forward_pre_hook(lambda _m,args:rows.append(args[0].shape[1]))
    try:
        output=selected_forward(model,ids,[1,3],past_key_values=past,use_cache=True)
    finally:
        hook.remove()
    assert rows==[32] and output.logits.shape[1]==2
    assert torch.allclose(output.logits,expected.logits[:,[1,3]],atol=1e-7)
    assert all(torch.equal(t,c) for t,c in zip([t for pair in (past or []) for t in pair],copies))
    assert all(torch.equal(t,r) for pair,reference in zip(output.past_key_values,expected.past_key_values)
               for t,r in zip(pair,reference))


def test_exception_removes_temporary_head_hook():
    model=TinyModel()
    initial=len(model.model.transformer.ln_f._forward_hooks)
    def broken(*args,**kwargs):raise RuntimeError('failed head')
    model.model.transformer.ff_out.forward=broken
    with pytest.raises(RuntimeError,match='failed head'):
        selected_forward(model,torch.tensor([[1,2]]),[0])
    assert len(model.model.transformer.ln_f._forward_hooks)==initial


def test_padding_never_discards_rows_or_changes_hidden_values():
    hidden=torch.randn(1,40,8)
    assert torch.equal(pad_rows(hidden,32),hidden)
    short=hidden[:,:3]
    padded=pad_rows(short,32)
    assert torch.equal(padded[:,:3],short) and (padded[:,3:]==0).all()
    with pytest.raises(ValueError):pad_rows(short,-1)
