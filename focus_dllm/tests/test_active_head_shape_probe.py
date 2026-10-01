from types import SimpleNamespace

import pytest
import torch

from focus_dllm.tuning.active_head_shape_probe import project


@pytest.mark.parametrize('tied',[False,True])
@pytest.mark.parametrize('scaled',[False,True])
def test_isolated_head_obeys_weight_tying_and_native_scaling(tied,scaled):
    weight=torch.tensor([[1.,2.],[-3.,4.],[5.,6.]])
    normed=torch.tensor([[[.7,-.3],[.2,.9]]])
    head=torch.nn.Linear(2,3,bias=False)
    with torch.no_grad():head.weight.copy_(weight)
    model=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
        d_model=2,scale_logits=scaled,weight_tying=tied),transformer=SimpleNamespace(
        wte=SimpleNamespace(weight=weight),ff_out=head)))
    expected=normed @ weight.T
    if scaled:expected=expected/(2**.5)
    assert torch.allclose(project(model,normed),expected)
