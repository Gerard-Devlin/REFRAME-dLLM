from types import SimpleNamespace

import pytest
import torch

from focus_dllm.tuning.gpu_contract import check_binding


def test_lost_tmux_binding_fails_before_cuda_initialization(monkeypatch):
    monkeypatch.setenv('FOCUS_RESEARCH_GPU_UUID','GPU-research')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:pytest.fail('CUDA accessed before env check'))
    with pytest.raises(RuntimeError,match='binding'):
        check_binding()


def test_wrong_visible_or_resolved_device_rejected(monkeypatch):
    monkeypatch.setenv('FOCUS_RESEARCH_GPU_UUID','GPU-research')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-other')
    with pytest.raises(RuntimeError,match='binding'):
        check_binding()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-research')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(uuid='GPU-other',name='test'))
    with pytest.raises(RuntimeError,match='different physical'):
        check_binding()


def test_required_binding_and_single_card(monkeypatch):
    monkeypatch.delenv('FOCUS_RESEARCH_GPU_UUID',raising=False)
    with pytest.raises(RuntimeError,match='Missing'):
        check_binding(required=True)
    assert check_binding() is None
    monkeypatch.setenv('FOCUS_RESEARCH_GPU_UUID','GPU-research')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-research')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    with pytest.raises(RuntimeError,match='exactly one'):
        check_binding()
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(uuid='GPU-research',name='test'))
    assert check_binding()['actual_uuid']=='GPU-research'
