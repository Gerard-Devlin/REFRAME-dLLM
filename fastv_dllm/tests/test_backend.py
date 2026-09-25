import torch

from fastv_dllm.backend import AttentionBackend


def test_sdpa_context_is_noop_and_reports_zero():
    with AttentionBackend("sdpa") as backend:
        assert torch.ones(1).item() == 1
    assert backend.report()["flash_calls"] == 0


def test_unknown_backend_rejected():
    try:
        AttentionBackend("magic")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown backend must fail closed")
