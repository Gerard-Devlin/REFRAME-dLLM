import pytest
import torch

from generate import generate_with_dual_cache
from reframe_dllm.generate import generate_reframe
from reframe_dllm.model import ReframeConfig, ReframeSession, RefreshRequired
from reframe_dllm.oracle import NativeOracleWrapper, OracleProbe
from reframe_dllm.transport import fit_transport


@torch.no_grad()
def test_full_forward_matches_original_and_restores_methods(tiny_model):
    x = torch.randint(0, 120, (1, 24))
    expected = tiny_model(x).logits
    s = ReframeSession(tiny_model)
    actual, _ = s.full(x)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    assert all("attention" not in b.__dict__ for b in s.blocks)


@pytest.mark.parametrize("kind", ["stale", "pair"])
@torch.no_grad()
def test_noncontiguous_pilots_and_fresh_rows_match_full_state(tiny_model, kind):
    x = torch.randint(0, 120, (1, 48))
    s = ReframeSession(tiny_model, ReframeConfig(kind=kind, pilots=8))
    expected, _ = s.full(x)
    actual = s.partial(x, 20, 24)
    torch.testing.assert_close(actual, expected[:, 20:24], atol=2e-7, rtol=2e-5)
    assert s.last_positions.numel() < x.shape[1]


@torch.no_grad()
def test_materialization_and_transport_have_same_closed_loop_logits(tiny_model):
    x = torch.randint(0, 120, (1, 48))
    configs = [ReframeConfig(pilots=8, materialize=m, max_pilot_error=10) for m in (False, True)]
    sessions = [ReframeSession(tiny_model, c) for c in configs]
    for s in sessions:
        s.full(x)
    x[:, 20:24] = torch.randint(0, 120, (1, 4))
    a, b = [s.partial(x, 20, 24) for s in sessions]
    torch.testing.assert_close(a, b, atol=3e-7, rtol=2e-5)


@torch.no_grad()
def test_pending_write_uses_committed_ids_and_excludes_duplicate_kv(tiny_model):
    x = torch.randint(0, 120, (1, 48))
    s = ReframeSession(tiny_model, ReframeConfig(kind="stale"))
    s.full(x)
    old = {i: tuple(t.clone() for t in kv) for i, kv in s.reference.items()}
    pending = torch.arange(16, 20)
    x[:, 16:20] = torch.randint(0, 120, (1, 4))
    s.partial(x, 20, 24, pending)
    assert not torch.equal(s.reference[0][0][:, :, pending], old[0][0][:, :, pending])
    keep = torch.cat((torch.arange(16), torch.arange(20, 48)))
    for layer in old:
        for a, b in zip(s.reference[layer], old[layer]):
            torch.testing.assert_close(a[:, :, keep], b[:, :, keep], atol=0, rtol=0)
    assert s.stats["commit_token_rows"] == 4


@torch.no_grad()
def test_failed_late_layer_is_transactional_and_fallback_is_counted(tiny_model, monkeypatch):
    import reframe_dllm.model as module
    s = ReframeSession(tiny_model)
    x = torch.randint(0, 120, (1, 48))
    expected, _ = s.full(x)
    old = {i: tuple(t.clone() for t in kv) for i, kv in s.reference.items()}
    calls = 0

    def failing_fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RefreshRequired("injected_second_layer_failure")
        return fit_transport(*args, **kwargs)

    monkeypatch.setattr(module, "fit_transport", failing_fit)
    with pytest.raises(RefreshRequired):
        s.partial(x, 20, 24, torch.arange(16, 20))
    for layer in old:
        for a, b in zip(s.reference[layer], old[layer]):
            assert torch.equal(a, b)
    assert all("attention" not in b.__dict__ for b in s.blocks)
    calls = 0
    actual, refreshed = s.step(x, 20, 24, torch.arange(16, 20))
    assert refreshed and s.stats["fallback_refreshes"] == 1
    assert s.stats["full_forwards"] == 2 and s.stats["partial_forwards"] == 2
    torch.testing.assert_close(actual, expected[:, 20:24])


@torch.no_grad()
def test_stale_refresh_each_block_matches_native_dualcache(tiny_model):
    prompt = torch.tensor([[2, 4, 6, 8, 10, 12, 14, 16]])
    native, nnfe = generate_with_dual_cache(tiny_model, prompt, gen_length=8,
                                            block_length=4, steps=8, threshold=1, mask_id=127)
    ours, nfe, stats = generate_reframe(tiny_model, prompt, gen_length=8,
                                       block_length=4, threshold=1, mask_id=127,
                                       config=ReframeConfig(kind="stale", refresh_blocks=1))
    assert torch.equal(native, ours)
    assert nfe == nnfe == 8
    assert stats["full_forwards"] == 2


@torch.no_grad()
def test_cross_block_decoding_tracks_actual_work(tiny_model):
    prompt = torch.arange(2, 18).unsqueeze(0)
    out, nfe, stats = generate_reframe(tiny_model, prompt, gen_length=12, block_length=4,
                                       threshold=1, mask_id=127,
                                       config=ReframeConfig(kind="stale", refresh_blocks=3))
    assert not (out == 127).any()
    assert torch.equal(out[:, :16], prompt)
    assert stats["full_forwards"] == 1 and nfe == 12
    assert stats["commit_token_rows"] == 8


@torch.no_grad()
def test_oracle_does_not_change_baseline_or_its_cache(tiny_model):
    prompt = torch.arange(2, 18).unsqueeze(0)
    records = []
    probe = OracleProbe(records.append, layers=(0, 1), pilots=(4,), ages=(1,))
    kwargs = dict(gen_length=8, block_length=4, steps=8, threshold=1, mask_id=127)
    baseline, nfe = generate_with_dual_cache(tiny_model, prompt, **kwargs)
    wrapped = NativeOracleWrapper(tiny_model, probe, prompt.shape[1], 4)
    observed, onfe = generate_with_dual_cache(wrapped, prompt, **kwargs)
    assert torch.equal(baseline, observed) and nfe == onfe
    assert records and all(r["oracle"] for r in records)
    assert {r["kind"] for r in records} == {"stale", "shift", "scale", "pair"}
    assert max(r["execution_error"] for r in records) < 1e-5


def test_batch_and_configuration_guards(tiny_model):
    with pytest.raises(ValueError):
        ReframeConfig(pilots=1)
    with pytest.raises(ValueError):
        ReframeConfig(refresh_blocks=0)
    with pytest.raises(ValueError):
        ReframeSession(tiny_model).full(torch.ones(2, 4, dtype=torch.long))


@torch.no_grad()
def test_logit_audit_does_not_change_decoding(tiny_model):
    prompt = torch.arange(2, 18).unsqueeze(0)
    kwargs = dict(gen_length=8, block_length=4, threshold=1, mask_id=127,
                  config=ReframeConfig(kind="stale", refresh_blocks=2))
    baseline, nfe, _ = generate_reframe(tiny_model, prompt, **kwargs)
    observed, onfe, stats = generate_reframe(tiny_model, prompt, audit_every=2, **kwargs)
    assert torch.equal(baseline, observed) and nfe == onfe
    assert stats["diagnostic_run"] and stats["audit_full_forwards"] == 4
    assert all(0 <= a["top1_agreement"] <= 1 for a in stats["audits"])
