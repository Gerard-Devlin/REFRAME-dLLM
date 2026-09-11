import pytest
import torch

from reframe_dllm.transport import (attention_lse, fit_transport, grouped_attention,
                                    identity, relative_error)


def data(tokens=19, heads=2, dim=8, dtype=torch.float64):
    return torch.randn(1, heads, tokens, dim, dtype=dtype)


def transformed(k, v):
    t = identity(k, v)
    t.real += 0.1 * torch.randn_like(t.real)
    t.imag += 0.2 * torch.randn_like(t.imag)
    t.key_bias += torch.randn_like(t.key_bias)
    t.value_scale += 0.1 * torch.randn_like(t.value_scale)
    t.value_bias += torch.randn_like(t.value_bias)
    return t


@pytest.mark.parametrize("qheads", [2, 4])
def test_group_transport_matches_materialization_and_gqa(qheads):
    torch.manual_seed(41)
    q = data(5, qheads)
    groups = []
    for n in [19, 11]:
        k, v = data(n), data(n)
        groups.append((k, v, transformed(k, v)))
    groups.append((data(3), data(3), None))
    expected = grouped_attention(q, groups, materialize=True)
    actual = grouped_attention(q, groups)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_group_translation_must_change_normalizer():
    q = torch.tensor([[[[2.0 ** 0.5, 0.0]]]], dtype=torch.float64)
    k = torch.zeros_like(q)
    va, vb = torch.ones_like(q), torch.zeros_like(q)
    t = identity(k, va)
    t.key_bias[..., 0] = 4
    actual = grouped_attention(q, [(k, va, t), (k, vb, None)])
    torch.testing.assert_close(actual, torch.full_like(actual, torch.sigmoid(torch.tensor(4.0, dtype=torch.float64))))
    assert abs(actual.flatten()[0].item() - 0.5) > 0.48


def test_fit_on_pilots_generalizes_for_planted_transform():
    torch.manual_seed(32)
    k, v = data(80), data(80)
    t = transformed(k, v)
    kt, vt = t.key(k), t.value(v)
    fitted = fit_transport(k[:, :, :16], v[:, :, :16], kt[:, :, :16], vt[:, :, :16], ridge=0)
    torch.testing.assert_close(fitted.key(k[:, :, 16:]), kt[:, :, 16:], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(fitted.value(v[:, :, 16:]), vt[:, :, 16:], rtol=1e-12, atol=1e-12)
    assert fitted.is_safe()


def test_inverse_write_preserves_new_token_in_current_frame():
    k, v = data(), data()
    t = transformed(k, v)
    newk, newv = data(3), data(3)
    kr, vr = t.inverse(newk, newv)
    torch.testing.assert_close(t.key(kr), newk, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(t.value(vr), newv, rtol=1e-12, atol=1e-12)
    t.value_scale[..., 0] = 0
    assert not t.is_safe()


@pytest.mark.parametrize("kind", ["stale", "shift", "scale", "pair"])
def test_unchanged_state_is_identity_and_empty_pilots_are_safe(kind):
    k, v = data(), data()
    t = fit_transport(k, v, k, v, kind)
    torch.testing.assert_close(t.key(k), k)
    torch.testing.assert_close(t.value(v), v)
    empty = fit_transport(k[:, :, :0], v[:, :, :0], k[:, :, :0], v[:, :, :0], kind)
    assert empty.is_safe()
    torch.testing.assert_close(empty.key(k), k)


def test_bfloat16_reference_is_finite_and_close():
    torch.manual_seed(19)
    q, k, v = data(3, dtype=torch.bfloat16), data(dtype=torch.bfloat16), data(dtype=torch.bfloat16)
    t = transformed(k, v)
    a = grouped_attention(q, [(k, v, t), (k[:, :, :2], v[:, :, :2], None)])
    b = grouped_attention(q, [(k, v, t), (k[:, :, :2], v[:, :, :2], None)], materialize=True)
    assert torch.isfinite(a).all()
    assert relative_error(a, b) < 0.02


def test_good_pilots_do_not_certify_heldout_tokens():
    torch.manual_seed(31)
    k, v = data(32), data(32)
    kt, vt = k.clone(), v.clone()
    kt[:, :, 8:] += 3
    vt[:, :, 8:] += 10
    t = fit_transport(k[:, :, :8], v[:, :, :8], kt[:, :, :8], vt[:, :, :8])
    assert relative_error(t.value(v[:, :, :8]), vt[:, :, :8]) < 1e-12
    assert relative_error(t.value(v[:, :, 8:]), vt[:, :, 8:]) > 0.8


def test_chunked_attention_matches_torch_sdpa():
    q, k, v = data(23), data(17), data(17)
    out, lse = attention_lse(q, k, v, query_chunk=4)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)
    assert torch.isfinite(lse).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_lse_and_group_merge():
    pytest.importorskip("flash_attn")
    torch.manual_seed(3)
    q = data(4, dim=32, dtype=torch.bfloat16).cuda()
    k, v = data(21, dim=32, dtype=torch.bfloat16).cuda(), data(21, dim=32, dtype=torch.bfloat16).cuda()
    groups = [(k, v, transformed(k, v)), (k[:, :, :3], v[:, :, :3], None)]
    a = grouped_attention(q, groups, "flash")
    b = grouped_attention(q, groups, "torch")
    assert relative_error(a, b) < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_bfloat16_transport_against_materialization():
    torch.manual_seed(23)
    q = data(5, dim=32, dtype=torch.bfloat16).cuda()
    k, v = data(31, dim=32, dtype=torch.bfloat16).cuda(), data(31, dim=32, dtype=torch.bfloat16).cuda()
    t = transformed(k, v)
    groups = [(k, v, t), (k[:, :, :4], v[:, :, :4], None)]
    actual = grouped_attention(q, groups)
    expected = grouped_attention(q, groups, materialize=True)
    assert relative_error(actual, expected) < 0.02
