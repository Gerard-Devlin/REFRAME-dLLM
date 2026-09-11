"""Post-RoPE affine transport with exact group log-normalizer correction.

Tensors use [batch, heads, tokens, channels]. LLaDA RoPE pairs the first
half of channels with the second half (NOT neighboring channels).
"""
from dataclasses import dataclass
import math

import torch


def work(x):
    return x if x.dtype == torch.float64 else x.float()


def expand_heads(x, heads):
    if heads % x.shape[1]:
        raise ValueError("Query heads must be divisible by KV heads")
    return x if x.shape[1] == heads else x.repeat_interleave(heads // x.shape[1], 1)


@dataclass
class Transport:
    real: torch.Tensor
    imag: torch.Tensor
    key_bias: torch.Tensor
    value_scale: torch.Tensor
    value_bias: torch.Tensor

    def key(self, k):
        x, y = work(k).chunk(2, -1)
        return torch.cat((self.real * x - self.imag * y,
                          self.imag * x + self.real * y), -1) + self.key_bias

    def query(self, q):
        x, y = work(q).chunk(2, -1)
        a, b = expand_heads(self.real, q.shape[1]), expand_heads(self.imag, q.shape[1])
        return torch.cat((a * x + b * y, -b * x + a * y), -1)

    def value(self, v):
        return work(v) * self.value_scale + self.value_bias

    def output(self, o):
        return (work(o) * expand_heads(self.value_scale, o.shape[1])
                + expand_heads(self.value_bias, o.shape[1]))

    def lse_bias(self, q):
        return (work(q) * expand_heads(self.key_bias, q.shape[1])).sum(-1) / math.sqrt(q.shape[-1])

    def inverse(self, k, v):
        x, y = (work(k) - self.key_bias).chunk(2, -1)
        norm = self.real.square() + self.imag.square()
        kr = torch.cat(((self.real * x + self.imag * y) / norm,
                        (-self.imag * x + self.real * y) / norm), -1)
        vr = (work(v) - self.value_bias) / self.value_scale
        return kr, vr

    def is_safe(self, min_scale=0.25, max_scale=4.0):
        ks = (self.real.square() + self.imag.square()).sqrt()
        vs = self.value_scale.abs()
        checks = [torch.isfinite(t).all() for t in vars(self).values()]
        checks += [(ks >= min_scale).all(), (ks <= max_scale).all(),
                   (vs >= min_scale).all(), (vs <= max_scale).all()]
        return bool(torch.stack(checks).all().item())


def identity(k, v):
    if k.shape[-1] % 2:
        raise ValueError("Post-RoPE key dimension must be even")
    kh, vh = work(k[..., :1, :]), work(v[..., :1, :])
    # Shape-only construction also supports zero-length token groups.
    shape = (*k.shape[:2], 1, k.shape[-1] // 2)
    real = torch.ones(shape, dtype=kh.dtype, device=k.device)
    vb = torch.zeros((*v.shape[:2], 1, v.shape[-1]), dtype=vh.dtype, device=v.device)
    return Transport(real, torch.zeros_like(real),
                     torch.zeros((*k.shape[:2], 1, k.shape[-1]), dtype=kh.dtype, device=k.device),
                     torch.ones_like(vb), vb)


def fit_transport(k0, v0, kt, vt, kind="pair", ridge=1e-3):
    """Fit using pilot rows ONLY; ridge shrinks scales/rotations to identity.

    stale: identity; shift: translation; scale: pairwise real key scales and
    channelwise value scales, no bias; pair: complex key scale/rotation+bias,
    channelwise affine values. Statistics use FP32 (FP64 for FP64 tests).
    """
    if kind not in {"stale", "shift", "scale", "pair"}:
        raise ValueError(f"Unknown transport kind: {kind}")
    if ridge < 0 or k0.shape != kt.shape or v0.shape != vt.shape:
        raise ValueError("Invalid ridge or mismatched pilot shapes")
    t = identity(k0, v0)
    if kind == "stale" or k0.shape[-2] == 0:
        return t
    k0, v0, kt, vt = map(work, (k0, v0, kt, vt))
    if kind == "shift":
        t.key_bias = (kt - k0).mean(-2, keepdim=True)
        t.value_bias = (vt - v0).mean(-2, keepdim=True)
        return t
    affine = kind == "pair"
    means = [x.mean(-2, keepdim=True) if affine else torch.zeros_like(x[..., :1, :])
             for x in (k0, v0, kt, vt)]
    km, vm, ktm, vtm = means
    x, y = (k0 - km).chunk(2, -1)
    u, v = (kt - ktm).chunk(2, -1)
    den = (x.square() + y.square()).mean(-2, keepdim=True) + ridge
    eps = torch.finfo(den.dtype).eps
    t.real = ((x * u + y * v).mean(-2, keepdim=True) + ridge) / den.clamp_min(eps)
    t.imag = ((x * v - y * u).mean(-2, keepdim=True) / den.clamp_min(eps)
              if affine else torch.zeros_like(t.real))
    # With no pilot variation, choose identity and a mean shift.
    t.real = torch.where(den > eps, t.real, torch.ones_like(t.real))
    t.key_bias = ktm - t.key(km)
    den_v = (v0 - vm).square().mean(-2, keepdim=True) + ridge
    t.value_scale = (((v0 - vm) * (vt - vtm)).mean(-2, keepdim=True) + ridge) / den_v.clamp_min(eps)
    t.value_scale = torch.where(den_v > eps, t.value_scale, torch.ones_like(t.value_scale))
    t.value_bias = vtm - vm * t.value_scale
    return t


def attention_lse(q, k, v, backend="torch", query_chunk=128):
    """Normalized attention AND logsumexp. No dropout, causal mask or padding.

    flash requires FA2's public return_attn_probs=True interface. With zero
    dropout FA2 2.8.3 returns LSE without allocating the full probability map.
    torch is a chunked FP32 reference, not an optimized performance baseline.
    """
    if k.shape[-2] == 0:
        shape = (*q.shape[:-1], v.shape[-1])
        return work(q).new_zeros(shape), work(q).new_full(q.shape[:-1], -torch.inf)
    if backend == "flash":
        if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("flash backend requires CUDA FP16/BF16")
        from flash_attn import flash_attn_func
        out, lse, _ = flash_attn_func(q.transpose(1, 2).contiguous(),
                                     k.transpose(1, 2).contiguous(),
                                     v.transpose(1, 2).contiguous(),
                                     dropout_p=0.0, causal=False, return_attn_probs=True)
        return out.transpose(1, 2), lse[..., :q.shape[-2]]
    if backend != "torch" or query_chunk < 1:
        raise ValueError("backend must be torch or flash, query_chunk must be positive")
    k, v = expand_heads(work(k), q.shape[1]), expand_heads(work(v), q.shape[1])
    outs, lses = [], []
    for qc in work(q).split(query_chunk, -2):
        scores = qc @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        lses.append(torch.logsumexp(scores, -1))
        outs.append(torch.softmax(scores, -1) @ v)
    return torch.cat(outs, -2), torch.cat(lses, -1)


def grouped_attention(q, groups, backend="torch", materialize=False):
    """groups=(reference_K, reference_V, transform_or_None), each token once.

    materialize is the diagnostic control: explicitly reconstruct all current
    K/V. Normal execution only transforms queries/outputs and group LSEs.
    """
    groups = [(k, v, t) for k, v, t in groups if k.shape[-2]]
    if not groups:
        raise ValueError("Attention needs at least one key")
    if materialize:
        keys = [(k if t is None else t.key(k).to(k.dtype)) for k, _, t in groups]
        vals = [(v if t is None else t.value(v).to(v.dtype)) for _, v, t in groups]
        return attention_lse(q, torch.cat(keys, -2), torch.cat(vals, -2), backend)[0].to(q.dtype)
    outputs, normalizers = [], []
    for k, v, t in groups:
        qt = q if t is None else t.query(q).to(q.dtype)
        o, lse = attention_lse(qt, k, v, backend)
        outputs.append(work(o) if t is None else t.output(o))
        normalizers.append(lse if t is None else lse + t.lse_bias(q))
    lse = torch.stack(normalizers, 0)
    weights = torch.softmax(lse, dim=0)
    return (torch.stack(outputs, 0) * weights.unsqueeze(-1)).sum(0).to(q.dtype)


def relative_error(prediction, target):
    return float(((work(prediction) - work(target)).norm() / work(target).norm().clamp_min(1e-12)).item())
