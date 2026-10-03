"""Dense diagnostic reference and independently written version-selection kernel.

Standard online-softmax attention; each original position has one selected KV.
This module imports Triton only when the GPU implementation is requested.
"""
import math
import torch


def dense_reference(q, base_k, base_v, draft_k, draft_v, mapping, choices):
    # q: M,H,D; cache: N,H,D; private draft: B,H,D.
    n = base_k.shape[0]
    scores_b = torch.einsum('mhd,nhd->mhn', q.float(), base_k.float())
    scores_d = torch.einsum('mhd,bhd->mhb', q.float(), draft_k.float())
    selected = choices[:, mapping.clamp_min(0)] & (mapping[None] >= 0)
    scores = torch.cat((scores_b.masked_fill(selected[:, None], -torch.inf),
                        scores_d.masked_fill(~choices[:, None], -torch.inf)), -1)
    p = (scores/math.sqrt(q.shape[-1])).softmax(-1)
    return (torch.einsum('mhn,nhd->mhd', p[:, :, :n], base_v.float())
            + torch.einsum('mhb,bhd->mhd', p[:, :, n:], draft_v.float()))


def streaming(q, base_k, base_v, draft_k, draft_v, mapping, choices):
    from .kernels import versioned_attention
    if not q.is_cuda or q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise ValueError('BF16 CUDA with128-dimensional heads required')
    m, h, d = q.shape
    b, n = draft_k.shape[0], base_k.shape[0]
    if not 0 < b <= 32 or choices.shape != (m, b) or mapping.shape != (n,):
        raise ValueError('Invalid version-selection geometry')
    out = torch.empty_like(q)
    versioned_attention[((m+31)//32, h)](
        q, base_k, base_v, draft_k, draft_v, mapping, choices, out,
        M=m, N=n, B=b, H=h, D=d, SCALE=1/math.sqrt(d),
        BM=32, BN=64, BD=32, num_warps=4, num_stages=2)
    return out
