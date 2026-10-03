# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
# Modified from pinned VILA-Lab/Flash-dLLM, flash_cache_triton.py:
# 7437a550fd3d1a0752edcbf58bd015ad69083068.
# QKV primitive retained; added common-only public writes. Joint attention uses
# 32 query rows with ALL 128 local keys, preserving the full visibility matrix.
import triton
import triton.language as tl

@triton.jit
def tiled_qkv(
    Xn, Q1, K1, V1, # (acc_q_len, d_model)
    Pos,
    Wq, Wk, Wv, # (d_model, d_model)
    RotarySin, RotaryCos, # (acc_k_len, head_dim)
    block_table,
    PublicK, PublicV, CommonN,
    HALF: tl.constexpr,
    D_MODEL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)  # block over query rows
    pid_h = tl.program_id(1)  # head

    bt_row = block_table + pid_m * 4
    start_n = tl.load(bt_row + 0)
    end_n = tl.load(bt_row + 1)
    start_m = tl.load(bt_row + 2)
    end_m = tl.load(bt_row + 3)
    
    range_half = tl.arange(0, HALF)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_h_half = pid_h * HEAD_DIM + range_half
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = offs_m < end_m
    pos = tl.load(Pos + offs_m, mask=q_mask, other=0.0)
    
    offs_q_half = offs_m[:, None] * D_MODEL + offs_h_half[None, :]
    offs_x = offs_m[:, None] * D_MODEL + offs_d[None, :]

    offs_w1 = offs_h_half[None, :] * D_MODEL + offs_d[:, None]
    offs_w2 = (offs_h_half + HALF)[None, :] * D_MODEL + offs_d[:, None]

    # QKV projection for a single head
    acc_q1 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_k1 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_v1 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)

    acc_q2 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_k2 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_v2 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)

    for d in range(0, D_MODEL, BLOCK_D):
        x = tl.load(Xn + offs_x + d, mask=q_mask[:, None], other=0.0)

        wq1 = tl.load(Wq + offs_w1 + d)
        wk1 = tl.load(Wk + offs_w1 + d)
        wv1 = tl.load(Wv + offs_w1 + d)

        wq2 = tl.load(Wq + offs_w2 + d)
        wk2 = tl.load(Wk + offs_w2 + d)
        wv2 = tl.load(Wv + offs_w2 + d)

        acc_q1 += tl.dot(x, wq1)
        acc_k1 += tl.dot(x, wk1)
        acc_v1 += tl.dot(x, wv1)

        acc_q2 += tl.dot(x, wq2)
        acc_k2 += tl.dot(x, wk2)
        acc_v2 += tl.dot(x, wv2)

    # Apply rotary for a single head.
    offs_rot = pos[:, None] * HEAD_DIM + range_half[None, :]
    sin1 = tl.load(RotarySin + offs_rot, mask=q_mask[:, None], other=0.0)
    sin2 = tl.load(RotarySin + offs_rot + HALF, mask=q_mask[:, None], other=0.0)
    cos1 = tl.load(RotaryCos + offs_rot, mask=q_mask[:, None], other=0.0)
    cos2 = tl.load(RotaryCos + offs_rot + HALF, mask=q_mask[:, None], other=0.0)

    rq1 = acc_q1 * cos1 - acc_q2 * sin1
    rq2 = acc_q2 * cos2 + acc_q1 * sin2
    rk1 = acc_k1 * cos1 - acc_k2 * sin1
    rk2 = acc_k2 * cos2 + acc_k1 * sin2

    tl.store(Q1 + offs_q_half, rq1, mask=q_mask[:, None])
    tl.store(Q1 + offs_q_half + HALF, rq2, mask=q_mask[:, None])

    tl.store(K1 + offs_q_half, rk1, mask=q_mask[:, None])
    tl.store(K1 + offs_q_half + HALF, rk2, mask=q_mask[:, None])

    tl.store(V1 + offs_q_half, acc_v1, mask=q_mask[:, None])
    tl.store(V1 + offs_q_half + HALF, acc_v2, mask=q_mask[:, None])
    public = q_mask & (offs_m < CommonN)
    global_offs = pos[:, None] * D_MODEL + offs_h_half[None, :]
    tl.store(PublicK + global_offs, rk1, mask=public[:, None])
    tl.store(PublicK + global_offs + HALF, rk2, mask=public[:, None])
    tl.store(PublicV + global_offs, acc_v1, mask=public[:, None])
    tl.store(PublicV + global_offs + HALF, acc_v2, mask=public[:, None])


@triton.jit
def tiled_attention(Q, LocalK, LocalV, O, K, V, PosK, Mask, Table, Scale,
                    D_MODEL: tl.constexpr, HEAD_DIM: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, LOCAL: tl.constexpr):
    tile=tl.program_id(0);head=tl.program_id(1)
    row=Table+tile*4
    begin=tl.load(row);end=tl.load(row+1);m=tl.load(row+2)
    mi=m+tl.arange(0,BLOCK_M)
    ni=tl.arange(0,BLOCK_N)
    hi=head*HEAD_DIM+tl.arange(0,HEAD_DIM)
    qo=mi[:,None]*D_MODEL+hi[None,:]
    q=tl.load(Q+qo)
    maximum=tl.full((BLOCK_M,),float('-inf'),tl.float32)
    lse=tl.full((BLOCK_M,),float('-inf'),tl.float32)
    acc=tl.zeros((BLOCK_M,HEAD_DIM),tl.float32)
    # Same cached-key grouping and recurrence as the official primitive.
    for n in range(begin,end,BLOCK_N):
        n=tl.multiple_of(n,BLOCK_N)
        idx=n+ni;valid=idx<end
        pos=tl.load(PosK+idx,mask=valid,other=0)
        k=tl.load(K+pos[None,:]*D_MODEL+hi[:,None],mask=valid[None,:],other=0.)
        v=tl.load(V+pos[:,None]*D_MODEL+hi[None,:],mask=valid[:,None],other=0.)
        score=tl.dot(q,k)
        score+=tl.where(valid[None,:],0.,float('-inf'))
        score*=Scale
        newmax=tl.maximum(tl.max(score,1),maximum)
        p=tl.exp(score-newmax[:,None])
        mass=tl.sum(p,1)
        acc=acc*tl.exp(maximum-newmax)[:,None]
        acc+=tl.dot(p.to(v.dtype),v)
        lse=newmax+tl.log(tl.exp(lse-newmax)+mass)
        maximum=newmax
    # All local keys are visible according to the ORIGINAL joint matrix.
    # Splitting query tiles must never split/restrict their key context.
    local=tl.arange(0,LOCAL)
    k=tl.load(LocalK+local[None,:]*D_MODEL+hi[:,None])
    v=tl.load(LocalV+local[:,None]*D_MODEL+hi[None,:])
    allowed=tl.load(Mask+mi[:,None]*LOCAL+local[None,:])
    score=tl.dot(q,k)*Scale+tl.where(allowed,0.,float('-inf'))
    p=tl.exp(score-maximum[:,None])
    mass=tl.sum(p,1)
    acc+=tl.dot(p.to(v.dtype),v)
    lse=maximum+tl.log(tl.exp(lse-maximum)+mass)
    tl.store(O+qo,acc*tl.exp(maximum-lse)[:,None])
