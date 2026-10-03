"""Full-vocabulary FP64 normalization, returning ONLY required statistics.

No top-k renormalization or change to Flash confidence/acceptance semantics.
Chunked FP64 summation can differ from PyTorch's reduction by rounding; this
is not a bitwise or all-input decision-equivalence theorem.
"""
import torch


def reference(logits, labels=None):
    if logits.ndim!=2 or logits.shape[1]==0:raise ValueError('Nonempty vocabulary required')
    p=logits.double().softmax(-1)
    confidence,top=p.max(-1)
    if labels is not None:
        if labels.shape!=(logits.shape[0],):raise ValueError('One target per row required')
        confidence=p.gather(1,labels[:,None]).flatten()
    return confidence,top


_kernels=None


def kernels():
    # Triton 3.3 resolves JIT symbols through the function module globals.
    # Keep imports lazy for CPU-only tests, but expose modules to its compiler.
    global _kernels,tl,libdevice
    if _kernels is not None:return _kernels
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def partial(Z,M,D,I,V:tl.constexpr,STRIDE:tl.constexpr,CHUNKS:tl.constexpr,BLOCK:tl.constexpr):
        row=tl.program_id(0);chunk=tl.program_id(1)
        column=chunk*BLOCK+tl.arange(0,BLOCK)
        value=tl.load(Z+row*STRIDE+column,column<V,other=-float('inf')).to(tl.float32)
        maximum=tl.max(value,0)
        index=tl.min(tl.where((column<V)&(value==maximum),column,2147483647),0)
        # FP64 CUDA exp and sums, matching the official precision policy.
        denominator=tl.sum(libdevice.exp(value.to(tl.float64)-maximum.to(tl.float64)),0)
        slot=row*CHUNKS+chunk
        tl.store(M+slot,maximum);tl.store(D+slot,denominator);tl.store(I+slot,index)

    @triton.jit
    def finish(Z,L,M,D,I,C,T,V:tl.constexpr,STRIDE:tl.constexpr,CHUNKS:tl.constexpr,
               BLOCK:tl.constexpr,TARGET:tl.constexpr):
        row=tl.program_id(0);offset=tl.arange(0,BLOCK)
        maximum=tl.load(M+row*CHUNKS+offset,offset<CHUNKS,other=-float('inf'))
        totalmax=tl.max(maximum,0)
        sums=tl.load(D+row*CHUNKS+offset,offset<CHUNKS,other=0.)
        denominator=tl.sum(sums*libdevice.exp(maximum.to(tl.float64)-totalmax.to(tl.float64)),0)
        index=tl.load(I+row*CHUNKS+offset,offset<CHUNKS,other=2147483647)
        top=tl.min(tl.where(maximum==totalmax,index,2147483647),0)
        numerator=tl.full((),1.,tl.float64)
        if TARGET:
            label=tl.load(L+row)
            value=tl.load(Z+row*STRIDE+label,(label>=0)&(label<V),other=float('nan')).to(tl.float64)
            numerator=libdevice.exp(value-totalmax.to(tl.float64))
        tl.store(C+row,numerator/denominator);tl.store(T+row,top)

    _kernels=(partial,finish,triton)
    return _kernels


def statistics(logits,labels=None):
    if logits.ndim!=2 or logits.shape[1]==0:raise ValueError('Nonempty vocabulary required')
    if labels is not None and (labels.shape!=(logits.shape[0],) or labels.dtype not in (torch.int32,torch.int64)):
        raise ValueError('One integer target per row required')
    if not logits.is_cuda:return reference(logits,labels)
    if logits.stride(1)!=1:raise ValueError('Contiguous vocabulary axis required')
    if labels is not None and (labels.device!=logits.device or labels.stride(0)!=1):
        raise ValueError('Contiguous targets on logits device required')
    rows,vocabulary=logits.shape
    confidence=torch.empty(rows,device=logits.device,dtype=torch.float64)
    top=torch.empty(rows,device=logits.device,dtype=torch.long)
    if rows==0:return confidence,top
    partial,finish,triton=kernels();block=2048;chunks=triton.cdiv(vocabulary,block)
    maximum=torch.empty((rows,chunks),device=logits.device,dtype=torch.float32)
    denominator=torch.empty((rows,chunks),device=logits.device,dtype=torch.float64)
    indices=torch.empty((rows,chunks),device=logits.device,dtype=torch.int32)
    partial[(rows,chunks)](logits,maximum,denominator,indices,vocabulary,logits.stride(0),chunks,block,num_warps=4)
    finish[(rows,)](logits,labels if labels is not None else top,maximum,denominator,indices,
                   confidence,top,vocabulary,logits.stride(0),chunks,triton.next_power_of_2(chunks),
                   labels is not None,num_warps=4)
    return confidence,top
