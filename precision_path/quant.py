"""Explicit real FP8 GEMMs, no fake-quant or silent fallback.

Uses the existing PyTorch CUDA _scaled_mm ABI, tested separately at startup.
This is a per-tensor RTN/dynamic-activation baseline, not an optimized PTQ claim.
"""
import torch
from torch import nn

TARGETS={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}


def quantize_tensor(x):
    value=x.float()
    scale=(value.abs().amax()/448.0).clamp_min(1e-12)
    return (value/scale).clamp(-448,448).to(torch.float8_e4m3fn),scale


class FP8Linear(nn.Module):
    def __init__(self,linear):
        super().__init__()
        if linear.weight.device.type!='cuda' or linear.weight.dtype!=torch.bfloat16:
            raise ValueError('Load original BF16 linear on CUDA first')
        if linear.in_features%16 or linear.out_features%16:
            raise ValueError('FP8 GEMM needs projection dimensions divisible by 16')
        q,s=quantize_tensor(linear.weight.detach())
        self.register_buffer('weight_fp8',q.contiguous())
        self.register_buffer('scale',s)
        self.register_buffer('bias',None if linear.bias is None else linear.bias.detach().clone())
        self.in_features,self.out_features=linear.in_features,linear.out_features

    def forward(self,x):
        if x.device.type!='cuda' or x.dtype!=torch.bfloat16:
            raise ValueError('FP8 backend requires BF16 CUDA inputs')
        shape=x.shape
        a=x.reshape(-1,self.in_features).contiguous()
        # M padding for short static-low prefill is included in measured cost.
        pad=(-a.shape[0])%16
        if pad:
            a=torch.nn.functional.pad(a,(0,0,0,pad))
        aq,scale=quantize_tensor(a)
        y=torch._scaled_mm(aq,self.weight_fp8.t(),scale_a=scale,scale_b=self.scale,
                           out_dtype=torch.bfloat16,use_fast_accum=False)
        if not isinstance(y,torch.Tensor):
            raise RuntimeError('Unsupported _scaled_mm ABI; no fallback allowed')
        if pad:
            y=y[:-pad]
        if self.bias is not None:
            y=y+self.bias
        return y.reshape(*shape[:-1],self.out_features)


def convert(model):
    replaced=[]
    for name,module in list(model.named_modules()):
        if isinstance(module,nn.Linear) and name.split('.')[-1] in TARGETS:
            parent_name,_,child=name.rpartition('.')
            parent=model.get_submodule(parent_name)
            setattr(parent,child,FP8Linear(module))
            replaced.append(name)
    if not replaced:
        raise ValueError('No expected transformer projections found')
    return dict(backend='torch_scaled_mm_e4m3_tensorwise_v1',projections=replaced,
                weight_dtype='float8_e4m3fn',activation_dtype='float8_e4m3fn',
                accumulator_output='bfloat16',use_fast_accum=False,
                unchanged='BF16 embeddings/output head, native norms/RoPE/attention',
                activation_scaling='dynamic per tensor',weight_scaling='per tensor RTN',
                training=False)


@torch.no_grad()
def kernel_probe(device):
    layer=nn.Linear(256,256,device=device,dtype=torch.bfloat16)
    low=FP8Linear(layer)
    x=torch.randn(1,32,256,device=device,dtype=torch.bfloat16)
    y=low(x)
    torch.cuda.synchronize(device)
    if not torch.isfinite(y).all():
        raise RuntimeError('Real FP8 kernel produced non-finite output')
    return dict(operator='aten::_scaled_mm',input_dtype=str(low.weight_fp8.dtype),
                output_dtype=str(y.dtype),shape=list(y.shape),finite=True)
