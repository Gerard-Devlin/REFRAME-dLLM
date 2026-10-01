"""Process-local GEMM numeric control; the same setting applies to every path.

Disabling BF16 intermediate reduced reductions keeps BF16 inputs/outputs. It
does not certify shape-independent logits or equality to an earlier setting.
"""
from contextlib import AbstractContextManager

import torch


class BF16Reduction(AbstractContextManager):
    def __init__(self, disable=False):
        self.disable=disable
        self.audit={}

    def __enter__(self):
        self.old=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        if self.disable:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        self.audit.update(before=self.old,
            during=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            requested_disable=self.disable,restored=False,
            scope='Common to teacher generation, serial and batched verification in this process')
        return self

    def __exit__(self,*exc):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=self.old
        self.audit['restored']=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction==self.old
        return False
