"""Ridge interpolation of an update field from CURRENT shallow information.

The full cheap-prefix shallow forward already computes all MASK positions.
Its old-to-current changes form features; fresh deep KV is read at landmarks
only. Runtime ridge algebra is untrained, not an exact-Jacobian certificate.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ShallowTransport:
    source: torch.Tensor
    dropped: torch.Tensor
    weights: torch.Tensor
    shape: tuple

    @classmethod
    @torch.no_grad()
    def prepare(cls, old_hidden, current_hidden, source, dropped, *,
                prefix_length, kv_shape, regularization=.01):
        if old_hidden.ndim!=3 or old_hidden.shape!=current_hidden.shape or old_hidden.shape[0]!=1:
            raise ValueError('Matching batch-one shallow hidden states required')
        if old_hidden.device!=current_hidden.device or old_hidden.dtype!=current_hidden.dtype:
            raise ValueError('Shallow states have different device or dtype')
        if prefix_length<0 or len(kv_shape)!=4 or kv_shape[0]!=1 or kv_shape[-2]!=prefix_length+old_hidden.shape[1]:
            raise ValueError('Hidden states and KV canvas are not aligned')
        if not 0<regularization<=1:
            raise ValueError('Positive bounded ridge regularization required')
        for positions in (source,dropped):
            if positions.ndim!=1 or positions.dtype!=torch.long or positions.device!=old_hidden.device:
                raise ValueError('Device-aligned one-dimensional int64 positions required')
            if not positions.numel() or positions.unique().numel()!=positions.numel():
                raise ValueError('Unique nonempty position sets required')
            if int(positions.min())<prefix_length or int(positions.max())>=kv_shape[-2]:
                raise ValueError('Position outside shallow suffix')
        if torch.isin(source,dropped).any():
            raise ValueError('Unavailable dropped KV cannot be a landmark')
        delta=(current_hidden[0].float()-old_hidden[0].float())
        observed=delta.index_select(0,source-prefix_length)
        targets=delta.index_select(0,dropped-prefix_length)
        center=observed.mean(0,keepdim=True)
        features=observed-center;targets=targets-center
        gram=features@features.T
        ridge=(regularization*gram.diagonal().mean()).clamp_min(1e-6)
        matrix=gram+ridge*torch.eye(source.numel(),device=gram.device)
        # Solve against the cross-kernel, rather than explicitly forming an
        # inverse. The diagonal ridge and all weights use shallow inputs only.
        weights=torch.linalg.solve(matrix,(targets@features.T).T).T
        # Centered features sum to zero; enforce constant-field reproduction
        # explicitly despite FP32 solve/reduction error. Weights may be signed.
        weights-=weights.mean(-1,keepdim=True)
        weights+=1/source.numel()
        return cls(source.detach().clone(),dropped.detach().clone(),weights,tuple(kv_shape))

    @torch.no_grad()
    def predict(self,old,fresh):
        if tuple(old.shape)!=self.shape or fresh.shape!=old.shape:
            raise ValueError('KV canvas differs from the shallow geometry')
        if old.device!=self.weights.device or fresh.device!=old.device or fresh.dtype!=old.dtype:
            raise ValueError('KV device/dtype differs from shallow geometry')
        delta=(fresh.index_select(-2,self.source).float()-old.index_select(-2,self.source).float())[0]
        values=delta.transpose(0,1).reshape(self.source.numel(),-1)
        correction=(self.weights@values).reshape(self.dropped.numel(),old.shape[1],old.shape[3]).transpose(0,1)
        return old.index_select(-2,self.dropped).float()+correction.unsqueeze(0)
