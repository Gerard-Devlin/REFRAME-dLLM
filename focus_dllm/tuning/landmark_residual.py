"""Historical-feature barycentric transport of live landmark KV changes.

An untrained local interpolation primitive, not a lossless certificate. All
weights use OLD K/V only; only source/landmark rows of NEW K/V are read by the
predictor. Full fresh dropped rows are used solely by the separate diagnostic.
Construction and application are real costs and cannot be called free.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalTransport:
    source: torch.Tensor
    dropped: torch.Tensor
    neighbors: torch.Tensor
    weights: torch.Tensor
    shape: tuple

    @classmethod
    @torch.no_grad()
    def prepare(cls, old_pair, source, dropped, neighbors=4):
        key,value=old_pair
        if key.ndim!=4 or key.shape!=value.shape or key.shape[0]!=1:
            raise ValueError('Matching batch-one historical KV required')
        if key.device!=value.device or key.dtype!=value.dtype:
            raise ValueError('Historical K/V device and dtype differ')
        if not isinstance(neighbors,int) or neighbors<1:
            raise ValueError('Positive integer neighborhood required')
        for positions in (source,dropped):
            if positions.ndim!=1 or positions.dtype!=torch.long or positions.device!=key.device:
                raise ValueError('Device-aligned one-dimensional int64 positions required')
            if not positions.numel() or positions.unique().numel()!=positions.numel():
                raise ValueError('Nonempty unique positions required')
            if int(positions.min())<0 or int(positions.max())>=key.shape[-2]:
                raise ValueError('Position outside historical canvas')
        if torch.isin(source,dropped).any():
            raise ValueError('Dropped labels cannot be landmarks')
        # Each KV kind has equal aggregate feature energy after centering. This
        # is history-only scaling, never regression on unavailable fresh labels.
        future=torch.cat((source,dropped))
        features=[]
        for tensor in old_pair:
            selected=tensor[0].index_select(1,future).float()
            centered=selected-selected.mean(1,keepdim=True)
            scale=centered.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-6)
            features.append(centered/scale)
        combined=torch.cat(features,-1)
        landmarks=combined[:,:source.numel()]
        targets=combined[:,source.numel():]
        squared=(targets.square().sum(-1,keepdim=True)+
            landmarks.square().sum(-1).unsqueeze(1)-
            2*torch.matmul(targets,landmarks.transpose(1,2))).clamp_min_(0)
        distances,indices=squared.topk(min(neighbors,source.numel()),dim=-1,largest=False,sorted=True)
        # The kth-neighbor radius defines the scale, without a fitted bandwidth.
        radius=distances[...,-1:].clamp_min(1e-6)
        weights=(-distances/radius).softmax(-1)
        return cls(source.detach().clone(),dropped.detach().clone(),indices,
            weights,tuple(key.shape))

    @torch.no_grad()
    def predict(self, old, fresh):
        if tuple(old.shape)!=self.shape or fresh.shape!=old.shape:
            raise ValueError('KV canvas differs from the prepared geometry')
        if old.device!=self.weights.device or fresh.device!=old.device or fresh.dtype!=old.dtype:
            raise ValueError('KV device/dtype differs from the prepared geometry')
        delta=(fresh[0].index_select(1,self.source).float()-
            old[0].index_select(1,self.source).float())
        heads=torch.arange(old.shape[1],device=old.device)[:,None,None]
        nearby=delta[heads,self.neighbors]
        correction=(nearby*self.weights.unsqueeze(-1)).sum(-2)
        return old.index_select(-2,self.dropped).float()+correction.unsqueeze(0)


@torch.no_grad()
def error_metrics(plan,old,fresh):
    """Fresh dropped rows are diagnostic labels only, not predictor features."""
    prediction=plan.predict(old,fresh)
    baseline=old.index_select(-2,plan.dropped).float()
    labels=fresh.index_select(-2,plan.dropped).float()
    source_delta=fresh.index_select(-2,plan.source).float()-old.index_select(-2,plan.source).float()
    mean_prediction=baseline+source_delta.mean(-2,keepdim=True)
    return torch.stack(((labels-baseline).square().sum(),
        (labels-prediction).square().sum(),(labels-mean_prediction).square().sum(),
        torch.tensor(labels.numel(),device=old.device,dtype=torch.float32)))
