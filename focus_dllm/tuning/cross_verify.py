"""Complementary masked views for empirical joint-update verification.

This is not a native-action/distribution certificate. Each prediction sees only
the opposite group's proposals; mutually consistent wrong proposals can pass.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CrossViews:
    canvas: torch.Tensor
    positions: torch.Tensor
    drafts: torch.Tensor
    owners: torch.Tensor


def make_views(canvas,positions,drafts,*,mask_id,special_ids=(),block_length=32):
    if canvas.ndim!=2 or canvas.shape[0]!=1 or canvas.dtype!=torch.long:
        raise ValueError('Batch-one int64 current canvas required')
    if positions.ndim!=1 or drafts.shape!=positions.shape or not 2<=positions.numel()<=8:
        raise ValueError('Between two and eight aligned candidate positions required')
    if any(t.dtype!=torch.long or t.device!=canvas.device for t in (positions,drafts)):
        raise ValueError('Aligned int64 candidate tensors required')
    if positions.unique().numel()!=positions.numel() or int(positions.min())<0:
        raise ValueError('Distinct nonnegative candidate positions required')
    if int(positions.max())>=min(block_length,canvas.shape[1]):
        raise ValueError('Candidates cannot cross the current block')
    if not bool((canvas[0,positions]==mask_id).all()):
        raise ValueError('Candidates must still be unresolved in the current state')
    forbidden=set(special_ids)|{mask_id}
    if any(int(token) in forbidden for token in drafts):
        raise ValueError('Special tokens cannot be speculative release targets')
    owners=torch.arange(positions.numel(),device=canvas.device)%2
    views=canvas.repeat(2,1)
    for row in range(2):
        visible=owners!=row
        views[row,positions[visible]]=drafts[visible]
        assert bool((views[row,positions[~visible]]==mask_id).all())
    return CrossViews(views,positions.clone(),drafts.clone(),owners)


def acceptance(logits,views,threshold=.90):
    if logits.ndim!=3 or logits.shape[:2]!=(2,views.positions.numel()):
        raise ValueError('Two aligned masked-view logit rows required')
    if logits.device!=views.canvas.device or not 0<threshold<=1:
        raise ValueError('Aligned logits and valid threshold required')
    if bool(((views.drafts<0)|(views.drafts>=logits.shape[-1])).any()):
        raise ValueError('Draft outside vocabulary')
    indices=torch.arange(views.positions.numel(),device=logits.device)
    own=logits[views.owners,indices]
    probability=own.double().softmax(-1).gather(-1,views.drafts[:,None]).squeeze(-1)
    accepted=(own.argmax(-1)==views.drafts)&(probability>=threshold)
    return accepted,probability
