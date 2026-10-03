"""Cheap coarse interaction from normal-forward Q and existing layer4 cache.

No additional Transformer call. Pooled-key softmax, age and last *observed*
drift are planning proxies, not true attention or causal dependencies.
"""
import math
import torch


def pooled_keys(keys,tiles):
    """One gather/reduction for equal-width tiles, at most one tail gather."""
    if not tiles:
        return keys.new_empty((0,)+tuple(keys.shape[1:]),dtype=torch.float32)
    width=len(tiles[0]);full=len(tiles) if len(tiles[-1])==width else len(tiles)-1
    if any(len(t)!=width for t in tiles[:full]):
        raise ValueError('Equal-width tiles and at most one short tail required')
    pieces=[]
    if full:
        index=torch.tensor([p for t in tiles[:full] for p in t],device=keys.device)
        pieces.append(keys.index_select(0,index).view(full,width,*keys.shape[1:]).float().mean(1))
    if full<len(tiles):
        index=torch.tensor(tiles[-1],device=keys.device)
        pieces.append(keys.index_select(0,index).float().mean(0,keepdim=True))
    return torch.cat(pieces,0) if len(pieces)>1 else pieces[0]


@torch.no_grad()
def measure(runtime, query, query_positions, candidate_positions, *, tile_width=4, pool_tiles=8, requirements_per_candidate=2):
    ledger = runtime.ledger; n = len(ledger.tokens)
    dirty = set(ledger.dirty())
    known = [i for i in range(n) if ledger.tokens[i] != 126336 and i not in dirty]
    tiles = [tuple(known[i:i+tile_width]) for i in range(0,len(known),tile_width)]
    if not candidate_positions:
        return dict(tiles=(),requirements=(),interaction=())
    locate = {p:i for i,p in enumerate(query_positions)}
    q = query.index_select(0,torch.tensor([locate[p] for p in candidate_positions],device=query.device)).float()
    keys = runtime.cache[3][0]
    candidate_k = keys.index_select(0,torch.tensor(candidate_positions,device=q.device)).float()
    interaction = torch.einsum('ihd,jhd->ij',q,candidate_k)/(q.shape[1]*math.sqrt(q.shape[-1]))
    interaction = interaction.softmax(-1).cpu().tolist()
    if not tiles:
        return dict(tiles=(),requirements=((),)*len(candidate_positions),interaction=interaction)
    centers = pooled_keys(keys,tiles)
    coarse = torch.einsum('ihd,jhd->ij',q,centers)/(q.shape[1]*math.sqrt(q.shape[-1]))
    saliency = coarse.softmax(-1).cpu().tolist()
    scores = []
    for row in saliency:
        scores.append([s*(1+.05*min(20,max(ledger.epoch-ledger.observed_epoch[p] for p in t))
                         +min(2.,sum(ledger.drift[p] for p in t)/len(t))) for s,t in zip(row,tiles)])
    ranked = sorted(range(len(tiles)),key=lambda j:(-sum(row[j] for row in scores),j))[:pool_tiles]
    pool = tuple(tiles[j] for j in ranked)
    requirements = tuple(tuple(sorted(sorted(range(len(pool)),key=lambda j:(-row[ranked[j]],j))[:requirements_per_candidate])) for row in scores)
    return dict(tiles=pool,requirements=requirements,interaction=interaction)
