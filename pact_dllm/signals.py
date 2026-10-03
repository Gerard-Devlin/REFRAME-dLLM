"""Cheap coarse interaction from normal-forward Q and existing layer4 cache.

No additional Transformer call. Pooled-key softmax, age and last *observed*
drift are planning proxies, not true attention or causal dependencies.
"""
import math
import torch


@torch.no_grad()
def measure(runtime, query, query_positions, candidate_positions, *, tile_width=4, pool_tiles=8, requirements_per_candidate=2):
    ledger = runtime.ledger; n = len(ledger.tokens)
    known = [i for i in range(n) if ledger.tokens[i] != 126336 and i not in ledger.dirty()]
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
    centers = torch.stack([keys[list(t)].float().mean(0) for t in tiles])
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
