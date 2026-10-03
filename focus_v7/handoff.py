"""CPU geometry for a two-whole-state diagnostic; no online decoder."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Update:
    positions: tuple[int,...]
    tokens: tuple[int,...]


def transition(positions,probabilities,tokens,*,threshold=.9,mask_id=126336):
    if (not positions or len(set(positions))!=len(positions)
            or not len(positions)==len(probabilities)==len(tokens) or not 0<threshold<=1):
        raise ValueError('invalid transition geometry')
    if any(not math.isfinite(p) or not 0<=p<=1 for p in probabilities):
        raise ValueError('invalid confidence')
    legal=[i for i,t in enumerate(tokens) if t!=mask_id]
    if not legal:raise ValueError('no legal transition')
    chosen=[i for i in legal if probabilities[i]>=threshold]
    if not chosen:chosen=[max(legal,key=lambda i:(probabilities[i],-positions[i]))]
    chosen.sort(key=lambda i:positions[i])
    return Update(tuple(int(positions[i]) for i in chosen),tuple(int(tokens[i]) for i in chosen))


def insert(window,ids,update,*,mask_id=126336):
    if (len(window)!=len(ids) or len(set(window))!=len(window)
            or len(update.positions)!=len(update.tokens)
            or len(set(update.positions))!=len(update.positions)):
        raise ValueError('invalid state/update')
    result=list(ids);index={p:i for i,p in enumerate(window)}
    for p,t in zip(update.positions,update.tokens):
        if p not in index or result[index[p]]!=mask_id or t==mask_id:
            raise ValueError('not an uncommitted legal position')
        result[index[p]]=int(t)
    return result


def build_call(current,tracked,window,update=None,*,paired=False):
    """[W0,T0,W1,T1], each local64 tile has one physical key version."""
    if len(tracked)!=32 or len(window)!=32 or len(set(tracked+window))!=64:
        raise ValueError('fixed64-row private branch required')
    state=current['state'];canvas=current['canvas'];device=canvas.device
    if state['active_batch']!=[0] or int(state['block_m'])!=32:
        raise ValueError('pinned batch1/block32 only')
    key_length=int(state['seqlen_k'][0]);private=list(window)+list(tracked)
    if any(p<0 or p>=key_length for p in private):raise ValueError('uninitialized position')
    window_ids=canvas[torch.tensor(window,device=device)].tolist()
    tracked_ids=canvas[torch.tensor(tracked,device=device)].tolist()
    if any(v!=126336 for v in window_ids) or any(v==126336 for v in tracked_ids):
        raise ValueError('invalid clean query identities')
    branch=window_ids if update is None else insert(window,window_ids,update)
    external=[p for p in range(key_length) if p not in set(private)]
    ids=window_ids+tracked_ids+branch+tracked_ids if paired else branch+tracked_ids
    qpos=private+private if paired else private
    width=128 if paired else 64
    blocks=torch.tensor([[0,len(external),s,s+64] for s in range(0,width,64)],
                        device=device,dtype=torch.int32)
    # Kernel stride is64 per tile, NOT a128x128 global mask.
    mask=torch.ones((width,64),device=device,dtype=torch.bool)
    positions=[torch.tensor(qpos,device=device),torch.tensor(external,device=device,dtype=torch.long),
        state['rotary_emb_pos'],state['info'],state['attn_scores'],mask]
    lengths=[list(state['start_layer']),blocks,None,state['query_tracked_blocks'],None,
        list(state['active_batch']),int(state['num_active']),int(state['max_length']),
        32,int(state['block_n']),state['elastic_cache'],True]
    return torch.tensor([ids],device=device),positions,lengths


def summarize(rows,timing):
    matches=sum(r['proposal_matches_fresh_update'] for r in rows)
    action_agreement=all(r['packed_actions_equal'] for r in rows)
    valid=sum(r['proposal_matches_fresh_update'] and r['packed_actions_equal'] for r in rows)
    fraction=valid/len(rows) if rows else 0
    ceilings=[(1+fraction)*r['single_mean_seconds']/r['packed_mean_seconds'] for r in timing]
    return dict(windows=len(rows),exact_proposed_updates=matches,
        exact_fraction=matches/len(rows) if rows else 0,all_packed_actions_equal=action_agreement,
        usable_fraction=fraction,optimistic_opportunity_ratios=ceilings,
        gate_passed=len(rows)>=16 and matches/len(rows)>=.5 and action_agreement
            and len(ceilings)==2 and min(ceilings)>1.15,
        scope='Same frozen-bank conditional operator; diagnostic ceiling, not online acceleration or quality.')
