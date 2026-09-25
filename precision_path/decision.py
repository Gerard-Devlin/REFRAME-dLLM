"""Native actions and a fixed low-result-only ranking score (not an error bound)."""
import torch


def active(state,mask_id=151665,small=8):
    positions=(state==mask_id).nonzero().flatten()
    if not len(positions):
        raise ValueError('No active MASK')
    start=int(positions[0])//small*small
    return start,state[start:start+small]==mask_id


def choose(tokens,probabilities,mask,threshold):
    confidence=probabilities.gather(-1,tokens[:,None]).squeeze(-1)
    scores=torch.where(mask,confidence,-torch.inf)
    selected=(scores>threshold)&mask
    if mask.any():
        selected[scores.argmax()]=True
    return selected,confidence


def action_list(tokens,selected,offset=0):
    return [(int(i)+offset,int(tokens[i])) for i in selected.nonzero().flatten()]


def stability_score(logits,mask,selected,threshold):
    """Ideal-logit radius used ONLY as a conformal ranking feature.

    Includes identities of committed tokens, threshold crossings of every
    remaining token, and forced-position competition when none crosses tau.
    BF16 softmax/rounding is NOT certified by this analytic float32 score.
    """
    if not 0<threshold<1:
        raise ValueError('Score requires threshold strictly between zero and one')
    z=logits.float()
    values=z.topk(2,dim=-1).values
    token_margin=(values[:,0]-values[:,1])/2
    p=z.softmax(-1).amax(-1).clamp(1e-7,1-1e-7)
    odds=torch.logit(p)
    tau=torch.logit(torch.as_tensor(threshold,device=z.device))
    radius=torch.minimum(token_margin[selected].amin(),((odds[mask]-tau).abs()/2).amin())
    if not bool((p[mask]>threshold).any()) and int(mask.sum())>1:
        top=odds[mask].topk(2).values
        radius=torch.minimum(radius,(top[0]-top[1])/4)
    return radius.clamp_min(0)


def cache_tensors(cache):
    if cache is None:
        return []
    return [(cache[i][0],cache[i][1]) for i in range(len(cache))]


def snapshot_cache(cache):
    return [(k.clone(),v.clone()) for k,v in cache_tensors(cache)]


def assert_cache_unchanged(cache,saved):
    actual=cache_tensors(cache)
    if len(actual)!=len(saved) or any(not torch.equal(a,b)
            for pair,old in zip(actual,saved) for a,b in zip(pair,old)):
        raise AssertionError('Low precision mutated persistent BF16 KV; stop')
