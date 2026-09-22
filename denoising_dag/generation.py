"""Optional full generation using the proposal's bounded search policy.

Commit the complete selected lookahead path. Preserve every search history
within the tree. The memo table lives for one immutable completed-prefix cache.
This decoder is experimental and has no established quality advantage.
"""
import hashlib
import torch
from .backend import MASK_ID,EOS_ID
from .search import Executor,search


@torch.no_grad()
def generate(backend,prompt,reuse,depth=3,width=4,max_new_tokens=128,max_nodes=4096):
    block=backend.block_size
    device=backend.device
    prefix=len(prompt)
    if not prompt or MASK_ID in prompt:
        raise ValueError('Nonempty prompt without MASK required')
    output=list(prompt)
    complete=prefix//block*block
    cache,seed=None,None
    if complete:
        result=backend.call(input_ids=torch.tensor([prompt[:complete]],device=device),
            use_cache=True,update_past_key_values=True,block_size=block,logits_to_keep=1)
        cache=result.past_key_values
        logits=result.logits[:,-1].float()
        logits[:,MASK_ID]=-torch.inf
        seed=int(logits.argmax(-1))
    start=complete
    logical=physical=hits=0
    key_seconds=dispatch_seconds=0.
    decision_hash=hashlib.sha256()
    block_stats=[]
    while len(output)-prefix<max_new_tokens:
        known=output[start:]
        raw=known+[MASK_ID]*(block-len(known))
        if not known:
            if seed is None:
                raise ValueError('No preceding clean block for boundary seed')
            raw[0]=seed
        snap=backend.make_snapshot(raw,cache,f'generate:{start}')
        executor=Executor(snap.context,lambda states:backend.predict(snap,states),reuse,max_nodes)
        while MASK_ID in raw:
            result=search(raw,executor,MASK_ID,depth,width,max_nodes)
            if not result.path:
                raise RuntimeError('Lookahead made no progress')
            raw=list(result.state)
            decision_hash.update(result.decisions.encode())
        logical+=executor.logical_rows
        physical+=executor.physical_rows
        hits+=executor.hits
        key_seconds+=executor.key_seconds
        dispatch_seconds+=executor.dispatch_seconds
        block_stats.append(dict(start=start,logical_rows=executor.logical_rows,
                                physical_rows=executor.physical_rows,hits=executor.hits))
        output=output[:start]+raw
        generated=output[prefix:prefix+max_new_tokens]
        if EOS_ID in generated or len(generated)>=max_new_tokens:
            ended=EOS_ID in generated
            if ended:
                generated=generated[:generated.index(EOS_ID)]
            return dict(tokens=generated,truncated=not ended,decisions=decision_hash.hexdigest(),
                        logical_rows=logical,search_physical_rows=physical,hits=hits,
                        key_seconds=key_seconds,dispatch_seconds=dispatch_seconds,blocks=block_stats)
        # The mutable native cache advances only after the current search ends.
        result=backend.call(input_ids=torch.tensor([raw],device=device),past_key_values=cache,
            use_cache=True,update_past_key_values=True,block_size=block,logits_to_keep=1)
        cache=result.past_key_values
        logits=result.logits[:,-1].float()
        logits[:,MASK_ID]=-torch.inf
        seed=int(logits.argmax(-1))
        start+=block
