"""Separate conservative admission and own-paid frontier refresh ablations.

Original generation.py and all frozen results remain unchanged. This is an
approximate algorithm experiment, not a lossless or novelty claim.
"""
import ast
from dataclasses import dataclass
import inspect
import math
import textwrap

import torch

from .generation import generate
from .greedy import decide as old_decide
from .mechanism import full_call
from .packet import promotion_plan as old_promotion


@dataclass(frozen=True)
class Commit:
    accepted: int
    indices: tuple[int,...]
    tokens: tuple[int,...]
    kind: str

    @property
    def progress(self):return len(self.tokens)

    @property
    def correction(self):return self.tokens[0] if self.kind=='clean_fallback' else None


def select_commit(probabilities,top1,drafts,*,clean_p,clean_ids,forbidden=(),threshold=.9,gamma=.8):
    """Independent clean safe roots + a verified prefix; no forced audit token.

    A conflicting high-confidence clean root stops its draft prefix. Clean roots
    never consume proposals; draft KV only comes from the accepted prefix.
    One highest-confidence CLEAN argmax is the no-progress fallback.
    """
    k=len(drafts);banned=set(forbidden)
    if not k or not all(len(x)==k for x in (probabilities,top1,clean_p,clean_ids)):
        raise ValueError('invalid admission geometry')
    if any(not math.isfinite(p) or not 0<=p<=1 for p in list(probabilities)+list(clean_p)):
        raise ValueError('invalid confidence')
    safe=[i for i in range(k) if clean_p[i]>=threshold and clean_ids[i] not in banned]
    prefix=0;budget=1.
    for i in range(k):
        budget*=probabilities[i]
        if (drafts[i] in banned or top1[i]!=drafts[i] or budget<gamma
                or (i in safe and clean_ids[i]!=drafts[i])):break
        prefix+=1
    roots=[i for i in safe if i>=prefix]
    indices=list(range(prefix))+roots
    tokens=list(drafts[:prefix])+[clean_ids[i] for i in roots]
    kind='verified_prefix_and_clean_roots'
    if not indices:
        valid=[i for i in range(k) if clean_ids[i] not in banned]
        if not valid:raise ValueError('no legal progress token')
        i=max(valid,key=lambda i:(clean_p[i],-i))
        indices=[i];tokens=[clean_ids[i]];kind='clean_fallback'
    return Commit(prefix,tuple(indices),tuple(map(int,tokens)),kind)


def promotion(layout,candidates,tracked,decision):
    if not isinstance(decision,Commit):return old_promotion(layout,candidates,tracked,decision)
    k=layout.candidates
    if len(candidates)!=k or len(tracked)!=layout.tracked or len(set(candidates+tracked))!=k+layout.tracked:
        raise ValueError('invalid physical position geometry')
    if decision.indices[:decision.accepted]!=tuple(range(decision.accepted)):
        raise ValueError('accepted draft versions must form a prefix')
    dirty=tuple(candidates[i] for i in decision.indices[decision.accepted:])
    rows=list(range(layout.tracked));dest=list(tracked)
    for i,p in enumerate(candidates):
        if p in dirty:continue
        rows.append(layout.draft.start+i if i<decision.accepted else layout.clean.start+i)
        dest.append(p)
    return tuple(rows),tuple(dest),dirty


@torch.no_grad()
def generate_revised(model,tokenizer,external,ids,*,length=256,refresh=False,admission=False):
    barriers=[];last_frontier=[0]
    def barrier(current):
        unchanged=(current['warm_hidden'],current['row_for_position'],current['proposals'],current['dirty'])
        frontier=(current['remaining'][0]-current['prompt_length'])//32
        if not refresh or frontier<=last_frontier[0]:return unchanged
        raw=current['raw'];normalized=[None]
        def capture(_module,_args,value):normalized[0]=value.detach()
        handle=raw.model.transformer.ln_f.register_forward_hook(capture)
        query,pos,lengths=full_call(current,current['window'])
        try:
            output=model(query,use_cache=True,positions=pos,lengths=lengths,
                         focus_head_rows=(0,len(current['window'])))
        finally:handle.remove()
        if normalized[0] is None or normalized[0].shape[1]!=query.shape[1]:
            raise AssertionError('missing full own-paid epoch hidden')
        from .generation import valid_predictions
        p,t=valid_predictions(output.logits.squeeze(0)[:len(current['window'])],126336)
        proposals={i:(prob,token) for i,prob,token in zip(current['window'],p,t)}
        rows={p:i for i,p in enumerate(pos[0].tolist())}
        if any(b.k_cache is not k or b.v_cache is not v
               for b,(k,v) in zip(raw.model.transformer.blocks,current['bank'])):
            raise AssertionError('epoch refresh replaced committed cache storage')
        last_frontier[0]=frontier
        barriers.append(dict(frontier=frontier,full_query_rows=query.shape[1]))
        return normalized[0][0],rows,proposals,()

    def admit(probabilities,top1,drafts,*,clean_p,clean_ids,forbidden=()):
        if not admission:return old_decide(probabilities,top1,drafts,forbidden=forbidden)
        return select_commit(probabilities,top1,drafts,clean_p=clean_p,clean_ids=clean_ids,forbidden=forbidden)

    def committed(candidates,decision):
        return [candidates[i] for i in decision.indices] if isinstance(decision,Commit) else candidates[:decision.progress]

    source=inspect.unwrap(generate);tree=ast.parse(textwrap.dedent(inspect.getsource(source)))
    tree.body[0].decorator_list=[];counts=[0,0,0]
    class Change(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            if ast.unparse(node)=='window = remaining[:32]':
                counts[0]+=1
                extra=ast.parse('warm_hidden, row_for_position, proposals, dirty = _barrier(dict(locals()))').body[0]
                return [node,ast.copy_location(extra,node)]
            if (len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                    and node.targets[0].id=='decision' and isinstance(node.value,ast.Call)
                    and isinstance(node.value.func,ast.Name) and node.value.func.id=='decide'):
                counts[1]+=1;node.value.func.id='_admit'
                for name in ('clean_p','clean_ids'):
                    node.value.keywords.append(ast.keyword(arg=name,value=ast.Name(id=name,ctx=ast.Load())))
            if ast.unparse(node)=='committed = candidates[:decision.progress]':
                counts[2]+=1;node.value=ast.parse('_committed(candidates,decision)',mode='eval').body
            return node
    tree=Change().visit(tree)
    if counts!=[1,1,1]:raise ValueError('frozen generator structure changed')
    ast.fix_missing_locations(tree)
    scope=dict(source.__globals__,_barrier=barrier,_admit=admit,_committed=committed,promotion_plan=promotion)
    exec(compile(tree,source.__code__.co_filename+':conservative_frontier_ablation','exec'),scope)
    result=scope[source.__name__](model,tokenizer,external,ids,length=length)
    result['nfe']+=len(barriers)
    result['epoch_barriers']=barriers
    result['configuration']=dict(refresh=refresh,admission=admission,threshold=.9,gamma=.8,frontier_width=32)
    result['approximation']='Paid cache epochs and conditional audit; not native-equivalent'
    return result
