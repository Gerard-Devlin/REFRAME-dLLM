"""Reallocate a fixed64-query cache-producing packet to clean coverage.

Private replica labels remain all-layer isolated. This changes the approximate
conditional program, not the original model or a mathematical lossless runtime.
"""
import ast
from dataclasses import dataclass
import inspect
import math
import textwrap

import torch

from .generation import generate, select_tracked
from .revision import Commit


@dataclass(frozen=True)
class BudgetLayout:
    clean_count: int
    candidates: int
    width: int = 64

    def __post_init__(self):
        if not (1 <= self.candidates <= self.clean_count <= 32
                and self.candidates <= 16 and self.tracked >= 16):
            raise ValueError('invalid clean/audit query budget')

    @property
    def tracked(self):return self.width-self.clean_count-2*self.candidates

    @property
    def clean(self):return slice(self.tracked,self.tracked+self.clean_count)

    @property
    def draft(self):return slice(self.clean.stop,self.clean.stop+self.candidates)

    @property
    def audit(self):return slice(self.draft.stop,self.width)


def budget_mask(layout):
    t,m,k,w=layout.tracked,layout.clean_count,layout.candidates,layout.width
    mask=torch.zeros((w,w),dtype=torch.bool)
    mask[:t+m,:t+m]=True
    i,j=torch.arange(k)[:,None],torch.arange(k)[None,:]
    mask[layout.draft,:t]=True
    mask[layout.audit,:t]=True
    mask[layout.draft,layout.clean.start+k:layout.clean.stop]=True
    mask[layout.audit,layout.clean.start+k:layout.clean.stop]=True
    mask[layout.draft,layout.clean.start:layout.clean.start+k]=j>i
    mask[layout.audit,layout.clean.start:layout.clean.start+k]=j>i
    mask[layout.draft,layout.draft]=j<=i
    mask[layout.audit,layout.draft]=j<i
    mask[layout.audit,layout.audit]=torch.eye(k,dtype=torch.bool)
    return mask


def build_budget_call(state,k,clean_count):
    if state['active_batch']!=[0] or int(state['block_m'])!=32:
        raise ValueError('only pinned batch1/block32 is supported')
    layout=BudgetLayout(clean_count,k)
    decoded=int(state['num_decoded'][0]);full=state['full_pos'][0]
    if decoded<layout.tracked:raise ValueError('not enough committed background')
    tracked=full[decoded-layout.tracked:decoded]
    clean=full[decoded:decoded+clean_count]
    drafts=state['x_draft'][clean[:k]]
    clean_ids=state['x'][clean]
    if not bool((clean_ids==int(state['mask_id'])).all()):
        raise ValueError('clean candidate is already committed')
    external=torch.cat((full[:decoded-layout.tracked],full[decoded+clean_count:int(state['seqlen_k'][0])]))
    private_positions=torch.cat((tracked,clean))
    if (len(torch.unique(private_positions))!=len(private_positions)
            or bool(torch.isin(external,private_positions).any())
            or len(torch.unique(external))!=len(external)):
        raise ValueError('duplicate physical key version')
    query=torch.cat((state['x'][tracked],clean_ids,drafts,clean_ids[:k])).unsqueeze(0)
    qpos=torch.cat((tracked,clean,clean[:k],clean[:k]))
    blocks=torch.tensor([[0,external.numel(),0,layout.width]],device=query.device,dtype=torch.int32)
    positions=[qpos,external,state['rotary_emb_pos'],state['info'],state['attn_scores'],
               budget_mask(layout).to(query.device)]
    lengths=[list(state['start_layer']),blocks,None,state['query_tracked_blocks'],None,
             list(state['active_batch']),int(state['num_active']),int(state['max_length']),
             32,int(state['block_n']),state['elastic_cache'],True]
    return query,positions,lengths,layout,clean,drafts


def budget_commit(probabilities,top1,drafts,*,clean_p,clean_ids,capacity,forbidden=(),threshold=.9,gamma=.8):
    m,k=len(clean_p),len(drafts);banned=set(forbidden)
    if (not 1<=k<=m or len(clean_ids)!=m or len(probabilities)!=k or len(top1)!=k
            or capacity<k or not 0<threshold<=1 or not 0<gamma<=1):
        raise ValueError('invalid admission/capacity geometry')
    if any(not math.isfinite(p) or not 0<=p<=1 for p in list(clean_p)+list(probabilities)):
        raise ValueError('nonfinite/invalid probability')
    safe=[i for i in range(m) if clean_p[i]>=threshold and clean_ids[i] not in banned]
    prefix=0;budget=1.
    for i in range(k):
        budget*=probabilities[i]
        if (drafts[i] in banned or top1[i]!=drafts[i] or budget<gamma
                or (i in safe and clean_ids[i]!=drafts[i])):break
        prefix+=1
    roots=[i for i in safe if i>=prefix]
    if len(roots)>capacity-prefix:
        roots=sorted(roots,key=lambda i:(-clean_p[i],i))[:capacity-prefix]
    indices=list(range(prefix))+roots
    tokens=list(drafts[:prefix])+[clean_ids[i] for i in roots]
    kind='verified_prefix_and_clean_roots'
    if not indices:
        legal=[i for i in range(m) if clean_ids[i] not in banned]
        if not legal:raise ValueError('no legal clean fallback')
        i=max(legal,key=lambda i:(clean_p[i],-i))
        indices=[i];tokens=[clean_ids[i]];kind='clean_fallback'
    return Commit(prefix,tuple(indices),tuple(map(int,tokens)),kind)


def budget_promotion(layout,clean,tracked,decision):
    clean,tracked=tuple(map(int,clean)),tuple(map(int,tracked))
    if (len(clean)!=layout.clean_count or len(tracked)!=layout.tracked
            or len(set(clean+tracked))!=len(clean)+len(tracked)
            or not 0<=decision.accepted<=layout.candidates
            or len(set(decision.indices))!=decision.progress
            or decision.indices[:decision.accepted]!=tuple(range(decision.accepted))
            or any(i<0 or i>=layout.clean_count for i in decision.indices)
            or decision.progress>min(16,layout.tracked)):
        raise ValueError('invalid promotion/identity repair geometry')
    dirty=tuple(clean[i] for i in decision.indices[decision.accepted:])
    rows=list(range(layout.tracked));dest=list(tracked)
    for i,p in enumerate(clean):
        if p in dirty:continue
        rows.append(layout.draft.start+i if i<decision.accepted else layout.clean.start+i)
        dest.append(p)
    return tuple(rows),tuple(dest),dirty


class TrackedScheduler:
    def __init__(self,policy):
        if policy not in ('recent','age'):raise ValueError('invalid scheduler')
        self.policy=policy;self.last={};self.step=0;self.history=[]

    def select(self,known,changed,count):
        if len(set(known))!=len(known) or not set(changed).issubset(known):
            raise ValueError('invalid clean committed context')
        mandatory=list(dict.fromkeys(changed))
        if len(mandatory)>count:raise ValueError('repair exceeds query budget')
        if self.policy=='recent':chosen=select_tracked(known,changed,count)
        else:
            rest=sorted((p for p in known if p not in set(mandatory)),key=lambda p:(self.last.get(p,0),p))
            chosen=mandatory+rest[:count-len(mandatory)]
            if len(chosen)!=count:raise ValueError('insufficient background')
        self.history.append(dict(step=self.step,tracked=chosen,
            max_age=max(self.step-self.last.get(p,0) for p in chosen)))
        return chosen

    def installed(self,destinations):
        self.step+=1
        self.last.update({int(p):self.step for p in destinations})


@torch.no_grad()
def generate_budget(model,tokenizer,external,ids,*,length=256,clean_limit=32,draft_limit=8,
                    tracking='recent',observer=None):
    if (clean_limit,draft_limit) not in ((16,16),(32,8)):
        raise ValueError('predeclared packet layouts only')
    scheduler=TrackedScheduler(tracking);geometry=[]
    def build(state,k):
        m=active_clean[0]
        result=build_budget_call(state,k,m)
        geometry.append(dict(clean=m,draft=k,tracked=result[3].tracked,query_rows=64))
        return result
    active_clean=[0]
    def layout(m,k):active_clean[0]=m;return BudgetLayout(m,k)
    def admit(probabilities,top1,drafts,*,clean_p,clean_ids,forbidden=()):
        return budget_commit(probabilities,top1,drafts,clean_p=clean_p,clean_ids=clean_ids,
                             capacity=16,forbidden=forbidden)
    def promotion(layout,candidates,tracked,decision):
        rows,dest,dirty=budget_promotion(layout,candidates,tracked,decision)
        scheduler.installed(dest)
        return rows,dest,dirty
    def observe(current):
        if observer is not None:observer(current)
    source=inspect.unwrap(generate);tree=ast.parse(textwrap.dedent(inspect.getsource(source)))
    tree.body[0].decorator_list=[];counts=[0]*9
    class Change(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node);code=ast.unparse(node)
            replacements={
                'candidates = sorted(window, key=lambda p: (-proposals[p][0], p))[:16]':
                    (0,'candidates = sorted(window, key=lambda p: (-proposals[p][0], p))[:_clean_limit]'),
                'k = len(candidates)':(1,'k = min(_draft_limit, len(candidates))'),
                'layout = Layout(k)':(2,'layout = _budget_layout(len(candidates), k)'),
                'drafts_cpu = [proposals[p][1] for p in candidates]':
                    (3,'drafts_cpu = [proposals[p][1] for p in candidates[:k]]'),
                'draft_ids[torch.tensor(candidates, device=raw.device)] = torch.tensor(drafts_cpu, device=raw.device)':
                    (4,'draft_ids[torch.tensor(candidates[:k], device=raw.device)] = torch.tensor(drafts_cpu, device=raw.device)'),
                'committed = candidates[:decision.progress]':
                    (5,'committed = [candidates[i] for i in decision.indices]')}
            if code in replacements:
                i,value=replacements[code];counts[i]+=1
                return ast.copy_location(ast.parse(value).body[0],node)
            if (len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                    and node.targets[0].id=='decision' and isinstance(node.value,ast.Call)
                    and isinstance(node.value.func,ast.Name) and node.value.func.id=='decide'):
                counts[6]+=1;node.value.func.id='_budget_admit'
                for name in ('clean_p','clean_ids'):
                    node.value.keywords.append(ast.keyword(arg=name,value=ast.Name(id=name,ctx=ast.Load())))
                extra=ast.parse('_budget_observe(dict(locals()))').body[0]
                return [node,ast.copy_location(extra,node)]
            return node
        def visit_keyword(self,node):
            if node.arg=='focus_head_rows' and ast.unparse(node.value)=='(layout.clean.start, 3 * k)':
                counts[7]+=1;node.value=ast.parse('(layout.clean.start, layout.width-layout.clean.start)',mode='eval').body
            return self.generic_visit(node)
        def visit_ImportFrom(self,node):
            if node.module=='focus_v6.audit' and any(a.name=='Layout' for a in node.names):
                counts[8]+=1
            return node
    tree=Change().visit(tree)
    if counts!=[1]*9:raise ValueError(f'frozen generator changed: {counts}')
    ast.fix_missing_locations(tree)
    scope=dict(source.__globals__,_clean_limit=clean_limit,_draft_limit=draft_limit,
        _budget_layout=layout,_budget_admit=admit,_budget_observe=observe,
        select_tracked=scheduler.select,build_call=build,promotion_plan=promotion)
    exec(compile(tree,source.__code__.co_filename+':query_budget','exec'),scope)
    result=scope[source.__name__](model,tokenizer,external,ids,length=length)
    result.update(query_geometry=geometry,tracked_history=scheduler.history,
        configuration=dict(clean_limit=clean_limit,draft_limit=draft_limit,tracking=tracking,
                           threshold=.9,gamma=.8,query_width=64),
        approximation='Conditional private audit, expanded clean coverage, approximate background KV')
    return result
