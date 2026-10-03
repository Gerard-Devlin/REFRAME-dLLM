"""Train-free experimental decoder, with paid clean repair and DAG commits."""
from dataclasses import dataclass, asdict
import time
import torch
from .engine import Runtime, prepare, probabilities
from .graph import DAG, build_dag
from .planner import joint_plan
from .signals import measure

MASK, EOS = 126336, 126081
VARIANTS = ('reference','cache_only','decode_only','joint')


@dataclass(frozen=True)
class Config:
    length: int = 256
    block: int = 32
    threshold: float = .9
    verify_threshold: float = .8
    candidates: int = 16
    max_parents: int = 2
    max_ancestors: int = 8
    price: float = .12
    tile_width: int = 4
    pool_tiles: int = 8
    requirements_per_candidate: int = 2

    def validate(self):
        if (self.length <= 0 or self.block != 32 or self.length % self.block
                or not 0 < self.threshold <= 1 or not 0 < self.verify_threshold <= 1
                or not 1 <= self.candidates <= 16 or self.tile_width != 4
                or not 0 <= self.pool_tiles <= 8 or self.requirements_per_candidate < 0
                or not 0 <= self.max_parents <= 4 or self.max_ancestors < 0 or self.price < 0):
            raise ValueError('Unsupported preregistered geometry/probability configuration')


def plan_cycle(variant, signals, confidence, config):
    if variant not in VARIANTS:
        raise ValueError('Unknown ablation')
    count = len(confidence)
    dag = (build_dag(signals['interaction'],max_parents=config.max_parents,max_ancestors=config.max_ancestors)
           if variant in ('decode_only','joint') else DAG(tuple(() if i == 0 else (i-1,) for i in range(count))))
    if variant in ('cache_only','joint'):
        plan = joint_plan(dag,[p*p for p in confidence],signals['requirements'],
                          [len(t) for t in signals['tiles']],price=config.price)
    else:
        plan = dict(candidates=tuple(range(count)),tiles=tuple(range(len(signals['tiles']))),objective=None)
    return dag,plan


@torch.no_grad()
def generate(model, prompt, rotary_factory, config=Config(), *, variant='joint', forbidden=(), trace=False, audit=None):
    config.validate()
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    prompt = tuple(map(int,prompt)); device = model.device
    if not prompt or MASK in prompt:
        raise ValueError('A nonempty legal prompt, without generation masks, is required')
    end = len(prompt)+config.length; padded = ((end+127)//128)*128
    canvas = list(prompt)+( [MASK]*config.length )+[EOS]*(padded-end)
    torch.cuda.synchronize(); started = time.perf_counter()
    rotary = rotary_factory(padded,model.config.d_model//model.config.n_heads,model.config.rope_theta,device)
    runtime = Runtime(model,rotary)
    stats = dict(prefill_calls=1,proposal_calls=0,verify_calls=0,seed_commits=0,verified_commits=0,
                 draft_proposed=0,draft_selected=0,own_passes=0,dependency_vetoes=0,
                 optional_refresh_rows=0,mandatory_refresh_rows=0,clean_mask_rows=0,
                 queries=0,planner_seconds=0.,signal_seconds=0.,repair_seconds=0.)
    # The direct-block runtime bypasses model.forward. Count layer executions
    # independently; private perturbation controls use a separate callback.
    attention_executions=[0]
    from . import engine
    def counted_attention(*a,**kw):
        attention_executions[0]+=1
        return engine.streaming(*a,**kw)
    runtime.attention=counted_attention
    actions=[]; warm=runtime.prefill(canvas,range(len(prompt),len(prompt)+config.block))
    stats['queries'] += warm['queries']
    first = True
    for block_start in range(len(prompt),end,config.block):
        block_end = block_start+config.block
        while True:
            active = tuple(i for i in range(block_start,block_end) if runtime.ledger.tokens[i] == MASK)
            if not active:
                break
            dirty = runtime.ledger.dirty()
            if first:
                result = warm; first = False
            else:
                clean = tuple(sorted(set(active+dirty)))
                rows = prepare(runtime.ledger.tokens,clean,rotary,device)
                locate = {p:i for i,p in enumerate(clean)}
                rows.consume=torch.tensor([locate[p] for p in active],device=device)
                result=runtime.run(rows); stats['proposal_calls'] += 1;stats['queries'] += result['queries']
                torch.cuda.synchronize(); tick=time.perf_counter()
                runtime.promote_clean(result); torch.cuda.synchronize()
                stats['repair_seconds'] += time.perf_counter()-tick
            runtime.check_repaired()
            stats['mandatory_refresh_rows'] += len(dirty)
            p=probabilities(result['logits']); confidence,tokens=p.max(-1)
            # Never reveal MASK, even if it wins a malformed state.
            eligible = tokens != MASK
            seed = (confidence >= config.threshold)&eligible
            if not seed.any():
                if not eligible.any():
                    raise ValueError('Model proposed only MASK; no silent schedule change')
                best=confidence.masked_fill(~eligible,-1).argmax(); seed[best]=True
            seed_indices=seed.nonzero().flatten().cpu().tolist()
            seed_positions=tuple(active[i] for i in seed_indices)
            seed_tokens=tuple(int(tokens[i]) for i in seed_indices)
            # Drafts are proposed from this legal normal state, before seed commit.
            avoid = set(forbidden)|{MASK,EOS}
            ranking=confidence.argsort(descending=True,stable=True).cpu().tolist()
            chosen=[i for i in ranking if i not in seed_indices and int(tokens[i]) not in avoid][:config.candidates]
            cp=tuple(active[i] for i in chosen);ct=tuple(int(tokens[i]) for i in chosen)
            scores=tuple(float(confidence[i]) for i in chosen)
            runtime.commit(seed_positions,seed_tokens);stats['seed_commits'] += len(seed_positions)
            entry=dict(block=block_start-len(prompt),seed_positions=list(seed_positions),seed_tokens=list(seed_tokens),
                       candidate_positions=list(cp),candidate_tokens=list(ct),accepted_positions=[],accepted_tokens=[])
            if cp:
                torch.cuda.synchronize(); tick=time.perf_counter()
                signals=measure(runtime,result['query'],active,cp,tile_width=config.tile_width,
                                pool_tiles=config.pool_tiles,requirements_per_candidate=config.requirements_per_candidate)
                torch.cuda.synchronize();stats['signal_seconds'] += time.perf_counter()-tick
                tick=time.perf_counter(); dag,plan=plan_cycle(variant,signals,scores,config)
                stats['planner_seconds'] += time.perf_counter()-tick
                selected=plan['candidates'];stats['draft_proposed'] += len(cp);stats['draft_selected'] += len(selected)
                if audit is not None:
                    # Read-only hypothetical full proposal graph: test isolation
                    # even if the planner chooses an empty execution set.
                    probe_masks=tuple(i for i in range(block_start,block_end) if runtime.ledger.tokens[i]==MASK)
                    probe_clean=tuple(sorted(set(runtime.ledger.dirty()+probe_masks)))
                    probe_ready=prepare(runtime.ledger.tokens,probe_clean,rotary,device,
                        candidate_positions=cp,candidate_tokens=ct,dag=dag)
                    audit(runtime,probe_ready,dag)
                if selected:
                    subdag=dag.subset(selected)
                    positions=tuple(cp[i] for i in selected);proposals=tuple(ct[i] for i in selected)
                    optional=tuple(sorted({p for j in plan['tiles'] for p in signals['tiles'][j]}))
                    mandatory=runtime.ledger.dirty()
                    masks=tuple(i for i in range(block_start,block_end) if runtime.ledger.tokens[i]==MASK)
                    clean=tuple(sorted(set(optional+mandatory+masks)))
                    ready=prepare(runtime.ledger.tokens,clean,rotary,device,candidate_positions=positions,
                                  candidate_tokens=proposals,dag=subdag)
                    verify=runtime.run(ready);stats['verify_calls'] += 1;stats['queries'] += verify['queries']
                    vp=probabilities(verify['logits'])
                    proposed=torch.tensor(proposals,device=device)
                    prob=vp.gather(1,proposed[:,None]).flatten()
                    passes=((verify['logits'].argmax(-1)==proposed)&(prob>=config.verify_threshold)).cpu().tolist()
                    accepted=subdag.closed_accept(passes)
                    paths=subdag.label_paths()
                    accepted_bits=sum(1<<i for i in accepted)
                    assert all(paths[len(selected)+i]&~accepted_bits==0 for i in accepted)
                    torch.cuda.synchronize();tick=time.perf_counter()
                    runtime.promote_clean(verify);torch.cuda.synchronize()
                    stats['repair_seconds'] += time.perf_counter()-tick
                    runtime.check_repaired()  # Seed identities repaired; no draft cache promoted.
                    ap=tuple(positions[i] for i in accepted);av=tuple(proposals[i] for i in accepted)
                    runtime.commit(ap,av)
                    stats['verified_commits'] += len(ap);stats['own_passes'] += sum(passes)
                    stats['dependency_vetoes'] += sum(passes)-len(ap)
                    stats['optional_refresh_rows'] += len(set(optional)-set(mandatory)-set(masks))
                    stats['mandatory_refresh_rows'] += len(mandatory);stats['clean_mask_rows'] += len(masks)
                    entry.update(accepted_positions=list(ap),accepted_tokens=list(av),selected=list(selected),
                                 parents=[list(x) for x in subdag.parents],passes=passes,probabilities=prob.cpu().tolist(),
                                 refresh_rows=len(clean),required_tiles=list(plan['tiles']),proxy_objective=plan['objective'])
            if trace:
                actions.append(entry)
            assert stats['seed_commits']+stats['verified_commits'] <= config.length
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    generated=runtime.ledger.tokens[len(prompt):end]
    assert MASK not in generated
    assert stats['prefill_calls']+stats['proposal_calls']+stats['verify_calls']==runtime.nfe
    assert attention_executions[0]==runtime.nfe*len(model.model.transformer.blocks)
    assert stats['seed_commits']+stats['verified_commits']==config.length
    return dict(token_ids=generated,seconds=seconds,nfe=runtime.nfe,attention_calls=attention_executions[0],stats=stats,actions=actions,
                config=asdict(config),variant=variant,
                scope='Approximate train-free sampler; fixed work, no early EOS shortcut or losslessness claim')
