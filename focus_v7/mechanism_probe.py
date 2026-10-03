"""Frozen v7 trajectory audit: admission versus paid external-cache freshness."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics as python_stats
import time
from types import SimpleNamespace

import torch

from focus_dllm.tuning.flash_readout import ModelReadout, suppress_official_prints
from focus_v6.audit_probe import forbid_sdpa
from .cache import capture_private_kv
from .generation import generate
from .greedy import decide
from .mechanism import full_call, instrument, private_cache, statistics


def logit_error(a,b):
    delta=a.float()-b.float();delta[:,126336]=0
    if not bool(torch.isfinite(delta).all()):raise AssertionError('nonfinite shadow logits')
    return dict(max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()))


def summarize(rows):
    admitted=[(p,token['kind']) for row in rows for token in row['commits'] for p in [token['probability']]]
    result=dict(packets=len(rows),commits=len(admitted),admission={})
    for kind in ('accepted_draft','correction'):
        values=[p for p,k in admitted if k==kind]
        result['admission'][kind]=dict(count=len(values),below_05=sum(p<.5 for p in values),
            below_09=sum(p<.9 for p in values),mean_probability=python_stats.mean(values) if values else None)
    shadows=[r for r in rows if 'shadow' in r]
    def agreement(field):
        pairs=[(a,b) for r in shadows for a,b in zip(r['shadow'][field[0]],r['shadow'][field[1]])]
        return dict(matches=sum(a==b for a,b in pairs),positions=len(pairs),
                    fraction=sum(a==b for a,b in pairs)/len(pairs) if pairs else None)
    result['shadows']=dict(windows=len(shadows),
        old_clean_vs_full=agreement(('old_clean_top1','full_clean_top1')),
        fresh_clean_vs_full=agreement(('fresh_clean_top1','full_clean_top1')),
        old_audit_vs_fresh_audit=agreement(('old_audit_top1','fresh_audit_top1')),
        old_actual_commit_vs_full=agreement(('old_committed_tokens','full_at_committed')),
        mean_audit_logit_rms=python_stats.mean(r['shadow']['audit_logit_error']['rms'] for r in shadows) if shadows else None,
        old_fresh_decision_changes=sum(not r['shadow']['same_decision'] for r in shadows),
        scope='Paid same-state shadows; full MASK top1 is not gold or sequential verification equivalence')
    return result


@torch.no_grad()
def main():
    from focus_dllm.common import sha256,write_json
    from focus_dllm.llada_common import MODEL_ID,REVISION,prompt_ids
    from focus_dllm.tuning.competitors import generation_prompt,load_external,load_model,select_samples
    from focus_dllm.tuning.gpu_contract import check_binding
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--third-party',type=Path,required=True)
    parser.add_argument('--datasets',type=Path,nargs=3,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    reference=json.loads(args.reference.read_text())
    saved={(r['task'],str(r['id'])):r['outputs']['focus_v7'] for r in reference['records']}
    cls,external,adaptation=load_external(args.third_party,'flash_verify')
    source=Path(inspect.getsourcefile(inspect.unwrap(external)))
    assert sha256(source)==reference['official_sha256']
    raw,tokenizer=load_model(SimpleNamespace(method='flash_verify'),cls)
    compact=ModelReadout(raw,compact=True,minimum=32)
    torch.set_num_threads(1);torch.manual_seed(1234)
    report=dict(model=MODEL_ID,revision=REVISION,binding=check_binding(required=True),
        implementation={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        datasets={str(p):sha256(p) for p in args.datasets},reference_sha256=sha256(args.reference),
        official_sha256=sha256(source),adaptation=adaptation,
        config=dict(length=256,k=16,shadow_packet_indices=[0,1,4,8],prompts_per_task=2,
                    probability_bands=[.5,.9],seed=51713,offset=0),
        records=[],controls=[],prompts=[],
        scope='Reused development prompts. Only original v7 owns online state. No new candidate generation/accuracy claim.')
    write_json(args.output/'diagnostic.json',report)
    context={};window_count=[0];calls=[0];shadow_calls=[0];in_shadow=[False]
    def count(_model,_inputs):
        calls[0]+=1
        if in_shadow[0]:shadow_calls[0]+=1
    handle=raw.register_forward_pre_hook(count)

    def packet(current):
        layout=current['layout']
        with capture_private_kv() as kv:
            value=compact(current['query'],use_cache=True,positions=current['positions'],
                          lengths=current['lengths'],focus_head_rows=(layout.clean.start,3*layout.candidates))
        logits=value.logits.squeeze(0)
        return logits[layout.clean],logits[layout.audit],kv

    def observe(current):
        index=len(current['packets']);layout=current['layout']
        candidates=current['candidates'];decision=current['decision'];drafts=current['drafts']
        old_clean=statistics(current['clean'])
        old_audit=statistics(current['audit'],drafts)
        positions=candidates
        commits=[]
        for i,token in enumerate(decision.tokens):
            commits.append(dict(position=positions[i],token=token,
                kind='accepted_draft' if i<decision.accepted else 'correction',
                probability=old_audit['confidence'][i],
                clean_token=old_clean['top1'][i],clean_confidence=old_clean['confidence'][i]))
        row=dict(**context,packet=index,candidates=list(positions),drafts=current['drafts_cpu'],
                 accepted=decision.accepted,commits=commits,old_clean=old_clean,old_audit=old_audit)
        if index in (0,1,4,8):
            blocks=raw.model.transformer.blocks
            saved_objects=[(b.k_cache,b.v_cache) for b in blocks]
            versions=[(k._version,v._version) for k,v in saved_objects]
            edges=[(b.k_cache.clone(),b.v_cache.clone()) for b in (blocks[0],blocks[-1])]
            adapter_current=compact.current
            in_shadow[0]=True
            try:
                if window_count[0]==0:
                    repeated_clean,repeated_audit,repeated_kv=packet(current)
                    error=logit_error(current['audit'],repeated_audit)
                    clean_error=logit_error(current['clean'],repeated_clean)
                    kv_error=max(float((a.float()-b.float()).abs().max())
                        for old,new in zip(current['captured'],repeated_kv) for a,b in zip(old,new))
                    if error['max_abs']!=0 or clean_error['max_abs']!=0 or kv_error!=0:
                        raise AssertionError('old packet repetition differs')
                    report['controls'].append(dict(**context,audit_error=error,clean_error=clean_error,kv_error=kv_error))
                with private_cache(blocks):
                    query,pos,lengths=full_call(current,positions)
                    torch.cuda.synchronize();started=time.perf_counter()
                    full=compact(query,use_cache=True,positions=pos,lengths=lengths,
                                 focus_head_rows=(0,layout.candidates))
                    torch.cuda.synchronize();full_seconds=time.perf_counter()-started
                    full_stats=statistics(full.logits.squeeze(0)[:layout.candidates])
                    fresh_clean,fresh_audit,_=packet(current)
                    fresh_c=statistics(fresh_clean);fresh_a=statistics(fresh_audit,drafts)
                    fresh_decision=decide(fresh_a['candidate_probability'],fresh_a['top1'],current['drafts_cpu'],
                                          forbidden=(126336,))
                    row['shadow']=dict(old_clean_top1=old_clean['top1'],full_clean_top1=full_stats['top1'],
                        full_clean_confidence=full_stats['confidence'],fresh_clean_top1=fresh_c['top1'],
                        fresh_clean_confidence=fresh_c['confidence'],old_audit_top1=old_audit['top1'],
                        fresh_audit_top1=fresh_a['top1'],fresh_audit=fresh_a,
                        audit_logit_error=logit_error(current['audit'],fresh_audit),
                        fresh_clean_vs_full_matches=[a==b for a,b in zip(fresh_c['top1'],full_stats['top1'])],
                        old_committed_tokens=list(decision.tokens),full_at_committed=full_stats['top1'][:decision.progress],
                        fresh_tokens=list(fresh_decision.tokens),fresh_accepted=fresh_decision.accepted,
                        same_decision=fresh_decision==decision,paid_full_forward_seconds=full_seconds,
                        cost_scope='Private diagnostic timing includes first-use compilation; not an inference baseline')
                window_count[0]+=1
            finally:
                in_shadow[0]=False;compact.current=adapter_current
            if any(b.k_cache is not k or b.v_cache is not v for b,(k,v) in zip(blocks,saved_objects)):
                raise AssertionError('cache object restoration failed')
            if versions!=[(b.k_cache._version,b.v_cache._version) for b in blocks]:
                raise AssertionError('observer mutated live cache')
            for (k,v),b in zip(edges,(blocks[0],blocks[-1])):
                if not torch.equal(k,b.k_cache) or not torch.equal(v,b.v_cache):
                    # The uninitialized allocation beyond seqlen can contain NaN.
                    if not torch.allclose(k,b.k_cache,rtol=0,atol=0,equal_nan=True) or not torch.allclose(v,b.v_cache,rtol=0,atol=0,equal_nan=True):
                        raise AssertionError('live cache contents changed')
            row['shadow']['live_cache_unchanged']=True
        report['records'].append(row)

    traced=instrument(generate,observe)
    try:
        for task,path in zip(('humaneval','mbpp','math'),args.datasets):
            for sample in select_samples(path,2,0):
                ident=str(sample.get('id',sample.get('task_id')))
                context.clear();context.update(task=task,id=ident)
                window_count[0]=0;calls[0]=0;shadow_calls[0]=0
                ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                with forbid_sdpa(),suppress_official_prints():
                    value=traced(compact,tokenizer,external,ids,length=256)
                ref=saved[(task,ident)]
                if (value['text']!=ref['text'] or value['raw_token_ids']!=ref['raw_token_ids']
                        or value['packets']!=ref['packets'] or value['nfe']!=ref['nfe']):
                    raise AssertionError('read-only diagnostic changed frozen v7 trajectory')
                assert calls[0]==value['nfe']+shadow_calls[0]
                report['prompts'].append(dict(**context,online_nfe=value['nfe'],shadow_calls=shadow_calls[0],
                    windows=window_count[0],sdpa_calls=0,text_sha256=hashlib.sha256(value['text'].encode()).hexdigest(),
                    complete_tokens_actions_text_nfe_unchanged=True))
                write_json(args.output/'diagnostic.json',report)
                print('PROMPT',task,ident,'windows',window_count[0],'online',value['nfe'],'shadow',shadow_calls[0],flush=True)
        report['summary']=summarize(report['records'])
        assert report['implementation']=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
        assert sha256(source)==report['official_sha256']
        write_json(args.output/'diagnostic.json',report)
        (args.output/'complete').write_text('OK\n')
        print('FINAL',json.dumps(report['summary']),flush=True)
    finally:handle.remove()


if __name__=='__main__':main()
