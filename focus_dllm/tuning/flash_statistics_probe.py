"""Consumed 64-row head + full-vocabulary FP64 sufficient-statistics check.

Official Flash decoder/cache/commit/verification/stopping rules remain intact.
Two-stage FP64 reductions change floating-point grouping, not the vocabulary
or probability definition. Every actual commit is checked on finite examples.
"""
import argparse
import inspect
from pathlib import Path
import statistics as stats
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .flash_readout import ModelReadout,generator,suppress_official_prints
from .flash_statistics import statistics,reference
from .padded_head import project


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    binding=check_binding(required=True)
    from ..common import sha256,write_json
    from ..llada_common import MODEL_ID,REVISION,prompt_ids
    from .competitors import load_external,load_model,generation_prompt,select_samples
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--third-party',type=Path,required=True)
    parser.add_argument('--datasets',type=Path,nargs=3,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    cls,external,adaptation=load_external(args.third_party,'flash_verify')
    official=Path(inspect.getsourcefile(inspect.unwrap(external)))
    model,tokenizer=load_model(SimpleNamespace(method='flash_verify'),cls)
    torch.manual_seed(1234);torch.set_num_threads(1)
    data=dict(zip(('humaneval','mbpp','math'),args.datasets))
    report=dict(model=MODEL_ID,revision=REVISION,gpu_binding=binding,adaptation=adaptation,
        datasets={k:sha256(p) for k,p in data.items()},official_generate_sha256=sha256(official),
        implementation={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        config=dict(length=256,block=32,threshold=.9,gamma=.8,track_num=4,mask_num=4,
                    seed=51713,offset=0,prompts_per_task=2,minimum_head_rows=64,clean_repeats=2),
        scope='Six reused dev prompts per Flash decoder. No new weights/query/KV pruning. '
              'Full-vocabulary FP64 exp/sums, reduced intermediate storage; reduction grouping can round differently. '
              'Finite step/cache/text/NFE parity only; no universal losslessness or independent accuracy claim. '
              'Only clean full-request latency is used for speed comparison; traced/shadow/microcost excluded.',
        backend=dict(engine='Pinned official fused Triton + FP64 vocabulary reductions',torch_sdpa_calls=0),
        synthetic=[],heads=[],costs=[],prompts=[])
    write_json(args.output/'diagnostic.json',report)
    def measure(fn):
        torch.cuda.synchronize();started=time.perf_counter();value=fn();torch.cuda.synchronize()
        return value,time.perf_counter()-started
    # Include actual vocabulary, chunk edges, equal maximum ties, extreme values,
    # empty rows, target probabilities and strided contiguous-vocabulary slices.
    for rows,vocab in ((0,17),(3,17),(2,2049),(4,4096),(3,model.config.embedding_size)):
        z=torch.randn(rows+1,vocab,device=model.device,dtype=torch.bfloat16)[1:]
        if rows:
            z[0].fill_(0);z[0,0]=8;z[0,-1]=8
        labels=torch.arange(rows,device=model.device,dtype=torch.long)%vocab
        for selected in (None,labels):
            expected=reference(z,selected);actual=statistics(z,selected)
            error=float((expected[0]-actual[0]).abs().max()) if rows else 0.
            top_match=torch.equal(expected[1],actual[1])
            report['synthetic'].append(dict(rows=rows,vocabulary=vocab,target=selected is not None,
                                           maximum_probability_error=error,top1_equal=top_match))
            assert error<5e-13 and top_match,'Vocabulary reduction control failed'
    original_sdpa=F.scaled_dot_product_attention
    def watched(*a,**kw):
        report['backend']['torch_sdpa_calls']+=1;return original_sdpa(*a,**kw)
    F.scaled_dot_product_attention=watched
    try:
        for verify in (True,False):
            mode='flash_verify' if verify else 'flash_cache'
            for task,path in data.items():
                for sample in select_samples(path,2,0):
                    ident=sample.get('id',sample.get('task_id'))
                    ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                    prompt=torch.tensor(ids,device=model.device)
                    flags=dict(calls=0,hidden=None,cache_len=None,profiled=set())
                    def count(_module,_args,kw):
                        flags['calls']+=1
                        if not kw['lengths'][-1]:flags['cache_len']=int(kw['lengths'][2][0,1])
                    counter=model.register_forward_pre_hook(count,with_kwargs=True)
                    def generate(fn,proxy):
                        responses,steps=[None],[0];flags['calls']=0
                        with suppress_official_prints():
                            fn(proxy,[prompt],[len(ids)],1,responses,steps,gen_length=256,block_length=32,
                                threshold=.9,gamma=.8,track_num=4,mask_num=4,verify=verify,tokenizer=tokenizer,stop_tokens=[])
                        torch.cuda.synchronize()
                        return dict(text=responses[0],calls=flags['calls'],iterations=steps[0])
                    compact=ModelReadout(model,compact=True,minimum=64)
                    compiled=generator(external,statistics=statistics)
                    try:
                        generate(external,model);generate(compiled,compact)
                        def capture(_m,_a,h):flags['hidden']=h
                        nh=model.model.transformer.ln_f.register_forward_hook(capture)
                        def observe(spec,inputs,kwargs,output):
                            start,n=spec['start'],spec['count'];h=flags['hidden']
                            z=output.logits[0,start:start+n]
                            labels=inputs[0][0,start-n:start] if spec['verify'] and n else None
                            expected=reference(z,labels);actual=statistics(z,labels)
                            error=float((expected[0]-actual[0]).abs().max()) if n else 0.
                            head=project(model,h[:,start:start+n],minimum=64 if n else 0)[0]
                            head_error=float((z.float()-head.float()).abs().max()) if n else 0.
                            decision=(torch.equal(expected[1],actual[1]) and
                                      (torch.equal(expected[0].cumprod(0)>=.8,actual[0].cumprod(0)>=.8)
                                       if spec['verify'] else torch.equal(expected[0].sort(descending=True).indices,
                                                                        actual[0].sort(descending=True).indices) and
                                       torch.equal(expected[0]>=.9,actual[0]>=.9)))
                            report['heads'].append(dict(method=mode,task=task,id=ident,call=flags['calls'],
                                verify=spec['verify'],full_rows=h.shape[1],consumed_rows=n,
                                maximum_probability_error=error,head_maximum_logit_error=head_error,
                                statistics_decision_match=decision))
                            key='verify' if spec['verify'] else 'normal'
                            if n and key not in flags['profiled'] and (spec['verify'] or h.shape[1]<=256):
                                def native_probability():
                                    p=z.double().softmax(-1)
                                    return p.max(-1) if labels is None else p.gather(1,labels[:,None]).flatten()
                                methods=dict(head_full=lambda:project(model,h,minimum=0),
                                             head64=lambda:project(model,h[:,start:start+n],minimum=64),
                                             probability_native=native_probability,
                                             probability_fused=lambda:statistics(z,labels))
                                cost={name:[] for name in methods}
                                for fn in methods.values():fn()
                                for _ in range(5):
                                    for name,fn in methods.items():
                                        value,seconds=measure(fn);cost[name].append(seconds);del value
                                report['costs'].append(dict(method=mode,task=task,id=ident,kind=key,
                                    full_rows=h.shape[1],consumed_rows=n,seconds=cost,
                                    scope='Fixed same hidden/logits; excluded from clean request latency.'))
                                flags['profiled'].add(key)
                            flags['hidden']=None
                        actions=[]
                        traced=generator(external,lambda p,t:actions.append((p.tolist(),t.tolist())))
                        try:baseline=generate(traced,ModelReadout(model,observer=observe))
                        finally:nh.remove();flags['hidden']=None
                        valid=flags['cache_len']
                        saved=[(b.k_cache[:valid].clone(),b.v_cache[:valid].clone()) for b in model.model.transformer.blocks]
                        adapted_actions=[]
                        adapted=generator(external,lambda p,t:adapted_actions.append((p.tolist(),t.tolist())),statistics=statistics)
                        candidate=generate(adapted,compact)
                        cache_match=valid==flags['cache_len'] and all(
                            torch.equal(b.k_cache[:valid].view(torch.int16),k.view(torch.int16)) and
                            torch.equal(b.v_cache[:valid].view(torch.int16),v.view(torch.int16))
                            for b,(k,v) in zip(model.model.transformer.blocks,saved))
                        del saved
                        timings=dict(official=[],optimized=[]);clean=dict(official=[],optimized=[])
                        for repeat in range(2):
                            for name in (('official','optimized') if repeat==0 else ('optimized','official')):
                                fn,proxy=(external,model) if name=='official' else (compiled,compact)
                                value,seconds=measure(lambda:generate(fn,proxy))
                                timings[name].append(seconds);clean[name].append(value)
                        clean_match=all(value==baseline for values in clean.values() for value in values)
                        action_match=actions==adapted_actions
                        report['prompts'].append(dict(method=mode,task=task,id=ident,reference=baseline,candidate=candidate,
                            exact_step_actions_match=action_match,cache_match=cache_match,
                            traced_text_calls_iterations_match=baseline==candidate,clean_parity=clean_match,
                            finite_parity_pass=action_match and cache_match and baseline==candidate and clean_match,
                            action_count=len(actions),clean_seconds=timings,
                            speedup=stats.median(timings['official'])/stats.median(timings['optimized'])))
                        write_json(args.output/'diagnostic.json',report)
                        print(f'{mode} {task} {ident}: parity={report["prompts"][-1]["finite_parity_pass"]}, '
                              f'speed={report["prompts"][-1]["speedup"]:.4f}',flush=True)
                    finally:counter.remove()
        assert report['backend']['torch_sdpa_calls']==0 and report['official_generate_sha256']==sha256(official)
        summary={}
        for mode in ('flash_verify','flash_cache'):
            rows=[p for p in report['prompts'] if p['method']==mode];heads=[h for h in report['heads'] if h['method']==mode]
            summary[mode]=dict(prompts=len(rows),finite_parity_pass=sum(p['finite_parity_pass'] for p in rows),
                head_states=len(heads),head_bitwise_match=sum(h['head_maximum_logit_error']==0 for h in heads),
                statistics_decisions_match=sum(h['statistics_decision_match'] for h in heads),
                maximum_probability_error=max(h['maximum_probability_error'] for h in heads),
                pooled_speedup=sum(sum(p['clean_seconds']['official']) for p in rows)/
                               sum(sum(p['clean_seconds']['optimized']) for p in rows),
                median_paired_speedup=stats.median(p['speedup'] for p in rows))
        report['summary']=summary
        report['continuation_gate_pass']=all(s['finite_parity_pass']==6 and s['pooled_speedup']>1.02 for s in summary.values())
        write_json(args.output/'diagnostic.json',report);(args.output/'complete').write_text('OK\n')
    finally:F.scaled_dot_product_attention=original_sdpa


if __name__=='__main__':main()
