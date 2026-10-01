"""Pinned Flash head regression, finite trajectory parity and clean latency.

No query pruning, cache/attention-mask change, new acceptance rule or training.
This is execution engineering over the OFFICIAL Flash-Verify/Flash-Cache.
"""
import argparse
import hashlib
import inspect
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .flash_readout import ModelReadout,generator,suppress_official_prints
from .padded_head import project


def head_decision(z, verify, labels=None):
    if z.shape[1]==0:return [],[],[]
    p=z[0].double().softmax(-1);conf,top=p.max(-1)
    if verify:
        value=p.gather(-1,labels[:,None]).flatten()
        accepted=int((value.cumprod(0)>=.8).sum())
        return list(range(accepted)),labels[:accepted].tolist(),value.tolist()
    _,order=conf.sort(descending=True)
    count=max(int((conf>=.9).sum()),1)
    take=order[:count]
    return take.tolist(),top[take].tolist(),conf.tolist()


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
    torch.set_num_threads(1)
    data=dict(zip(('humaneval','mbpp','math'),args.datasets))
    report=dict(model=MODEL_ID,revision=REVISION,gpu_binding=binding,adaptation=adaptation,
        datasets={k:sha256(p) for k,p in data.items()},official_generate_sha256=sha256(official),
        implementation={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        config=dict(length=256,block=32,threshold=.9,gamma=.8,track_num=4,mask_num=4,
            seed=51713,offset=0,prompts_per_task=2,head_min_rows=32,clean_repeats=2),
        scope='Six REUSED dev prompts per decoder. SAME official decoder/cache/visibility/threshold/stop/formatting. '
            'Only normalized consumed head rows projected, with 32-row padding and original full-vocab softmax. '
            'Finite exact action/cache parity is not all-input native losslessness. Engineering gain over '
            'official Flash, not a new algorithm contribution. Clean timings include cache init/prefill/sampling/stop; '
            'shadow head/profile and action logging excluded from latency.',
        backend=dict(engine='Pinned official Flash fused Triton; original attention/projection/cache',torch_sdpa_calls=0),
        prompts=[],heads=[],head_costs=[])
    write_json(args.output/'diagnostic.json',report)
    sdpa=F.scaled_dot_product_attention
    def watched(*a,**k):
        report['backend']['torch_sdpa_calls']+=1;return sdpa(*a,**k)
    F.scaled_dot_product_attention=watched
    def measured(fn):
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
        return value,time.perf_counter()-start
    try:
        for verify in (True,False):
            mode='flash_verify' if verify else 'flash_cache'
            for task,path in data.items():
                for sample in select_samples(path,2,0):
                    ident=sample.get('id',sample.get('task_id'))
                    ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                    prompt=torch.tensor(ids,device=model.device)
                    flags=dict(calls=0,normalized=None,cache_len=None,benchmarked=set())
                    def count(_m,_a,kw):
                        flags['calls']+=1
                        if not kw['lengths'][-1]:flags['cache_len']=int(kw['lengths'][2][0,1])
                    counter=model.register_forward_pre_hook(count,with_kwargs=True)
                    def generate(fn,proxy):
                        responses,steps=[None],[0];flags['calls']=0
                        with suppress_official_prints():
                            fn(proxy,[prompt],[len(ids)],1,responses,steps,gen_length=256,block_length=32,
                                threshold=.9,gamma=.8,track_num=4,mask_num=4,verify=verify,tokenizer=tokenizer,stop_tokens=[])
                        torch.cuda.synchronize();return dict(text=responses[0],iterations=steps[0],calls=flags['calls'])
                    compact_fn=generator(external)
                    compact_proxy=ModelReadout(model,compact=True)
                    try:
                        # Warm both ACTUAL request workloads; excluded from scored latency.
                        generate(external,model);generate(compact_fn,compact_proxy)
                        def capture(_m,_a,value):flags['normalized']=value
                        nh=model.model.transformer.ln_f.register_forward_hook(capture)
                        actions=[]
                        def observe(spec,inputs,kwargs,output):
                            h=flags['normalized'];start,n=spec['start'],spec['count']
                            ref=output.logits[:,start:start+n]
                            alt=project(model,h[:,start:start+n],minimum=32 if n else 0)
                            difference=float((ref.float()-alt.float()).abs().max()) if n else 0.
                            labels=inputs[0][0,start-n:start] if spec['verify'] and n else None
                            a=head_decision(ref,spec['verify'],labels);b=head_decision(alt,spec['verify'],labels)
                            report['heads'].append(dict(method=mode,task=task,id=ident,call=flags['calls'],
                                full_rows=h.shape[1],readout_start=start,consumed_rows=n,
                                projected_rows=max(n,32) if n else 0,max_logit_error=difference,
                                top1_equal=torch.equal(ref.argmax(-1),alt.argmax(-1)) if n else True,
                                raw_decision_match=a[:2]==b[:2],max_probability_error=max([abs(x-y) for x,y in zip(a[2],b[2])],default=0.)))
                            key='verify' if spec['verify'] else 'normal'
                            # First ordinary (not prefill) and first nonempty verification.
                            if key not in flags['benchmarked'] and n and (spec['verify'] or h.shape[1]<=256):
                                full=lambda:project(model,h,minimum=0)
                                selected=lambda:project(model,h[:,start:start+n],minimum=32)
                                full();selected()
                                timings=dict(full=[],consumed=[])
                                for _ in range(5):
                                    value,seconds=measured(full);timings['full'].append(seconds);del value
                                    value,seconds=measured(selected);timings['consumed'].append(seconds);del value
                                report['head_costs'].append(dict(method=mode,task=task,id=ident,kind=key,
                                    full_rows=h.shape[1],consumed_rows=n,**timings,
                                    scope='Same normalized hidden; head-only microcost, excluded from generation latency.'))
                                flags['benchmarked'].add(key)
                            flags['normalized']=None
                        traced=generator(external,lambda p,t:actions.append((p.tolist(),t.tolist())))
                        try:reference=generate(traced,ModelReadout(model,observer=observe))
                        finally:nh.remove();flags['normalized']=None
                        valid=flags['cache_len']
                        saved=[(b.k_cache[:valid].clone(),b.v_cache[:valid].clone()) for b in model.model.transformer.blocks]
                        changed_actions=[]
                        adapted=generator(external,lambda p,t:changed_actions.append((p.tolist(),t.tolist())))
                        candidate=generate(adapted,compact_proxy)
                        cache_same=all(torch.equal(b.k_cache[:valid].view(torch.int16),k.view(torch.int16)) and
                            torch.equal(b.v_cache[:valid].view(torch.int16),v.view(torch.int16))
                            for b,(k,v) in zip(model.model.transformer.blocks,saved))
                        del saved
                        same=reference==candidate and actions==changed_actions and cache_same
                        timing=dict(official=[],consumed=[]);clean_outputs=dict(official=[],consumed=[])
                        for repeat in range(2):
                            order=('official','consumed') if repeat==0 else ('consumed','official')
                            for name in order:
                                fn,proxy=(external,model) if name=='official' else (compact_fn,compact_proxy)
                                value,seconds=measured(lambda:generate(fn,proxy));timing[name].append(seconds)
                                clean_outputs[name].append(value)
                        clean_same=all(value==reference for values in clean_outputs.values() for value in values)
                        report['prompts'].append(dict(method=mode,task=task,id=ident,reference=reference,candidate=candidate,
                            exact_step_actions_match=actions==changed_actions,cache_match=cache_same,
                            traced_text_calls_iterations_match=reference==candidate,clean_parity=clean_same,
                            finite_parity_pass=same and clean_same,action_count=len(actions),
                            text_sha256=hashlib.sha256(reference['text'].encode()).hexdigest(),
                            clean_seconds=timing,speedup=statistics.median(timing['official'])/statistics.median(timing['consumed'])))
                        write_json(args.output/'diagnostic.json',report)
                        print(f'{mode} {task} {ident}: parity={same and clean_same}, speed={report["prompts"][-1]["speedup"]:.4f}',flush=True)
                    finally:counter.remove()
        assert report['backend']['torch_sdpa_calls']==0 and report['official_generate_sha256']==sha256(official)
        summary={}
        for mode in ('flash_verify','flash_cache'):
            rows=[p for p in report['prompts'] if p['method']==mode];heads=[h for h in report['heads'] if h['method']==mode]
            summary[mode]=dict(prompts=len(rows),finite_parity_pass=sum(p['finite_parity_pass'] for p in rows),
                consumed_head_states=len(heads),exact_head_states=sum(h['max_logit_error']==0 for h in heads),
                raw_head_decisions_match=sum(h['raw_decision_match'] for h in heads),
                median_paired_speedup=statistics.median(p['speedup'] for p in rows),
                pooled_speedup=sum(sum(p['clean_seconds']['official']) for p in rows)/sum(sum(p['clean_seconds']['consumed']) for p in rows))
        report['summary']=summary
        report['continuation_gate_pass']=all(s['finite_parity_pass']==6 and s['pooled_speedup']>1.02 for s in summary.values())
        write_json(args.output/'diagnostic.json',report);(args.output/'complete').write_text('OK\n')
    finally:F.scaled_dot_product_attention=sdpa


if __name__=='__main__':main()
