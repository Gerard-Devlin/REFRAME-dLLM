"""Private same-state cost/information-flow gate for FOCUS-v3 joint views.

No joint result is committed. Proposals come ONLY from a preceding, paid normal
call and are verified against the current legal canvas. Timings compare call
geometries, not equivalent algorithms or end-to-end speed/accuracy.
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

from .joint_views import layout,pending_candidates
from .joint_engine import owned_cache,joint_adapter,prepare_call


def private_kwargs(kwargs):
    result=dict(kwargs);result['positions']=list(kwargs['positions']);result['lengths']=list(kwargs['lengths'])
    result['positions'][4]=kwargs['positions'][4].clone()
    return result


def cache_bits_equal(a,b):return torch.equal(a.view(torch.int16),b.view(torch.int16))


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
    parser.add_argument('--tiled',action='store_true',help='Fixed 32-row Q tiles, same 128-row visibility graph')
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    cls,external,adaptation=load_external(args.third_party,'flash_verify')
    official_file=Path(inspect.getsourcefile(inspect.unwrap(external)))
    model,tokenizer=load_model(SimpleNamespace(method='flash_verify'),cls)
    torch.set_num_threads(1)
    forbidden=set(tokenizer.all_special_ids)|{126336,126081}
    replacements=[next(i for i in tokenizer.encode(s,add_special_tokens=False) if i not in forbidden) for s in ('0','1')]
    assert replacements[0]!=replacements[1]
    data=dict(zip(('humaneval','mbpp','math'),args.datasets))
    report=dict(model=MODEL_ID,revision=REVISION,gpu_binding=binding,adaptation=adaptation,
        datasets={k:sha256(p) for k,p in data.items()},official_generate_sha256=sha256(official_file),
        implementation={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        configuration=dict(length=256,block=32,threshold=.9,gamma=.8,seed=51713,offset=0,
            prompts_per_task=2,private_controls_per_prompt=2,joint_tile=128,common_active_max=32,
            common_tracked_max=64,private_proposals_max=16,stable_repeats=5,tiled=args.tiled),
        backend=dict(engine='Pinned official fused Triton QKV/attention; common-only cache scatter',torch_sdpa_calls=0),
        scope='Six reused development prompts. Private joint views are NEVER committed. '
            'Prior normal predictions are proposals, not replayed actions. Background KV belongs to '
            'the paid legal trajectory. Shared current MASK placeholders differ from fully recomputed '
            'conditional native states. Isolation and root equality do not imply task correctness. '
            'Full output heads for BOTH cost geometries; no active-head engineering advantage. '
            'Cache cloning/reset is diagnostic isolation, excluded from model-call timing and reported '
            'as setup. These are geometry costs, NOT end-to-end speed or a quality result.',
        falsification=dict(max_joint_to_two_call_ratio=.75,public_label_logit_change=0,
            public_label_cache_change=0,root_logit_difference=0),prompts=[],controls=[])
    write_json(args.output/'diagnostic.json',report)
    original_sdpa=F.scaled_dot_product_attention
    def watched(*a,**k):
        report['backend']['torch_sdpa_calls']+=1
        return original_sdpa(*a,**k)
    F.scaled_dot_product_attention=watched
    def measured(fn):
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
        return value,time.perf_counter()-start
    try:
        with joint_adapter(model):
            for task,path in data.items():
                for sample in select_samples(path,2,0):
                    ident=sample.get('id',sample.get('task_id'))
                    ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                    prompt=torch.tensor(ids,device=model.device)
                    flags=dict(busy=False,calls=0,shadow=0,previous=[],controls=0,pending=None)
                    def count(_m,_a,_k):flags['shadow' if flags['busy'] else 'calls']+=1
                    handle=model.register_forward_pre_hook(count,with_kwargs=True)
                    def generate():
                        responses,steps=[None],[0]
                        external(model,[prompt],[len(ids)],1,responses,steps,gen_length=256,
                            block_length=32,threshold=.9,gamma=.8,track_num=4,mask_num=4,
                            verify=True,tokenizer=tokenizer,stop_tokens=[])
                        torch.cuda.synchronize();return responses[0],steps[0]
                    try:
                        text,iterations=generate();clean_calls=flags['calls'];flags['calls']=0
                        def pre(_m,inputs,kwargs):
                            if flags['busy'] or kwargs['lengths'][-1] or flags['controls']>=2:return
                            query=inputs[0][0];pos=kwargs['positions'][0]
                            rowids=[i for i in range(min(32,len(query))) if int(query[i])==126336 and int(pos[i])<len(ids)+256]
                            active=[int(pos[i]) for i in rowids]
                            candidates=pending_candidates(flags['previous'],active,forbidden)
                            if len(candidates)<2:return
                            tracked=[i for i in range(32,len(query)) if int(query[i])!=126336 and int(pos[i])<len(ids)+256][-64:]
                            cache_len=int(kwargs['lengths'][2][0,1])
                            plan=layout(active,[126336]*len(active),[int(pos[i]) for i in tracked],
                                [int(query[i]) for i in tracked],[v['position'] for v in candidates],
                                [v['token'] for v in candidates],forbidden=forbidden,cache_length=cache_len)
                            torch.cuda.synchronize();setup=time.perf_counter()
                            q,k=prepare_call(plan,kwargs,cache_len,model.device,tiled=args.tiled)
                            replacement=next(v for v in replacements if v!=int(q[0,plan.data_begin]))
                            changed=q.clone();changed[0,plan.data_begin]=replacement
                            base=[(b.k_cache.clone(),b.v_cache.clone()) for b in model.model.transformer.blocks]
                            torch.cuda.synchronize();setup=time.perf_counter()-setup
                            flags['busy']=True
                            try:
                                with owned_cache(model):
                                    original,cold=measured(lambda:model(q,**k).logits)
                                    common_caches=[(b.k_cache[:cache_len].index_select(0,k['lengths'][10].common_positions).clone(),
                                        b.v_cache[:cache_len].index_select(0,k['lengths'][10].common_positions).clone())
                                        for b in model.model.transformer.blocks]
                                    outside=k['positions'][1].long()
                                    assert all(cache_bits_equal(b.k_cache.index_select(0,outside),oldk.index_select(0,outside)) and
                                        cache_bits_equal(b.v_cache.index_select(0,outside),oldv.index_select(0,outside))
                                        for b,(oldk,oldv) in zip(model.model.transformer.blocks,base)),'Private KV wrote outside common'
                                    altered,_=measured(lambda:model(changed,**k).logits)
                                    common_error=float((original[0,:plan.common].float()-altered[0,:plan.common].float()).abs().max())
                                    root_error=float((original[0,plan.proposal_rows[0]].float()-original[0,plan.verify_begin].float()).abs().max())
                                    own_error=float((original[0,plan.verify_begin].float()-altered[0,plan.verify_begin].float()).abs().max())
                                    common_cache_same=all(cache_bits_equal(b.k_cache.index_select(0,k['lengths'][10].common_positions),oldk) and
                                        cache_bits_equal(b.v_cache.index_select(0,k['lengths'][10].common_positions),oldv)
                                        for b,(oldk,oldv) in zip(model.model.transformer.blocks,common_caches))
                                    del altered,common_caches
                                    stable=[]
                                    for _ in range(5):
                                        value,seconds=measured(lambda:model(q,**k).logits);stable.append(seconds);del value
                                    prob=original[0,plan.verify_begin:plan.verify_begin+plan.search].double().softmax(-1)
                                    tokens=torch.tensor([v['token'] for v in candidates],device=model.device)
                                    target=prob.gather(-1,tokens[:,None]).flatten()
                                    accepted=int((target.cumprod(0)>=.8).sum())
                                    control=dict(task=task,id=ident,model_call=flags['calls'],active=len(active),tracked=len(tracked),
                                        proposals=len(candidates),proposal_source='preceding paid normal-call predictions',
                                        old_first_label=int(q[0,plan.data_begin]),perturbed_first_label=replacement,
                                        probabilities=target.cpu().tolist(),accepted_private_prefix=accepted,
                                        proposal_argmax_matches=int((prob.argmax(-1)==tokens).sum()),
                                        public_max_logit_change=common_error,root_max_logit_difference=root_error,
                                        root_own_label_max_logit_change=own_error,public_cache_label_independent=common_cache_same,
                                        outside_common_cache_unchanged=True,setup_seconds=setup,cold_joint_seconds=cold,
                                        stable_joint_seconds=stable,normal_query_rows=len(query),joint_query_rows=128,
                                        public_positions=list(plan.positions[:plan.common]),proposed_positions=[v['position'] for v in candidates])
                                    assert common_error==0 and own_error==0 and root_error==0 and common_cache_same
                                    del original,prob
                                assert all(cache_bits_equal(b.k_cache,oldk) and cache_bits_equal(b.v_cache,oldv)
                                    for b,(oldk,oldv) in zip(model.model.transformer.blocks,base)),'Actual cache changed'
                                flags['controls']+=1
                                flags['pending']=dict(control=control,normal_q=inputs[0].clone(),
                                    normal_kwargs=private_kwargs(kwargs),base=base,q=q,k=k)
                            finally:flags['busy']=False
                        def post(_m,inputs,kwargs,output):
                            if flags['busy']:return
                            if not kwargs['lengths'][-1]:
                                query=inputs[0][0];pos=kwargs['positions'][0]
                                count=min(32,len(query));prob=output.logits[0,:count].double().softmax(-1)
                                confidence,token=prob.max(-1)
                                flags['previous']=[dict(position=int(pos[i]),token=int(token[i]),confidence=float(confidence[i]))
                                    for i in range(count) if int(query[i])==126336]
                                return
                            packet=flags['pending']
                            if packet is None:return
                            flags['pending']=None;flags['busy']=True
                            try:
                                # True official draft-then-verify call geometry, including its two full
                                # heads. Conditional inputs differ from the private joint hypothesis.
                                verify_q=inputs[0].clone();vk=private_kwargs(kwargs)
                                times=[];reset=[]
                                with owned_cache(model):
                                    for repeat in range(6):
                                        torch.cuda.synchronize();start=time.perf_counter()
                                        for b,(oldk,oldv) in zip(model.model.transformer.blocks,packet['base']):
                                            b.k_cache.copy_(oldk);b.v_cache.copy_(oldv)
                                        torch.cuda.synchronize();reset.append(time.perf_counter()-start)
                                        nk=private_kwargs(packet['normal_kwargs']);vv=private_kwargs(vk)
                                        def pair():
                                            a=model(packet['normal_q'],**nk).logits
                                            b=model(verify_q,**vv).logits
                                            return a,b
                                        result,seconds=measured(pair);del result
                                        if repeat:times.append(seconds)
                                c=packet['control'];c.update(verify_query_rows=verify_q.shape[1],
                                    stable_official_two_call_seconds=times,diagnostic_cache_reset_seconds=reset,
                                    joint_to_two_call_ratio=statistics.median(c['stable_joint_seconds'])/statistics.median(times))
                                report['controls'].append(c)
                                print(f"joint control {len(report['controls'])}: ratio={c['joint_to_two_call_ratio']:.4f}, private_accept={c['accepted_private_prefix']}/{c['proposals']}",flush=True)
                                write_json(args.output/'diagnostic.json',report)
                            finally:flags['busy']=False
                        pre_handle=model.register_forward_pre_hook(pre,with_kwargs=True)
                        post_handle=model.register_forward_hook(post,with_kwargs=True)
                        try:seen,seen_iterations=generate()
                        finally:pre_handle.remove();post_handle.remove()
                        assert text==seen and iterations==seen_iterations and clean_calls==flags['calls']
                        assert flags['pending'] is None
                        report['prompts'].append(dict(task=task,id=ident,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                            official_iterations=iterations,clean_model_calls=clean_calls,observed_model_calls=flags['calls'],
                            private_shadow_calls=flags['shadow'],private_controls=flags['controls'],text_and_nfe_match=True))
                        write_json(args.output/'diagnostic.json',report)
                        print(f'Completed prompt {len(report["prompts"])}/6',flush=True)
                    finally:handle.remove()
        assert report['backend']['torch_sdpa_calls']==0 and report['official_generate_sha256']==sha256(official_file)
        ratios=[c['joint_to_two_call_ratio'] for c in report['controls']]
        report['summary']=dict(controls=len(ratios),median_joint_to_two_call_ratio=statistics.median(ratios) if ratios else None,
            private_prefix_accepted=sum(c['accepted_private_prefix'] for c in report['controls']),
            proposed=sum(c['proposals'] for c in report['controls']),
            flow_and_cache_checks_pass=True,cost_gate_pass=bool(ratios) and statistics.median(ratios)<=.75,
            scope='Geometry gate only. No online FOCUS-v3, task accuracy, independent validation or native losslessness claim.')
        write_json(args.output/'diagnostic.json',report);(args.output/'complete').write_text('OK\n')
    finally:F.scaled_dot_product_attention=original_sdpa


if __name__=='__main__':main()
