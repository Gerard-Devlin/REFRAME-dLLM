"""Measure real FP8 cost first; audit paired actions only after the cost gate.

The official BF16 generator ALWAYS drives the trajectory. Low precision never
commits tokens here. Normal calls share read-only BF16 KV, checked after each pair.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

from .common import MODEL_ID, REVISION, digest, load_prompts, sha256, snapshot, write_json
from .decision import active, choose, action_list, stability_score, snapshot_cache, assert_cache_unchanged
from .quant import convert, kernel_probe


def measured(fn):
    torch.cuda.synchronize()
    begin=time.perf_counter()
    result=fn()
    torch.cuda.synchronize()
    return result,time.perf_counter()-begin


def bounded_generate(model,ids,options):
    """Fail on no-progress quantized generation, never silently truncate it.

    With at least one MASK filled per ordinary call, each block needs at most
    32 denoise calls plus one clean cache write. Keep prefill allowance explicit.
    The same counter wrapper is used for BF16 and static-low latency baselines.
    """
    original=model.forward
    limit=2+(options['max_new_tokens']//options['block_size'])*(options['block_size']+1)
    calls=0
    def guarded(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls>limit:
            raise RuntimeError('Native generation exceeded progress bound (possible MASK prediction); stop backend')
        return original(*args,**kwargs)
    model.forward=guarded
    try:
        return model.generate(ids,**options)
    finally:
        model.forward=original


class PairedObserver:
    def __init__(self,reference,low,*,stage,threshold,repeats=5,max_states=4):
        self.reference,self.low=reference,low
        self.stage,self.threshold=stage,threshold
        self.repeats,self.max_states=repeats,max_states
        self.rows=[]; self.calls=0; self.normal_calls=0
        self.pending=None; self.expected_next=None; self.verified=0

    def __enter__(self):
        self.original_forward=self.reference.forward
        self.original_sample=self.reference.sample_with_top_p
        self.reference.forward,self.reference.sample_with_top_p=self.forward,self.sample
        return self

    def __exit__(self,*_):
        self.reference.forward,self.reference.sample_with_top_p=self.original_forward,self.original_sample

    def forward(self,*args,**kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None)
        self.calls+=1
        if self.expected_next is not None:
            if ids[0].tolist()!=self.expected_next:
                raise AssertionError('Recorded native action differs from next actual input')
            self.verified+=1; self.expected_next=None
        normal=(ids.shape==(1,32) and kwargs.get('update_past_key_values') is False
                and not kwargs.get('use_block_cache',False))
        self.pending=None
        if not normal:
            return self.original_forward(*args,**kwargs)
        self.normal_calls+=1
        if self.stage=='cost' and len(self.rows)>=self.max_states:
            return self.original_forward(*args,**kwargs)
        cache=kwargs.get('past_key_values')
        # Checks are diagnostic overhead, excluded from timed forward cost.
        saved=snapshot_cache(cache)
        ref_call=lambda:self.original_forward(*args,**kwargs)
        low_call=lambda:self.low(*args,**kwargs)
        reference,ref_time=measured(ref_call)
        low,low_time=measured(low_call)
        assert_cache_unchanged(cache,saved)
        timings=[]
        if self.stage=='cost':
            # Warm BOTH paths at the exact same state. Alternate AB/BA order.
            ref_call(); low_call(); torch.cuda.synchronize()
            for repeat in range(self.repeats):
                if repeat%2:
                    low,lo=measured(low_call); check,bf=measured(ref_call)
                else:
                    check,bf=measured(ref_call); low,lo=measured(low_call)
                if not torch.equal(reference.logits,check.logits):
                    raise AssertionError('Repeated BF16 forward changed at an identical state')
                timings.append(dict(bf16_seconds=bf,low_seconds=lo))
            assert_cache_unchanged(cache,saved)
            ref_time=statistics.median(t['bf16_seconds'] for t in timings)
            low_time=statistics.median(t['low_seconds'] for t in timings)
        low_logits=torch.cat((low.logits[:,:1],low.logits[:,:-1]),dim=1)
        state=ids[0].detach().clone()
        start,mask=active(state)
        self.pending=dict(state=state,start=start,mask=mask,low_logits=low_logits[0,start:start+8],
                          row=dict(call_index=self.calls,bf16_seconds=ref_time,low_seconds=low_time,
                                   repeats=timings,cache_tokens=0 if cache is None else cache.get_seq_length(),
                                   state_hash=digest(state.tolist())))
        return reference

    def sample(self,logits,top_p=.95,temperature=0):
        tokens,probs=self.original_sample(logits,top_p=top_p,temperature=temperature)
        if self.pending is None:
            return tokens,probs
        if temperature!=0 or tokens.shape!=(1,8):
            raise AssertionError('Unsupported native sampling path')
        p=self.pending; mask=p['mask']; row=p['row']
        selected,_=choose(tokens[0],probs[0],mask,self.threshold)
        bf_action=action_list(tokens[0],selected,p['start'])
        low_logits=p['low_logits']
        if not bool(torch.isfinite(low_logits).all()):
            raise RuntimeError('Non-finite FP8 logits; reject backend before calibration')
        low_tokens,low_probs=self.low.sample_with_top_p(low_logits[None],top_p=top_p,temperature=0)
        low_selected,_=choose(low_tokens[0],low_probs[0],mask,self.threshold)
        low_action=action_list(low_tokens[0],low_selected,p['start'])
        score,score_seconds=measured(lambda:stability_score(low_logits,mask,low_selected,self.threshold))
        row.update(score=float(score),score_seconds=score_seconds,
                   bf16_action=bf_action,low_action=low_action,equal=bf_action==low_action,
                   bf16_selected_eos=any(t==151645 for _,t in bf_action),
                   low_selected_eos=any(t==151645 for _,t in low_action))
        self.rows.append(row)
        expected=p['state'].tolist()
        for position,token in bf_action:
            expected[position]=token
        self.expected_next=expected; self.pending=None
        return tokens,probs


def cost_summary(rows,target=1.5):
    states=[s for row in rows for s in row['states']]
    if not states:
        raise ValueError('No ordinary denoising state was measured')
    bf=sum(s['bf16_seconds'] for s in states)
    low=sum(s['low_seconds'] for s in states)
    ratios=[s['low_seconds']/s['bf16_seconds'] for s in states]
    return dict(sampled_states=len(states),bf16_forward_seconds=bf,low_forward_seconds=low,
                rho=low/bf,min_state_rho=min(ratios),max_state_rho=max(ratios),
                zero_fallback_all_work_low_ceiling=bf/low,
                cost_gate_pass=bf/low>=target,cost_target=target,
                note='Same-state paired synchronized forward timings including activation quantization. '
                     'Sampled states, not an end-to-end upper-bound theorem. No gate/fallback/fixed overhead charged. '
                     'If this backend is too slow, stop it; do not infer all quantization backends are slow.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('cost','audit'),required=True)
    p.add_argument('--role',choices=('development','calibration','evaluation'),default='development')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prompts',type=int,default=4)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--threshold',type=float,default=.90)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--max-states',type=int,default=4)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--cost-report',type=Path)
    args=p.parse_args()
    if min(args.prompts,args.repeats,args.max_states)<1 or args.max_new_tokens<64 or args.max_new_tokens%32 or not 0<args.threshold<1:
        p.error('Positive sizes, generation multiple of 32 >=64, threshold strictly in (0,1)')
    if args.stage=='cost' and args.role!='development':
        p.error('Cost backend selection must use development prompts')
    cost=None
    if args.stage=='audit':
        if args.cost_report is None:
            p.error('Audit requires --cost-report from a passed real-backend cost test')
        cost=json.loads(args.cost_report.read_text())
        if cost.get('status')!='complete' or not cost.get('cost',{}).get('cost_gate_pass'):
            p.error('Cost gate did not pass; do not proceed with this backend')
    rank,world,local=(int(os.getenv(k,d)) for k,d in (('RANK',0),('WORLD_SIZE',1),('LOCAL_RANK',0)))
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    if world>1:
        dist.init_process_group('nccl',device_id=device)
    jobs=load_prompts(args.data,args.role,args.prompts,args.seed)
    if world>len(jobs):
        raise ValueError('Use at most one GPU per prompt')
    if rank==0:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Output must be new/empty')
        args.output.mkdir(parents=True,exist_ok=True)
    if world>1: dist.barrier()
    probe=kernel_probe(device)
    import transformers
    with torch.inference_mode():
        reference=transformers.AutoModelForCausalLM.from_pretrained(snapshot(),trust_remote_code=True,
                  local_files_only=True,dtype=torch.bfloat16).to(device).eval()
        reference.requires_grad_(False)
        low=copy.deepcopy(reference)
        backend=convert(low)
        low.requires_grad_(False)
        semantics=dict(model=MODEL_ID,revision=REVISION,backend=backend,
            prompt_manifest_sha256=sha256(args.data/'manifest.json'),
            block_size=32,small_block_size=8,batch_size=1,use_block_cache=False,
            mask_id=151665,stop_token=151645,
            threshold=args.threshold,max_new_tokens=args.max_new_tokens,temperature=0,
            torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(device),deterministic=True,
            score='ideal_logit_action_radius_v1_NOT_a_BF16_certificate',
            source_hashes={name:sha256(Path(__file__).parent/name) for name in ('probe.py','quant.py','decision.py','common.py')})
        config_hash=digest(semantics)
        if cost is not None and cost['config_hash']!=config_hash:
            raise ValueError('Cost report config differs (including output length/backend/runtime/source); rerun cost')
        options=dict(max_new_tokens=args.max_new_tokens,block_size=32,small_block_size=8,
                     threshold=args.threshold,temperature=0,use_block_cache=False,mask_id=151665,stop_token=151645)
        warm=torch.tensor([jobs[rank]['ids']],device=device)
        bounded_generate(reference,warm,{**options,'max_new_tokens':64})
        bounded_generate(low,warm,{**options,'max_new_tokens':64})
        rows=[]
        for index in range(rank,len(jobs),world):
            job=jobs[index]; ids=torch.tensor([job['ids']],device=device)
            expected,native_seconds=measured(lambda:bounded_generate(reference,ids.clone(),options))
            with PairedObserver(reference,low,stage=args.stage,threshold=args.threshold,
                                repeats=args.repeats,max_states=args.max_states) as observer:
                output=bounded_generate(reference,ids.clone(),options)
            if not torch.equal(expected,output):
                raise AssertionError('Paired observation changed BF16 native generation')
            static=None
            if args.stage=='audit':
                static_output,static_seconds=measured(lambda:bounded_generate(low,ids.clone(),options))
                static=dict(seconds=static_seconds,same_output_as_bf16=torch.equal(expected,static_output),
                            generated_tokens=static_output.shape[-1]-ids.shape[-1],
                            note='Static low ALL calls incl cache writes. Output agreement only; no task accuracy.')
                if len(observer.rows)!=observer.normal_calls:
                    raise AssertionError('Incomplete trajectory cannot be calibrated')
            row=dict(prompt_id=job['id'],role=args.role,complete=args.stage=='audit',
                native_seconds=native_seconds,calls=observer.calls,normal_calls=observer.normal_calls,
                verified_next_inputs=observer.verified,same_native_output=True,
                generated_tokens=output.shape[-1]-ids.shape[-1],states=observer.rows,static_low=static,
                risk_score=max((s['score'] for s in observer.rows if not s['equal']),default=0.0))
            write_json(args.output/f'prompts/{job["id"]}.json',row)
            rows.append(row)
            print(f'rank={rank} prompt={index+1}/{len(jobs)} normal={observer.normal_calls} '
                  f'paired={len(observer.rows)} mismatches={sum(not s["equal"] for s in observer.rows)}',flush=True)
        write_json(args.output/f'rank_{rank}.json',dict(config_hash=config_hash,rows=rows))
    if world>1: dist.barrier()
    if rank==0:
        reports=[json.loads((args.output/f'rank_{i}.json').read_text()) for i in range(world)]
        if any(r['config_hash']!=config_hash for r in reports):
            raise AssertionError('Workers used different hardware/runtime configurations')
        all_rows=[row for r in reports for row in r['rows']]
        if len(all_rows)!=len(jobs) or len({r['prompt_id'] for r in all_rows})!=len(jobs):
            raise AssertionError('Missing or duplicate prompts')
        report=dict(status='complete',stage=args.stage,role=args.role,config=semantics,config_hash=config_hash,
            kernel_probe=probe,source=str(args.data),
            source_hashes={n:sha256(args.data/n) for n in ('manifest.json','train.json','heldout.json')},
            peak_allocated_gib_rank0=torch.cuda.max_memory_allocated()/2**30,
            rows=all_rows,online_executor_implemented=False,trajectory_risk_guarantee_established=False)
        if args.stage=='cost':
            report['cost']=cost_summary(all_rows)
        else:
            report['cost_report_sha256']=sha256(args.cost_report)
            report['diagnostic']=dict(requests=len(all_rows),
                requests_with_any_low_action_difference=sum(any(not s['equal'] for s in r['states']) for r in all_rows),
                note='Full BF16-reference trajectories; no online acceptance decisions or task quality claim.')
            if args.role=='development':
                from .calibrate import development_curve
                curve=development_curve(all_rows)
                write_json(args.output/'development_risk_coverage.json',curve)
                import csv
                with (args.output/'development_risk_coverage.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(curve[0]))
                    writer.writeheader(); writer.writerows(curve)
        write_json(args.output/'summary.json',report)
        print(json.dumps(report.get('cost',report.get('diagnostic'))),flush=True)
    if world>1: dist.destroy_process_group()


if __name__=='__main__':
    main()
