"""Finite offline temporal batching ceiling; no speculative executor.

Full native states are reconstructed from saved normal releases. DualCache
states are captured from an unchanged normal trajectory. Future oracle states
are paid diagnostics, not a free drafter. BF16 batching is audited, not assumed
equivalent to original B1. No runtime/decoding/model setting is deployed.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from .temporal_batch import (MASK_ID,BLOCK,reconstruct_states,window_indices,
                            compact_head,extract_block,summary_times)
from .batch_verify_probe import decision

CONFIG=dict(widths=[1,2,4,8,16],repeats=7,warmups=2,length=256,threshold=.9,
            tasks=['humaneval','mbpp','math'],sample_seed=51713,offset=0,
            full_window='16 consecutive saved native states, starting min(N//3,N-16)',
            dual_window='Longest ordinary block; ties lowest block index; first <=16 states',
            head_modes=['full_vocab_all_positions','current_block_only'],seed=1234)


def cpu_cache(past):return [tuple(t.detach().cpu().clone() for t in pair) for pair in past]
def clone_cache(past,width):return [tuple(t.repeat(width,1,1,1) for t in pair) for pair in past]


def action(logits,state,targets):
    positions,values,confidence,local_next=decision(logits,state[targets],BLOCK,MASK_ID,.9)
    next_state=state.clone();next_state[targets[positions]]=values
    return targets[positions],values,confidence,next_state


def audit(reference,actual,rows,targets):
    max_error=0.;top1=[];actions=[];edges=[];logits_equal=[]
    for i,(ref,out) in enumerate(zip(reference,actual)):
        logits_equal.append(torch.equal(ref,out))
        max_error=max(max_error,float((ref.float()-out.float()).abs().max()))
        top1.append(torch.equal(ref.argmax(-1),out.argmax(-1)))
        left=action(ref,rows[i][0].cpu(),targets[i][0].cpu())
        right=action(out,rows[i][0].cpu(),targets[i][0].cpu())
        actions.append(torch.equal(left[0],right[0]) and torch.equal(left[1],right[1]))
        if i+1<len(rows):edges.append(torch.equal(right[3],rows[i+1][0].cpu()))
    return dict(states=len(actual),bitwise_logits_equal_states=sum(logits_equal),
        top1_equal_states=sum(top1),commit_action_equal_states=sum(actions),
        all_commit_actions_equal=all(actions),all_logits_bitwise_equal=all(logits_equal),
        max_logit_error=max_error,real_next_state_edges_equal=sum(edges),real_next_state_edges=len(edges))


class Workload:
    def __init__(self,model,rows,targets,compact,past=None,replace=None):
        self.model,self.rows,self.targets,self.compact=model,rows,targets,compact
        self.past,self.replace=past,replace
        self.serial_cache=clone_cache(past,1) if past is not None else None
        self.packed=torch.cat(rows,0);self.packed_targets=torch.cat(targets,0)
        self.batch_cache=None

    def prepare_batch(self):
        if self.past is not None:self.batch_cache=clone_cache(self.past,len(self.rows))

    def one(self,ids,targets,cache=None,replace=None):
        kwargs={} if cache is None else dict(past_key_values=cache,use_cache=True,replace_position=replace)
        if self.compact:
            with compact_head(self.model,targets):out=self.model(ids,**kwargs)
            return out.logits
        out=self.model(ids,**kwargs)
        return out.logits if cache is not None else extract_block(out.logits,targets)

    def raw(self,batched):
        if batched:
            replace=self.replace.expand(len(self.rows),-1) if self.replace is not None else None
            return self.one(self.packed,self.packed_targets,self.batch_cache,replace)
        out=None
        for row,target in zip(self.rows,self.targets):out=self.one(row,target,self.serial_cache,self.replace)
        return out

    def outputs(self,batched):
        if batched:
            out=self.raw(True).detach().cpu()
            return [out[i] for i in range(len(out))]
        values=[]
        for row,target in zip(self.rows,self.targets):
            values.append(self.one(row,target,self.serial_cache,self.replace)[0].detach().cpu())
        return values

    def verification(self,batched):
        # Native serial owns one mutable cache already. Speculative branches
        # must pay private full-cache allocation/copy each window, never alias.
        if batched:
            packed=torch.cat(self.rows,0)
            cache=clone_cache(self.past,len(self.rows)) if self.past is not None else None
            mask=self.replace.expand(len(self.rows),-1) if self.replace is not None else None
            logits=self.one(packed,self.packed_targets,cache,mask)
            next_state=None;prefix=True
            for i,(row,target) in enumerate(zip(self.rows,self.targets)):
                next_state=action(logits[i],row[0],target[0])[3]
                if i+1<len(self.rows):prefix=prefix and torch.equal(next_state,self.rows[i+1][0])
            return next_state,prefix
        next_state=None
        for row,target in zip(self.rows,self.targets):
            logits=self.one(row,target,self.serial_cache,self.replace)
            next_state=action(logits[0],row[0],target[0])[3]
        return next_state,None


def measure_pair(serial,parallel):
    for _ in range(CONFIG['warmups']):serial();parallel()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    wall=[[],[]];interval=[[],[]]
    for repeat in range(CONFIG['repeats']):
        for index in ([0,1] if repeat%2==0 else [1,0]):
            torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            began=time.perf_counter();start.record()
            value=(serial if index==0 else parallel)()
            end.record();torch.cuda.synchronize()
            wall[index].append(time.perf_counter()-began);interval[index].append(start.elapsed_time(end)/1000)
            del value
    return dict(serial=summary_times(wall[0]),batch=summary_times(wall[1]),
        serial_stream_interval=summary_times(interval[0]),batch_stream_interval=summary_times(interval[1]),
        matched_serial_over_batch=summary_times(wall[0])['median_seconds']/summary_times(wall[1])['median_seconds'],
        peak_gib=torch.cuda.max_memory_allocated()/2**30)


def capture_dual(model,prompt):
    from ..llada_decode import generate_dual_cache
    clean=generate_dual_cache(model,prompt,gen_length=CONFIG['length'])
    current=None;best=None;pending={};counts=[]
    def finish():
        nonlocal best
        if current is None:return
        counts.append(dict(block=current['block'],ordinary_total=current['ordinary_total']))
        if current['states'] and (best is None or current['ordinary_total']>best['ordinary_total']):
            best=dict(block=current['block'],ordinary_total=current['ordinary_total'],states=current['states'],
                logits=current['logits'],past=cpu_cache(current['past']),replace=current['replace'].cpu())
    def pre(_model,args,kwargs):
        nonlocal current
        if kwargs.get('past_key_values') is None:
            finish();pending['warm']=True;return
        current['ordinary_total']+=1
        if len(current['states'])<16:pending['state']=args[0].detach().cpu().clone()
    def post(_model,args,kwargs,out):
        nonlocal current
        if pending.pop('warm',False):
            block=len(counts);start=prompt.shape[1]+block*BLOCK
            replace=torch.zeros((1,prompt.shape[1]+CONFIG['length']),device=prompt.device,dtype=torch.bool)
            replace[:,start:start+BLOCK]=True
            current=dict(block=block,ordinary_total=0,states=[],logits=[],past=out.past_key_values,replace=replace)
        elif 'state' in pending:
            current['states'].append(pending.pop('state'));current['logits'].append(out.logits[0].detach().cpu().clone())
    handles=[model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post,with_kwargs=True)]
    began=time.perf_counter()
    try:observed=generate_dual_cache(model,prompt,gen_length=CONFIG['length']);finish()
    finally:
        for handle in handles:handle.remove()
    assert torch.equal(clean.output,observed.output) and clean.nfe==observed.nfe
    return best,dict(clean_nfe=clean.nfe,clean_seconds=clean.seconds,observation_seconds=time.perf_counter()-began,
        tokens_nfe_equal=True,block_availability=counts,
        scope='Two paid teacher trajectories. Only one best block cache retained; no full-trajectory KV dump.')


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    binding=check_binding(required=True)
    from ..llada_evaluate import load_model
    from ..llada_common import MODEL_ID,REVISION,prompt_ids
    from ..llada_backend import LLaDAAttentionBackend
    from ..common import sha256,write_json
    from .competitors import select_samples,generation_prompt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',type=Path,nargs=3,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    source={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    old=json.loads(args.reference.read_text());data=dict(zip(CONFIG['tasks'],args.datasets))
    report=dict(model=MODEL_ID,revision=REVISION,binding=binding,configuration=CONFIG,
        implementation=source,dataset_sha256={task:sha256(path) for task,path in data.items()},
        reference_sha256=sha256(args.reference),records=[],workloads=[],
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        scope='Three reused legitimate development prompts, fixed weights/BF16/FlashAttention. '
            'Offline real future states are not a drafter, end-to-end speed, task accuracy or an exactness certificate. '
            'Same serial operator/readout engineering is applied to each corresponding batch panel.',
        timing='Raw forward uses resident prepared inputs/cache and excludes sampler/drafter/initial setup. '
            'Verification includes native FP64 decisions, input packing, private batched DualCache copies '
            'and actual next-state edge checks; serial has no speculative edge checks or branch copies. '
            'CUDA event stream intervals include host enqueue gaps, not pure kernel time. '
            'Seven alternating-order repetitions, two warmups, no graphs/fused kernels/new settings.')
    write_json(args.output/'diagnostic.json',report)
    model,tokenizer=load_model('cuda:0');torch.set_num_threads(1);torch.manual_seed(CONFIG['seed'])
    with LLaDAAttentionBackend(model,'flash') as backend:
        for task,path in data.items():
            sample=select_samples(path,1,0)[0];ident=sample.get('id',sample.get('task_id'))
            ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
            prior=next(p for p in old['prompts'] if p['task']==task and str(p['id'])==str(ident))
            all_states,all_targets=reconstruct_states(ids,prior['trace'],prior['generated_token_ids'])
            indices=window_indices(len(all_states))
            full_rows=[torch.tensor([all_states[i]],device=model.device) for i in indices]
            full_targets=[torch.tensor([all_targets[i]],device=model.device) for i in indices]
            # Native B1 reference, outside timing. Its recorded release must be
            # reproduced before trying batch shape/readout engineering.
            reference=[]
            for row,target,index in zip(full_rows,full_targets,indices):
                out=extract_block(model(row).logits,target)[0].cpu();reference.append(out)
                pos,val,_,_=action(out,row[0].cpu(),target[0].cpu())
                expected=prior['trace'][index]
                assert pos.tolist()==[len(ids)+p for p in expected['commit_positions']] and val.tolist()==expected['commit_values']
            workload=dict(task=task,id=ident,prompt_tokens=len(ids),full_sequence_tokens=len(ids)+256,
                full_real_state_indices=indices,full_native_actions_match_saved=True)
            report['workloads'].append(workload)
            for panel in ('full_vocab_all_positions','current_block_only','dual_cache_native'):
                if panel=='dual_cache_native':
                    del full_rows,full_targets,reference;gc.collect();torch.cuda.empty_cache()
                    best,teacher=capture_dual(model,torch.tensor([ids],device=model.device))
                    workload['dual_teacher']=teacher
                    if best is None:
                        report['records'].append(dict(task=task,id=ident,panel=panel,status='No ordinary states; not substituted'))
                        write_json(args.output/'diagnostic.json',report);continue
                    rows=[state.to(model.device) for state in best['states']]
                    targets=[torch.arange(BLOCK,device=model.device)[None,:] for _ in rows]
                    past=[tuple(t.to(model.device) for t in pair) for pair in best['past']]
                    past_versions=[t._version for pair in past for t in pair]
                    replace=best['replace'].to(model.device);ref=best['logits']
                    workload['dual_selected_block']=best['block'];workload['dual_available_real_states']=len(rows)
                    compact=False
                else:
                    rows,targets,ref=full_rows,full_targets,reference;past=replace=None
                    compact=panel=='current_block_only'
                t1=None
                for width in CONFIG['widths']:
                    record=dict(task=task,id=ident,panel=panel,width=width)
                    report['records'].append(record)
                    if width>len(rows):
                        record.update(status='Insufficient real consecutive states; not duplicated',available=len(rows))
                        write_json(args.output/'diagnostic.json',report);continue
                    engine=None
                    try:
                        torch.cuda.synchronize();began=time.perf_counter()
                        engine=Workload(model,rows[:width],targets[:width],compact,past,replace);engine.prepare_batch()
                        torch.cuda.synchronize();record['prepared_setup_seconds']=time.perf_counter()-began
                        record['raw_forward']=measure_pair(lambda:engine.raw(False),lambda:engine.raw(True))
                        serial=engine.outputs(False);batched=engine.outputs(True)
                        record['serial_numeric']=audit(ref[:width],serial,rows[:width],targets[:width])
                        record['batch_numeric']=audit(ref[:width],batched,rows[:width],targets[:width])
                        if width==1:t1=record['raw_forward']['serial']['median_seconds']
                        record['B_times_T1_over_TB']=width*t1/record['raw_forward']['batch']['median_seconds']
                        # Free the prepared branch cache before timing fresh
                        # ownership, avoiding artificial two-batch peak memory.
                        engine.batch_cache=None;del serial,batched;gc.collect();torch.cuda.empty_cache()
                        record['verification']=measure_pair(lambda:engine.verification(False),lambda:engine.verification(True))
                        record['status']='measured'
                    except torch.cuda.OutOfMemoryError as error:
                        record.update(status='CUDA OOM; resource limit preserved, no retry',error=str(error),
                            allocated_gib=torch.cuda.memory_allocated()/2**30,reserved_gib=torch.cuda.memory_reserved()/2**30)
                    finally:
                        engine=None;gc.collect();torch.cuda.empty_cache()
                    write_json(args.output/'diagnostic.json',report)
                    print('CASE',json.dumps({k:record.get(k) for k in ['task','panel','width','status','B_times_T1_over_TB','batch_numeric']}),flush=True)
                if panel=='dual_cache_native':
                    # Authoritative CPU cache remains untouched. Read-only source
                    # data and private copies are released before the next prompt.
                    assert past_versions==[t._version for pair in past for t in pair],'Authoritative cache was modified'
                    workload['dual_authoritative_cache_versions_unchanged']=True
                    del rows,targets,ref,past,replace,best;gc.collect();torch.cuda.empty_cache()
            write_json(args.output/'diagnostic.json',report)
        report['backend']=backend.report()
        assert report['backend']['torch_sdpa_calls']==0 and report['backend']['flash_calls']>0
    assert source=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},'Frozen source changed'
    report['outcome']='Hardware and numeric ceiling recorded; no speculative executor launched'
    write_json(args.output/'diagnostic.json',report);(args.output/'complete').write_text('OK\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
