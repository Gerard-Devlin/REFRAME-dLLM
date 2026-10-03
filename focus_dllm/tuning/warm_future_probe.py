"""Read-only global future predictions from already-paid native full warm calls.

Noncontiguous positions in every later block are observed, unlike an immediate
next-block stable-prefix probe. No future prediction changes the teacher input.
Later teacher tokens are diagnostic references, never labels supplied to a model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def reference_schedule(events, final_tokens, block, thresholds, special_ids, eos_id):
    """Optimistic reference-schedule counts, NOT an online speed bound.

    Correct-only filtering deliberately uses the completed teacher as an oracle.
    Changed conditioning can change later actions, so emptied reference actions
    cannot be counted as calls saved by a deployed method.
    """
    if block<=0 or len(final_tokens)%block or not events:
        raise ValueError('Complete fixed-block trajectory required')
    if list(range(len(events)))!=[e['call'] for e in events]:
        raise ValueError('Missing or unordered teacher calls')
    if any(not 0<t<1 for t in thresholds):
        raise ValueError('Invalid probability threshold')
    first_eos=next((i for i,t in enumerate(final_tokens) if t==eos_id),len(final_tokens))
    results={}
    for threshold in thresholds:
        earliest, ready, released = {}, {}, set()
        counters=dict(eligible_predictions=0, special_predictions=0, repeated_conflicts=0,
            unique_eligible_positions=0, earliest_matches=0, earliest_wrong=0,
            pre_eos_positions=0, pre_eos_matches=0, post_eos_positions=0,
            oracle_emptied_refine_calls=0, oracle_emptied_refine_before_eos=0,
            oracle_fully_ready_blocks=0, oracle_fully_ready_blocks_before_eos=0)
        for event in events:
            start=event['block']*block
            if event['kind']!='warm' and event.get('future'):
                raise ValueError('Only already-paid full warm calls expose observations')
            positions=event['positions']; tokens=event['tokens']
            if not positions or len(positions)!=len(tokens) or len(set(positions))!=len(positions):
                raise ValueError('Native calls must have a nonempty unique commit set')
            if any(not start<=p<start+block or p in released for p in positions):
                raise ValueError('Invalid or duplicate native release')
            if any(final_tokens[p]!=t for p,t in zip(positions,tokens)):
                raise ValueError('Teacher final tokens do not match the commit ledger')
            if event['kind']=='refine' and all(p in ready for p in positions):
                counters['oracle_emptied_refine_calls']+=1
                counters['oracle_emptied_refine_before_eos']+=int(any(p<first_eos for p in positions))
            elif event['kind']=='warm':
                if all(p in ready for p in range(start,start+block)):
                    counters['oracle_fully_ready_blocks']+=1
                    counters['oracle_fully_ready_blocks_before_eos']+=int(start+block<=first_eos)
            else:
                if event['kind']!='refine':
                    raise ValueError('Unknown call type')
            # New observations from this call can only affect LATER blocks.
            for prediction in event.get('future',[]):
                p,t,c=prediction['position'],prediction['token'],prediction['confidence']
                if not start+block<=p<len(final_tokens) or p in released:
                    raise ValueError('Observation crosses its future boundary')
                if not 0<=c<=1:
                    raise ValueError('Invalid confidence')
                if c<threshold:
                    continue
                if t in special_ids:
                    counters['special_predictions']+=1
                    continue
                counters['eligible_predictions']+=1
                if p in earliest:
                    counters['repeated_conflicts']+=int(earliest[p]!=t)
                    continue
                earliest[p]=t
                match=t==final_tokens[p]
                counters['earliest_matches']+=int(match)
                counters['earliest_wrong']+=int(not match)
                if p<first_eos:
                    counters['pre_eos_positions']+=1
                    counters['pre_eos_matches']+=int(match)
                else:
                    counters['post_eos_positions']+=1
                if match:
                    ready[p]=t
            released.update(positions)
        if released!=set(range(len(final_tokens))):
            raise ValueError('Incomplete generation ledger')
        counters['unique_eligible_positions']=len(earliest)
        results[str(threshold)]=counters
    return dict(first_eos_position=first_eos, total_calls=len(events),
        warm_calls=sum(e['kind']=='warm' for e in events),
        refine_calls=sum(e['kind']=='refine' for e in events), thresholds=results)


class NativeObserver:
    """Hooks only read outputs; reconstruct every actual native commit."""
    def __init__(self, model, prompt, length, block, mask_id, threshold=.90):
        self.model,self.prompt,self.length,self.block=model,prompt,length,block
        self.mask_id,self.threshold=mask_id,threshold
        self.prefix=prompt.shape[1]
        self.expected=torch.full((1,self.prefix+length),mask_id,dtype=torch.long)
        self.expected[:,:self.prefix]=prompt.cpu()
        self.events,self.handles,self.pending=[],[],None
        self.warms=0

    def __enter__(self):
        self.handles.append(self.model.register_forward_pre_hook(self.before,with_kwargs=True))
        try:
            self.handles.append(self.model.register_forward_hook(self.after,with_kwargs=True))
        except Exception:
            self.handles[0].remove()
            raise
        return self

    def __exit__(self,*args):
        for handle in self.handles:
            handle.remove()
        self.pending=None

    def before(self, _module, args, kwargs):
        if self.pending is not None:
            raise AssertionError('Nested forward not supported')
        state=args[0] if args else kwargs['input_ids']
        past=kwargs.get('past_key_values')
        warm=past is None
        block_index=self.warms if warm else self.warms-1
        if not 0<=block_index<self.length//self.block:
            raise AssertionError('Unexpected block index')
        start=self.prefix+block_index*self.block
        offset=0 if warm else start
        if kwargs.get('replace_position') is not None:
            raise AssertionError('Probe uses PrefixCache, not mutable Dual branches')
        if not warm and past[0][0].shape[-2]!=start:
            raise AssertionError('Formal prefix boundary changed')
        if not torch.equal(state.cpu(),self.expected[:,offset:]):
            raise AssertionError('Reconstructed action ledger differs from native input')
        cache_versions=[(t.data_ptr(),t._version) for pair in past for t in pair] if past else []
        self.pending=(state.detach().clone(),offset,start,block_index,warm,past,cache_versions)

    def after(self, _module, _args, _kwargs, output):
        state,offset,start,block_index,warm,past,cache_versions=self.pending
        if past and cache_versions!=[(t.data_ptr(),t._version) for pair in past for t in pair]:
            raise AssertionError('Read-only prefix cache was mutated')
        target=(state[0,start-offset:start-offset+self.block]==self.mask_id).nonzero().flatten()+start-offset
        if target.numel()==0:
            raise AssertionError('Native called a completed block')
        logits=output.logits.index_select(1,target)
        tokens=logits.argmax(-1)
        confidence=logits.double().softmax(-1).gather(-1,tokens.unsqueeze(-1)).squeeze(-1)
        selected=confidence[0]>=self.threshold
        selected[confidence[0].argmax()]=True
        positions=(target[selected]+offset-self.prefix).tolist()
        values=tokens[0,selected].tolist()
        event=dict(call=len(self.events),block=block_index,kind='warm' if warm else 'refine',
            positions=positions,tokens=values,future=[])
        if warm:
            # Already-computed full vocabulary rows, no extra LM head/Transformer.
            # Softmax/readback overhead is paid by this DIAGNOSTIC, never free speed.
            for first in range(start+self.block,self.prefix+self.length,32):
                last=min(first+32,self.prefix+self.length)
                future=output.logits[:,first:last]
                ids=future.argmax(-1)
                probs=future.double().softmax(-1).gather(-1,ids.unsqueeze(-1)).squeeze(-1)
                eligible=(probs[0]>=.95)&(state[0,first:last]==self.mask_id)
                for row in eligible.nonzero().flatten().tolist():
                    event['future'].append(dict(position=first+row-self.prefix,
                        token=int(ids[0,row]),confidence=float(probs[0,row])))
            self.warms+=1
        self.expected[0,torch.tensor(positions)+self.prefix]=torch.tensor(values)
        self.events.append(event)
        self.pending=None


def save(path,obj):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2))
    temporary.replace(path)


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    binding=check_binding(required=True)
    from .run import load_model
    from .competitors import select_samples, generation_prompt
    from ..llada_common import MODEL_ID, REVISION, MASK_ID, prompt_ids
    from ..llada_decode import generate_prefix_cache
    from ..llada_backend import LLaDAAttentionBackend
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs=3,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(model=MODEL_ID,revision=REVISION,gpu_binding=binding,records=[],
        dataset_sha256=[digest(p) for p in args.datasets],
        implementation={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        configuration=dict(tasks=['humaneval','mbpp','math'],sample_seed=51713,offset=0,
            samples_per_task=2,lengths=[256,512],block=32,teacher_threshold=.90,
            future_thresholds=[.95,.99],bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        scope='12 reused development trajectories. Read-only reference-schedule diagnostic. '
        'No online releases, scorer edits, generation speedup, certificate or novelty claim. '
        'Correct-only counterfactual counts use teacher-final oracle filtering and are not an online upper bound.')
    save(args.output/'diagnostic.json',report)
    model,tokenizer=load_model('cuda:0')
    special=set(tokenizer.all_special_ids)|{MASK_ID}
    eos=tokenizer.eos_token_id
    if not isinstance(eos,int):
        raise ValueError('A single native EOS ID required')
    special.add(eos)
    report['special_ids']=sorted(special);report['eos_id']=eos
    torch.manual_seed(1234)
    for task,dataset in zip(('humaneval','mbpp','math'),args.datasets):
        for sample in select_samples(dataset,2,0):
            ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
            prompt=torch.tensor([ids],device=model.device)
            for length in (256,512):
                with LLaDAAttentionBackend(model,'flash') as backend:
                    clean=generate_prefix_cache(model,prompt,gen_length=length)
                    with NativeObserver(model,prompt,length,32,MASK_ID) as observer:
                        observed=generate_prefix_cache(model,prompt,gen_length=length)
                    assert torch.equal(clean.output,observed.output) and clean.nfe==observed.nfe
                    assert torch.equal(observer.expected,observed.output.cpu())
                    assert len(observer.events)==clean.nfe and observer.warms==length//32
                    assert backend.stats.torch_sdpa_calls==0
                    assert backend.stats.flash_calls==backend.stats.attention_calls
                    final=clean.output[0,len(ids):].tolist()
                    record=dict(task=task,id=sample.get('id',sample.get('task_id')),length=length,
                        teacher_tokens_match=True,teacher_nfe=clean.nfe,
                        teacher_output_sha256=hashlib.sha256(json.dumps(final).encode()).hexdigest(),
                        events=observer.events,reference_tokens=final,
                        schedule=reference_schedule(observer.events,final,32,[.95,.99],special,eos),
                        backend=vars(backend.stats),clean_seconds=clean.seconds,
                        observed_seconds=observed.seconds,timing_scope='Instrumentation time is not a native performance benchmark')
                report['records'].append(record)
                save(args.output/'diagnostic.json',report)
                print('SUMMARY',task,record['id'],length,json.dumps(record['schedule']),flush=True)
    (args.output/'complete').write_text('OK\n')


if __name__=='__main__':
    main()
