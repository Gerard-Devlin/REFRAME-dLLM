from pathlib import Path
import torch
from .core import (read,write,digest,prompt_splits,optimal_savings,windows,OPTIONS,
                   implementation_hash,validate_record)
from .model import load,add_lora,enable_checkpointing,hidden_for,adapters
from .trajectory import generate,synchronize
from .runtime import gather,barrier,gate

def splits(args,tokenizer):
    from competitor_budget.evaluate import prompt_ids
    return prompt_splits(args.data,[prompt_ids(tokenizer,r['question']) for r in read(args.dataset)])

def parity(teacher,student,record):
    """Same call shapes; differentiate through inference semantics, not train()."""
    from .model import GraphCache
    device=next(teacher.parameters()).device
    tc=sc=None; results=[]
    calls=[(x,True) for x in record['history']]+[(record['canvas'],False)]
    for ids,update in calls:
        x=torch.tensor([ids],device=device)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            t=teacher(x,use_cache=True,past_key_values=tc,update_past_key_values=update)
        if sc is None: sc=GraphCache()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s=student(x,use_cache=True,past_key_values=sc,update_past_key_values=update)
        error=(t.logits.float()-s.logits.float()).square().mean().sqrt()
        scale=t.logits.float().square().mean().sqrt().clamp_min(1e-8)
        rms=float(error/scale); agreement=float((t.logits.argmax(-1)==s.logits.argmax(-1)).float().mean())
        if rms>1e-6 or agreement!=1.: raise AssertionError(f'Zero-LoRA native path parity failed: {rms}')
        results.append(dict(cache_write=update,length=len(ids),relative_rms=rms,top1=agreement))
        tc=t.past_key_values
        del s,t
    # Full path checked separately: no claim full/cached numerics are identical.
    ids=[v for chunk in record['history'] for v in chunk]+record['canvas']
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=teacher(torch.tensor([ids],device=device),use_cache=False).logits
        b=student(torch.tensor([ids],device=device),use_cache=False).logits
    if not torch.equal(a,b): raise AssertionError('Zero-LoRA full path differs')
    return results

def run_audit(args,config,tokenizer,rank,world):
    rows=splits(args,tokenizer)['audit']; model=load(); local=[]
    # Warm actual inference kernels; never include warmup in measured native time.
    generate(model,rows[rank%len(rows)]['ids'])
    special=set(tokenizer.all_special_ids)
    for i,row in enumerate(rows):
        if i%world!=rank: continue
        native=generate(model,row['ids']); observed=generate(model,row['ids'],observe=True)
        if native['tokens']!=observed['tokens']: raise AssertionError('Observer changed output')
        observed['native_seconds']=native['seconds']
        observed['saved_forward_seconds']=optimal_savings(observed['calls'],special)
        local.append(observed)
        write(args.output/f'prompt_{row["id"]}.json',observed)
        print(f'audit rank={rank} prompt={i+1}/32 calls={native["counts"]["calls"]}',flush=True)
    all_rows=gather(local)
    # Zero-LoRA validation uses local trajectory before any optimizer update.
    candidates=[r for row in local for r in windows(row,special) if r['history']]
    if not candidates: raise RuntimeError('No local legal windows; cannot check training parity')
    student=load(); add_lora(student); enable_checkpointing(student)
    checks=parity(model,student,candidates[0])
    with torch.autocast('cuda',dtype=torch.bfloat16):
        h,student_cache=hidden_for(student,candidates[0],return_cache=True)
        h.float().square().mean().backward()
    nonzero=sum(p.grad is not None and bool(p.grad.abs().max()>0) for p in student.parameters() if p.requires_grad)
    if not nonzero: raise AssertionError('LoRA has no gradient')
    cache_grads=sum(x.grad is not None and bool(x.grad.abs().max()>0)
                    for pair in student_cache.entries for x in pair)
    if cache_grads!=2*len(student_cache.entries):
        raise AssertionError('Student history cache does not carry complete gradients')
    parity_rows=gather([dict(rank=rank,paths=checks,nonzero_adapter_gradients=nonzero,
                             differentiable_cache_tensors=cache_grads)])
    if rank==0:
        total=sum(c['seconds'] for r in all_rows for c in r['calls'])
        saved=sum(r['saved_forward_seconds'] for r in all_rows)
        ceiling=total/(total-saved) if total>saved else 1.
        kinds={k:dict(calls=sum(c['kind']==k for r in all_rows for c in r['calls']),
            seconds=sum(c['seconds'] for r in all_rows for c in r['calls'] if c['kind']==k))
            for k in ('denoise','prefill','cache_write')}
        summary=dict(stage='audit',config=config,pass_=ceiling>=1.5,
            forward_cost_ceiling=ceiling,total_forward_seconds=total,savable_forward_seconds=saved,
            native_seconds=sum(r['native_seconds'] for r in all_rows),categories=kinds,
            parity=parity_rows,prompts=len(all_rows),
            note='Optimistic forward-only matching model; not measured end-to-end acceleration.')
        summary['pass']=summary.pop('pass_'); write(args.output/'summary.json',summary)
        print(summary,flush=True)

def run_collect(args,config,tokenizer,rank,world):
    gate(args.audit,config); pool=splits(args,tokenizer); model=load()
    special=set(tokenizer.all_special_ids); metadata=[]
    for split in ('train','validation'):
        rows=pool[split]; n=0; seconds=0.
        for i,row in enumerate(rows):
            if i%world!=rank: continue
            trace=generate(model,row['ids'],observe=True)
            records=windows(trace,special); n+=len(records); seconds+=trace['seconds']
            write(args.output/split/f'{row["id"]}.json',dict(prompt_id=row['id'],records=records,
                  teacher_seconds=trace['seconds'],generated_tokens=trace['generated_tokens']))
            print(f'collect {split} rank={rank} {i+1}/{len(rows)} windows={len(records)}',flush=True)
        metadata.append(dict(split=split,rank=rank,states=n,teacher_seconds=seconds))
    meta=gather(metadata)
    if rank==0:
        write(args.output/'manifest.json',dict(config=config,audit_hash=digest(read(args.audit)),
            splits={s:[r['id'] for r in pool[s]] for s in ('train','validation')},metadata=meta))
    barrier()

def records(path,split):
    manifest=read(Path(path)/'manifest.json'); rows=[]
    for key in manifest['splits'][split]:
        item=read(Path(path)/split/f'{key}.json')
        if item['prompt_id']!=key: raise ValueError('Corrupt prompt shard')
        for r in item['records']:
            validate_record(r)
            if r['prompt_id']!=key: raise ValueError('Wrong prompt grouping')
            rows.append(r)
    if not rows: raise ValueError('No legal distillation states')
    return rows
