import math
import random
import shutil
import time
from pathlib import Path
import torch
from .core import read,write,OPTIONS,EOS,MASK,SMALL,digest
from .model import load,add_lora,load_adapters,merge,snapshot,hidden_for
from .trajectory import generate,choose,synchronize
from .runtime import gather,gsm_rows,barrier

def interval(diffs,seed=1234):
    rng=random.Random(seed); n=len(diffs)
    means=sorted(sum(rng.choices(diffs,k=n))/n for _ in range(2000))
    return [means[49],means[1949]]

def summary(rows):
    if not rows: raise ValueError('Empty evaluation')
    timings=sorted(r['seconds'] for r in rows); n=len(rows)
    return dict(examples=n,accuracy=sum(r['correct'] for r in rows)/n,
        mean_seconds=sum(timings)/n,p95_seconds=timings[math.ceil(.95*n)-1],
        total_nfe=sum(r['counts']['calls'] for r in rows),
        mean_denoise_per_block=sum(r['counts']['denoise'] for r in rows)/
            max(1,sum(r['counts']['cache_write']+1 for r in rows)),
        mean_generated_tokens=sum(r['generated_tokens'] for r in rows)/n,
        truncation_rate=sum(r['length_capped'] for r in rows)/n,
        eos_rate=sum(EOS in r['tokens'] for r in rows)/n)

def compare(reference,candidate):
    ref={r['sample_id']:r for r in reference}; cand={r['sample_id']:r for r in candidate}
    if set(ref)!=set(cand): raise ValueError('Evaluation shards differ')
    a,b=summary(reference),summary(candidate)
    speed=a['mean_seconds']/b['mean_seconds']; delta=b['accuracy']-a['accuracy']
    trunc=b['truncation_rate']-a['truncation_rate']
    return dict(speedup=speed,accuracy_delta=delta,truncation_delta=trunc,
        paired_accuracy_interval=interval([int(cand[k]['correct'])-int(ref[k]['correct']) for k in sorted(ref)]),
        screening_pass=speed>=1.2 and delta>=-.02 and trunc<=.02,
        note='Development screening only; not a lossless or population-level guarantee')

@torch.no_grad()
def evaluate_models(models,tokenizer,dataset,rank,world,thresholds=(.90,),split='dev'):
    from competitor_budget.evaluate import prompt_ids,extract_answer
    rows=gsm_rows(dataset,split); local=[]
    names=list(models)
    for model in models.values(): generate(model,prompt_ids(tokenizer,rows[rank%len(rows)]['question']))
    for i,row in enumerate(rows):
        if i%world!=rank: continue
        ids=prompt_ids(tokenizer,row['question'])
        order=names[(i//world)%len(names):]+names[:(i//world)%len(names)]
        for threshold in thresholds:
            for name in order:
                result=generate(models[name],ids,dict(OPTIONS,threshold=threshold))
                result.pop('calls'); result.update(sample_id=row['sample_id'],model=name,threshold=threshold)
                answer=extract_answer(tokenizer.decode(result['tokens'],skip_special_tokens=True))
                result['correct']=answer==extract_answer(row['answer'],gold=True)
                local.append(result)
        print(f'evaluate rank={rank} {i+1}/{len(rows)}',flush=True)
    all_rows=gather(local); metrics={}; pairs={}
    for name in names:
        for t in thresholds:
            selected=[r for r in all_rows if r['model']==name and r['threshold']==t]
            if len(selected)!=len(rows) or len({r['sample_id'] for r in selected})!=len(rows):
                raise AssertionError('Incomplete/duplicate evaluation shards')
            metrics[f'{name}@{t}']=summary(selected)
    reference=[r for r in all_rows if r['model']=='original' and r['threshold']==.90]
    if reference:
        for name in names:
            for t in thresholds:
                if name!='original': pairs[f'{name}@{t}']=compare(reference,[r for r in all_rows if r['model']==name and r['threshold']==t])
    return dict(split=split,metrics=metrics,comparisons=pairs,rows=all_rows)

@torch.no_grad()
def mechanism(student,teacher,rows,rank,world):
    local=[]
    for i,r in enumerate(rows):
        if i%world!=rank: continue
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s=student.lm_head(hidden_for(student,r))[r['start']:r['start']+SMALL]
            t=teacher.lm_head(hidden_for(teacher,r))[r['start']:r['start']+SMALL]
        action=choose(s,r['canvas'],r['start']); expected=dict(r['first']+r['second'])
        got=dict(action); second=dict(r['second'])
        other=[p-r['start'] for p in range(r['start'],r['start']+SMALL) if r['canvas'][p]==MASK and p not in expected]
        kl=0.
        if other:
            tl=t[other].float().log_softmax(-1); sl=s[other].float().log_softmax(-1)
            kl=float((tl.exp()*(tl-sl)).sum())
        teacher_action=choose(t,r['canvas'],r['start'])
        if teacher_action!=r['first']: raise AssertionError('Reconstructed teacher action differs from collected action')
        local.append(dict(exact=int(got==expected),second_correct=sum(got.get(p)==v for p,v in second.items()),
            second_total=len(second),wrong=sum(expected.get(p)!=v for p,v in action),commits=len(action),
            kl=kl,other=len(other),base_exact=int(dict(teacher_action)==expected)))
    all_rows=gather(local); n=len(all_rows)
    return dict(states=n,set_exact=sum(x['exact'] for x in all_rows)/n,
        teacher_one_step_set_exact=sum(x['base_exact'] for x in all_rows)/n,
        second_release=sum(x['second_correct'] for x in all_rows)/sum(x['second_total'] for x in all_rows),
        extra_wrong_rate=sum(x['wrong'] for x in all_rows)/max(1,sum(x['commits'] for x in all_rows)),
        preservation_kl=sum(x['kl'] for x in all_rows)/max(1,sum(x['other'] for x in all_rows)))

def merged_copy(state):
    model=load(); add_lora(model); load_adapters(model,state); merge(model); return model

@torch.no_grad()
def fixed_forward_seconds(model,record,repeats=20):
    """Same state, cache, and shape; excludes prefix construction."""
    from .model import GraphCache
    cache=GraphCache(); device=next(model.parameters()).device
    for chunk in record['history']:
        model(input_ids=torch.tensor([chunk],device=device),use_cache=True,past_key_values=cache,
              update_past_key_values=True,block_size=32)
    x=torch.tensor([record['canvas']],device=device)
    def call():
        return model(input_ids=x,use_cache=True,past_key_values=cache,
                     update_past_key_values=False,block_size=32)
    call(); synchronize(); values=[]
    for _ in range(repeats):
        synchronize(); started=time.perf_counter(); call(); synchronize()
        values.append(time.perf_counter()-started)
    return sorted(values)[len(values)//2]

@torch.no_grad()
def export_checkpoint(checkpoint_path,output,tokenizer,record):
    checkpoint_path=Path(checkpoint_path)
    if checkpoint_path.is_dir(): checkpoint_path=checkpoint_path/'rank0.pt'
    ckpt=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model=load(); add_lora(model); load_adapters(model,ckpt['adapter'])
    probe=tokenizer.apply_chat_template([{'role':'user','content':'Briefly compute 2 + 3.'}],
                                        tokenize=True,add_generation_prompt=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        before=model.lm_head(hidden_for(model,record)).float()
        native_before=generate(model,probe,dict(OPTIONS,max_new_tokens=64))
    merge(model)
    model.config.step_distill=ckpt['config']
    after=model.lm_head(hidden_for(model,record)).float()
    rms=float((before-after).square().mean().sqrt()/before.square().mean().sqrt().clamp_min(1e-8))
    top=float((before.argmax(-1)==after.argmax(-1)).float().mean())
    if not math.isfinite(rms) or rms>.02 or top<.99: raise AssertionError(f'Merge numerical check failed: {rms}, {top}')
    native_after=generate(model,probe,dict(OPTIONS,max_new_tokens=64))
    output=Path(output)
    if output.exists(): raise ValueError('Export path already exists')
    model.save_pretrained(output,safe_serialization=True); tokenizer.save_pretrained(output)
    for source in snapshot().glob('*.py'): shutil.copy2(source,output/source.name)
    del model; torch.cuda.empty_cache()
    reloaded=load(output)
    reloaded_logits=reloaded.lm_head(hidden_for(reloaded,record)).float()
    if not torch.equal(after,reloaded_logits): raise AssertionError('Export reload changed logits')
    a=choose(before[record['start']:record['start']+SMALL].bfloat16(),record['canvas'],record['start'])
    b=choose(after[record['start']:record['start']+SMALL].bfloat16(),record['canvas'],record['start'])
    report=dict(relative_rms=rms,top1=top,action_equal=a==b,
        short_generation_equal=native_before['tokens']==native_after['tokens'],reload_logits_equal=True,
        note='BF16 merge rounding can change decisions; final quality measured on merged export.')
    write(output/'merge_check.json',report); return report
