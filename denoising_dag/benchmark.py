"""Server benchmark: real native states first, optional complete search decoding."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time
import torch
from .backend import Backend,MODELS,MASK_ID,prompt_ids,write_json,file_hash
from .search import Executor,search,compare
from .generation import generate


def prompts(path,limit):
    if path:
        rows=json.loads(Path(path).read_text(encoding='utf-8'))
    else:
        from datasets import load_dataset
        dataset=load_dataset('openai/gsm8k','main',split='train')
        rows=[dict(id=f'train:{i}',**dataset[i]) for i in range(limit)]
    if len(rows)<limit or len({str(row['id']) for row in rows[:limit]})!=limit:
        raise ValueError('Insufficient or duplicate development prompts')
    return rows[:limit]


def measure_search(backend,snapshot,reuse,depth,width,max_nodes,retain=False):
    backend.reset_stats()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    executor=Executor(snapshot.context,lambda states:backend.predict(snapshot,states),reuse,max_nodes)
    result=search(snapshot.ids,executor,MASK_ID,depth,width,max_nodes,trace=retain)
    torch.cuda.synchronize()
    wall=time.perf_counter()-start
    stats=backend.stats()
    stats.update(wall_seconds=wall,logical_rows=executor.logical_rows,
        reused_rows=executor.hits,key_seconds=executor.key_seconds,
        dispatch_seconds=executor.dispatch_seconds,per_depth=executor.layers,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    return result,stats


def probe_one(backend,snapshot,depth,width,max_nodes,repeats):
    # Untimed paired audit also warms up every shape used by both executions.
    reference,_=measure_search(backend,snapshot,False,depth,width,max_nodes,True)
    candidate,_=measure_search(backend,snapshot,True,depth,width,max_nodes,True)
    audit=compare(reference,candidate,MASK_ID)
    seen=set()
    per_depth=[]
    for level in range(depth+1):
        states=[state for d,state,_ in reference.trace if d==level]
        unique=set(states)
        per_depth.append(dict(depth=level,logical_rows=len(states),unique_rows=len(unique-seen)))
        seen.update(unique)
    trials={'tree':[],'reuse':[]}
    for repeat in range(repeats):
        order=('tree','reuse') if repeat%2==0 else ('reuse','tree')
        for name in order:
            result,stats=measure_search(backend,snapshot,name=='reuse',depth,width,max_nodes)
            stats['matches_reference_decisions']=result.decisions==reference.decisions
            trials[name].append(stats)
    valid=audit['pass_'] and all(t['matches_reference_decisions'] for group in trials.values() for t in group)
    means={name:statistics.median(t['wall_seconds'] for t in group) for name,group in trials.items()}
    gpu={name:statistics.median(t['model_gpu_seconds'] for t in group) for name,group in trials.items()}
    return dict(snapshot=snapshot.context.snapshot_id,depth=depth,width=width,
        mask_count=snapshot.ids.count(MASK_ID),logical_rows=len(reference.trace),logical_tree_nodes=reference.visits,
        unique_rows=len(seen),duplicate_fraction=1-len(seen)/len(reference.trace),per_depth=per_depth,audit=audit,trials=trials,
        median_wall_seconds=means,median_model_gpu_seconds=gpu,
        measured_search_speedup=means['tree']/means['reuse'],
        validated_search_speedup=means['tree']/means['reuse'] if valid else None,
        semantic_check_pass=valid,
        scope='Same experimental lookahead policy and same max batch; not speedup over native v2 generation')


def run_probe(backend,rows,args):
    records=[]
    for sample in rows:
        print('CAPTURE_NATIVE_STATES',sample['id'],flush=True)
        ids=prompt_ids(backend.tokenizer,sample['question'])
        _,snapshots,_=backend.native(ids,args.max_new_tokens,args.threshold,args.snapshots,str(sample['id']))
        if not snapshots:
            raise RuntimeError(f'No eligible native denoising snapshot: {sample["id"]}')
        for snapshot in snapshots:
            numerics=backend.numerical_check(snapshot)
            if not numerics['singleton_native_exact']:
                raise AssertionError('Frozen-cache singleton forward differs from captured official forward')
            for depth in args.depths:
                result=probe_one(backend,snapshot,depth,args.width,args.max_nodes,args.repeats)
                result.update(prompt_id=sample['id'],prompt_tokens=len(ids),batch_numerics=numerics)
                append(args.output/'records.jsonl',result)
                records.append(result)
                print('PROBE',sample['id'],snapshot.context.snapshot_id,'depth',depth,
                      'duplicate',round(result['duplicate_fraction'],4),
                      'search_speedup',round(result['measured_search_speedup'],3),
                      'semantic_pass',result['semantic_check_pass'],flush=True)
    return records


def measure_generation(backend,ids,method,args):
    backend.reset_stats()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    if method=='native':
        tokens,_,native_info=backend.native(ids,args.max_new_tokens,args.threshold)
        result=dict(tokens=tokens,**native_info)
    else:
        result=generate(backend,ids,method=='reuse',args.depths[-1],args.width,args.max_new_tokens,args.max_nodes)
    torch.cuda.synchronize()
    wall=time.perf_counter()-start
    stats=backend.stats() if method!='native' else {}
    result.update(stats,wall_seconds=wall,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    return result


def run_generation(backend,rows,args):
    from relation_block.evaluate import answer
    records=[]
    for sample in rows:
        ids=prompt_ids(backend.tokenizer,sample['question'])
        trials=defaultdict(list)
        for repeat in range(args.repeats):
            order=('native','tree','reuse') if repeat%2==0 else ('reuse','tree','native')
            for method in order:
                result=measure_generation(backend,ids,method,args)
                text=backend.tokenizer.decode(result['tokens'],skip_special_tokens=True)
                predicted,target=answer(text),answer(sample['answer'],gold=True)
                result.update(prediction=text,extracted=predicted,target=target,
                              correct=predicted is not None and predicted==target)
                trials[method].append(result)
                print('GENERATE',sample['id'],method,'seconds',round(result['wall_seconds'],3),
                      'correct',result['correct'],flush=True)
        ref=trials['tree'][0]
        valid=all(t['tokens']==ref['tokens'] and t['decisions']==ref['decisions']
                  for m in ('tree','reuse') for t in trials[m])
        means={m:statistics.median(t['wall_seconds'] for t in ts) for m,ts in trials.items()}
        record=dict(prompt_id=sample['id'],prompt_tokens=len(ids),trials=dict(trials),semantic_check_pass=valid,
                    median_wall_seconds=means,measured_search_speedup=means['tree']/means['reuse'],
                    validated_search_speedup=means['tree']/means['reuse'] if valid else None,
                    measured_speedup_vs_native=means['native']/means['reuse'],
                    note='Native threshold decoder differs from bounded lookahead. Compare accuracy and truncation alongside latency.')
        append(args.output/'records.jsonl',record)
        records.append(record)
    return records


def append(path,value):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n')


def summarize(records,mode):
    groups=defaultdict(list)
    for r in records:
        groups[str(r['depth']) if mode=='probe' else 'generation'].append(r)
    summary={}
    for name,group in groups.items():
        valid=all(r['semantic_check_pass'] for r in group)
        times={m:sum(r['median_wall_seconds'][m] for r in group) for m in ('tree','reuse')}
        row=dict(cases=len(group),semantic_check_pass=valid,
                 tree_seconds=times['tree'],reuse_seconds=times['reuse'],
                 measured_speedup=times['tree']/times['reuse'],
                 validated_speedup=times['tree']/times['reuse'] if valid else None)
        if mode=='probe':
            logical=sum(r['logical_rows'] for r in group)
            unique=sum(r['unique_rows'] for r in group)
            row.update(logical_rows=logical,unique_rows=unique,duplicate_fraction=1-unique/logical)
        else:
            native=sum(r['median_wall_seconds']['native'] for r in group)
            row['speedup_vs_native']=native/times['reuse']
            row['accuracy']={m:sum(r['trials'][m][0]['correct'] for r in group)/len(group) for m in ('native','tree','reuse')}
            row['truncation_rate']={m:sum(r['trials'][m][0]['truncated'] for r in group)/len(group) for m in ('native','tree','reuse')}
            row['mean_generated_tokens']={m:sum(len(r['trials'][m][0]['tokens']) for r in group)/len(group) for m in ('native','tree','reuse')}
        summary[name]=row
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['probe','generate'],default='probe')
    p.add_argument('--size',choices=MODELS,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prompts',type=Path)
    p.add_argument('--limit',type=int,default=4)
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--depths',default='1,2,3')
    p.add_argument('--width',type=int,default=4)
    p.add_argument('--max-nodes',type=int,default=4096)
    p.add_argument('--snapshots',type=int,default=2)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--threshold',type=float,default=.9)
    args=p.parse_args()
    args.depths=[int(d) for d in args.depths.split(',')]
    if (min(args.limit,args.batch_size,args.width,args.max_nodes,args.snapshots,args.repeats)<1 or
        not args.depths or min(args.depths)<1 or args.max_new_tokens<32 or args.max_new_tokens%32 or
        not 0<=args.rank<args.world_size or not 0<args.threshold<=1):
        raise ValueError('Invalid experiment settings (generation cap must be a multiple of 32)')
    if max(sum(args.width**j for j in range(d+1)) for d in args.depths)>args.max_nodes:
        raise ValueError('Requested search exceeds max-nodes')
    rows=prompts(args.prompts,args.limit)[args.rank::args.world_size]
    if not rows:
        raise ValueError('Every worker needs at least one prompt; increase limit or use fewer GPUs')
    if args.output.exists():
        raise ValueError('Use a fresh worker output directory')
    args.output.mkdir(parents=True)
    torch.manual_seed(1234)
    print('LOAD',args.size,'device',torch.cuda.get_device_name(),'prompts',len(rows),flush=True)
    backend=Backend.load(args.size,batch_size=args.batch_size)
    warmup=prompt_ids(backend.tokenizer,'Explain step by step how to multiply seventeen by twenty-three.')
    _,snapshots,_=backend.native(warmup,64,args.threshold,1,'warmup')
    if not snapshots:
        raise RuntimeError('Native warmup produced no denoising state')
    gate=backend.numerical_check(snapshots[0])
    print('BACKEND_CHECK',json.dumps(gate),flush=True)
    write_json(args.output/'backend_check.json',gate)
    if not gate['singleton_native_exact']:
        raise AssertionError('Official native-cache parity failed')
    if args.mode=='generate':
        for reuse in (False,True):
            generate(backend,warmup,reuse,args.depths[-1],args.width,32,args.max_nodes)
    del snapshots
    records=run_probe(backend,rows,args) if args.mode=='probe' else run_generation(backend,rows,args)
    import transformers
    summary=dict(mode=args.mode,size=args.size,model=MODELS[args.size],
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,
        prompt_ids=[r['id'] for r in rows],groups=summarize(records,args.mode),
        source_hashes={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},
        limitations='Exploratory GSM8K development prompts. Probe measures search only, not faster text generation. Same policy decisions are checked, not claimed from approximate logits alone.')
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary['groups'],indent=2),flush=True)


if __name__=='__main__':
    main()
