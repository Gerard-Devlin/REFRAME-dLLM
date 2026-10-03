"""Fixed execution candidate versus pinned Flash on previously unselected IDs.

One physical GPU, common BF16 model/backend, two official decoding modes,
full-request clean timing, separate traced commits and frozen scorers.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from .flash_readout import ModelReadout,generator,suppress_official_prints
from .flash_statistics import statistics as probability_statistics


def validate(offset,limit,lengths):
    if offset<64 or limit<=0 or not lengths or len(set(lengths))!=len(lengths):
        raise ValueError('Use positive unique lengths and previously unselected offset >=64')
    if any(n not in (256,512) for n in lengths):raise ValueError('Fixed 256/512 protocol required')


def paired_intervals(before,after,seconds_before,seconds_after):
    import numpy as np
    n=len(before)
    if not n or any(len(v)!=n for v in (after,seconds_before,seconds_after)):
        raise ValueError('Complete paired records required')
    if min([*seconds_before,*seconds_after])<=0:raise ValueError('Positive full-request timing required')
    rng=np.random.default_rng(1234);indices=rng.integers(0,n,(10000,n))
    delta=np.asarray(after,dtype=float)-np.asarray(before,dtype=float)
    speed=np.asarray(seconds_before)[indices].sum(1)/np.asarray(seconds_after)[indices].sum(1)
    return dict(accuracy_difference_pp=float(delta.mean()*100),
                accuracy_difference_95ci_pp=(np.percentile(delta[indices].mean(1),[2.5,97.5])*100).tolist(),
                pooled_speedup=sum(seconds_before)/sum(seconds_after),
                speedup_95ci=np.percentile(speed,[2.5,97.5]).tolist(),bootstrap_samples=10000,seed=1234,
                scope='Paired screening interval; does not establish statistical losslessness')


def canvas_metadata(actions,prompt_tokens,length):
    # Only the official canvas commit sites supply these values; draft labels
    # are never interpreted as accepted output. Padding outside budget excluded.
    canvas=[126336]*length
    for positions,tokens in actions:
        if len(positions)!=len(tokens):raise ValueError('Incomplete canvas action')
        for position,token in zip(positions,tokens):
            index=position-prompt_tokens
            if 0<=index<length:canvas[index]=token
    eos=[i for i,t in enumerate(canvas) if t==126081]
    return dict(raw_token_ids=canvas,first_eos_in_budget=min(eos) if eos else None,
                budget_has_eos=bool(eos),budget_without_eos=not bool(eos),
                unrevealed_budget_positions=canvas.count(126336),
                provenance='Reconstructed from real official commits, excluding draft and padding')


def score_records(task,samples,records):
    from .suite import score
    methods=['official','optimized']
    original=score(task,samples,records,methods)
    if task not in ('math','gsm8k'):
        return dict(primary=original,primary_policy='Official code execution',original=original)
    from dllm_eval.score_answers import assess,policy_hash,MathComparison
    comparison=MathComparison() if task=='math' else None
    metric=None
    if task=='math':
        from dllm_eval.score_math import load_metric,installed_utils
        metric=load_metric(installed_utils())
    details={name:[] for name in methods}
    for sample,row in zip(samples,records):
        if task=='gsm8k':gold=sample['answer'].rsplit('####',1)[-1].strip()
        else:
            boxed=metric['last_boxed_only_string'](sample['solution'])
            if boxed is None:raise ValueError('Missing MATH reference')
            gold=metric['remove_boxed'](boxed)
        for name in methods:
            details[name].append(dict(id=row['id'],**assess(row[name]['text'],gold,task,comparison)))
    return dict(primary={name:[d['correct'] for d in values] for name,values in details.items()},
                primary_policy='Frozen final-expression-v3, uniform across both outputs',policy_sha256=policy_hash(),
                original=original,details=details,
                statuses={name:dict(Counter(d['status'] for d in values)) for name,values in details.items()})


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    binding=check_binding(required=True)
    from .competitors import load_external,load_model,select_samples,generation_prompt
    from ..common import write_json,sha256
    from ..llada_common import MODEL_ID,REVISION,prompt_ids
    from ..llada_evaluate import percentile
    from tqdm import tqdm
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--third-party',type=Path,required=True)
    for task in ('humaneval','gsm8k','mbpp','math'):
        parser.add_argument('--'+task+'-dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=16)
    parser.add_argument('--offset',type=int,default=64)
    parser.add_argument('--lengths',default='256,512')
    args=parser.parse_args();lengths=[int(n) for n in args.lengths.split(',')]
    validate(args.offset,args.limit,lengths)
    datasets={task:getattr(args,task+'_dataset') for task in ('humaneval','gsm8k','mbpp','math')}
    samples={task:select_samples(path,args.limit,args.offset) for task,path in datasets.items()}
    source={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    scorer=Path(__file__).resolve().parents[1]/'dllm-eval/dllm_eval'
    scorer_source={p.name:sha256(p) for p in scorer.glob('score*.py')}
    cls,external,adaptation=load_external(args.third_party,'flash_verify')
    original=Path(inspect.getsourcefile(inspect.unwrap(external)))
    identity=dict(model=MODEL_ID,revision=REVISION,binding=binding,dtype='bfloat16',batch=1,
        length=lengths,block=32,threshold=.9,gamma=.8,track_num=4,mask_num=4,
        minimum_head_rows=64,normalization='Full-vocabulary FP64 two-pass exp/sum',
        seed=1234,sample_seed=51713,offset=args.offset,limit=args.limit,
        datasets={task:sha256(path) for task,path in datasets.items()},
        ids={task:[str(s.get('id',s.get('task_id'))) for s in values] for task,values in samples.items()},
        implementation=source,scorer_implementation=scorer_source,official_generate_sha256=sha256(original),
        third_party_sources=json.loads((args.third_party/'sources.json').read_text()),adaptation=adaptation,
        scope='IDs not used for this round of configuration selection. Small screening set, not universal '
              'losslessness or full benchmark. Fixed candidate for all tasks/lengths/decoders. Engineering gain '
              'over the pinned official Flash implementation, no new algorithm or all-baselines claim.')
    manifest=args.output/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())==identity,'Resume configuration/source changed'
    else:write_json(manifest,identity)
    model,tokenizer=load_model(SimpleNamespace(method='flash_verify'),cls)
    torch.set_num_threads(1);torch.manual_seed(1234)
    compact=ModelReadout(model,compact=True,minimum=64)
    optimized=generator(external,statistics=probability_statistics)
    all_summaries={}
    for verify in (True,False):
        decoder='flash_verify' if verify else 'flash_cache'
        for task,values in samples.items():
            for length in lengths:
                cell=f'{task}_{length}_{decoder}';out=args.output/cell
                if (out/'complete').exists():
                    all_summaries[cell]=json.loads((out/'summary.json').read_text());continue
                def run(ids,name,actions=None):
                    prompt=torch.tensor(ids,device=model.device);calls=[0]
                    def count(_module,_args):calls[0]+=1
                    handle=model.register_forward_pre_hook(count)
                    if actions is None:fn=external if name=='official' else optimized
                    else:
                        fn=generator(external,lambda p,t:actions.append((p.tolist(),t.tolist())),
                            statistics=None if name=='official' else probability_statistics)
                    proxy=model if name=='official' and actions is None else (
                        ModelReadout(model) if name=='official' else compact)
                    responses,iterations=[None],[0]
                    try:
                        torch.cuda.synchronize();started=time.perf_counter()
                        with suppress_official_prints():
                            fn(proxy,[prompt],[len(ids)],1,responses,iterations,gen_length=length,block_length=32,
                                threshold=.9,gamma=.8,track_num=4,mask_num=4,verify=verify,tokenizer=tokenizer,stop_tokens=[])
                        torch.cuda.synchronize();seconds=time.perf_counter()-started
                    finally:handle.remove()
                    return dict(text=responses[0],seconds=seconds,nfe=calls[0],iterations=iterations[0],
                                peak_gib=torch.cuda.max_memory_allocated()/2**30)
                warm=prompt_ids(tokenizer,generation_prompt(values[0]),task,preformatted=True)
                warm_started=time.perf_counter()
                run(warm,'official');run(warm,'optimized')
                warm_seconds=time.perf_counter()-warm_started
                rows=[]
                for index,sample in enumerate(tqdm(values,desc=cell,ascii=False)):
                    path=out/'records'/f'{index:05d}.json'
                    ident=str(sample.get('id',sample.get('task_id')))
                    if path.exists():
                        row=json.loads(path.read_text());assert row['id']==ident
                        rows.append(row);continue
                    ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                    actions_before=[];actions_after=[]
                    traced_before=run(ids,'official',actions_before)
                    traced_after=run(ids,'optimized',actions_after)
                    row=dict(index=index,id=ident,prompt_tokens=len(ids),generation_budget=length,trace=dict(
                        exact_step_actions_match=actions_before==actions_after,
                        actions_before=actions_before,actions_after=actions_after,
                        official_canvas=canvas_metadata(actions_before,len(ids),length),
                        optimized_canvas=canvas_metadata(actions_after,len(ids),length),
                        text_nfe_match=all(traced_before[k]==traced_after[k] for k in ('text','nfe','iterations')),
                        scope='These timings contain action logging and are excluded from speed results'),
                        clean={name:[] for name in ('official','optimized')})
                    for repeat in range(2):
                        for name in (('official','optimized') if repeat==0 else ('optimized','official')):
                            torch.cuda.reset_peak_memory_stats()
                            row['clean'][name].append(run(ids,name))
                    for name in ('official','optimized'):
                        outcomes=row['clean'][name]
                        row[name]=dict(outcomes[0],seconds=statistics.median(v['seconds'] for v in outcomes),
                                       output_tokens=len(tokenizer.encode(outcomes[0]['text'],add_special_tokens=False)))
                    reference={k:traced_before[k] for k in ('text','nfe','iterations')}
                    row['clean_parity']=all({k:v[k] for k in reference}==reference for v in
                                            [*row['clean']['official'],*row['clean']['optimized']])
                    row['finite_parity_pass']=row['trace']['exact_step_actions_match'] and row['trace']['text_nfe_match'] and row['clean_parity']
                    write_json(path,row);rows.append(row)
                scoring_started=time.perf_counter();scores=score_records(task,values,rows)
                write_json(out/'scores.json',scores)
                metric={}
                for name in ('official','optimized'):
                    elapsed=[r[name]['seconds'] for r in rows]
                    metric[name]=dict(examples=len(rows),accuracy=sum(scores['primary'][name])/len(rows),
                        original_accuracy=sum(scores['original'][name])/len(rows),
                        mean_seconds=statistics.mean(elapsed),p50_seconds=percentile(elapsed,.5),p95_seconds=percentile(elapsed,.95),
                        mean_nfe=statistics.mean(r[name]['nfe'] for r in rows),
                        mean_output_tokens=statistics.mean(r[name]['output_tokens'] for r in rows),
                        eos_rate=statistics.mean(r['trace'][name+'_canvas']['budget_has_eos'] for r in rows),
                        budget_without_eos_rate=statistics.mean(r['trace'][name+'_canvas']['budget_without_eos'] for r in rows))
                comparison=paired_intervals(scores['primary']['official'],scores['primary']['optimized'],
                    [r['official']['seconds'] for r in rows],[r['optimized']['seconds'] for r in rows])
                result=dict(task=task,length=length,decoder=decoder,metrics=metric,paired=comparison,
                    finite_parity_pass=sum(r['finite_parity_pass'] for r in rows),examples=len(rows),
                    ids=[r['id'] for r in rows],primary_policy=scores['primary_policy'],
                    warmup_seconds=warm_seconds,scoring_seconds=time.perf_counter()-scoring_started,
                    scope=identity['scope'])
                write_json(out/'summary.json',result);(out/'complete').write_text('OK\n')
                all_summaries[cell]=result;write_json(args.output/'summary.json',all_summaries)
                print(f'Completed {cell}: '+json.dumps(dict(metrics=metric,paired=comparison,
                    finite_parity_pass=result['finite_parity_pass'])),flush=True)
    assert source=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},'Research source changed'
    assert scorer_source=={p.name:sha256(p) for p in scorer.glob('score*.py')},'Scoring source changed'
    assert identity['official_generate_sha256']==sha256(original),'Official generator changed'
    (args.output/'complete').write_text('OK\n')


if __name__=='__main__':main()
