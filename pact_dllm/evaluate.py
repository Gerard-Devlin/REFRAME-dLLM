"""Frozen-source PACT evaluation; mechanism smoke is distinct from quality128."""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import torch
from .decode import Config, VARIANTS, generate


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)


def implementation():
    repo=Path(__file__).resolve().parent.parent
    paths=list((repo/'pact_dllm').rglob('*.py'))+list((repo/'focus_dllm/tuning').rglob('*.py'))
    return {p.relative_to(repo).as_posix():digest(p) for p in paths}


def main():
    from focus_dllm.tuning.gpu_contract import check_binding
    from focus_dllm.tuning.competitors import load_external,load_model,generation_prompt,select_samples
    from focus_dllm.llada_common import prompt_ids,MODEL_ID,REVISION
    from focus_dllm.llada_evaluate import postprocess_output
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--third-party',type=Path,required=True)
    parser.add_argument('--datasets',type=Path,nargs=3,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scope',choices=('mechanism_smoke','development128','holdout128'),required=True)
    parser.add_argument('--limit',type=int,default=128)
    parser.add_argument('--length',type=int,default=256)
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.scope!='mechanism_smoke' and args.limit<128:parser.error('Quality selection requires128 or more per task')
    if args.scope=='holdout128' and args.offset<128:parser.error('Independent validation must exclude development128')
    if args.scope=='mechanism_smoke' and (args.limit>2 or args.length!=64):parser.error('Smoke is two already-used prompts/task at64tokens')
    config=Config(length=args.length);config.validate()
    args.output.mkdir(parents=True,exist_ok=args.resume)
    binding=check_binding(required=True);torch.set_num_threads(1)
    cls,official,adaptations=load_external(args.third_party,'flash_verify')
    module=importlib.import_module('generate')
    frozen=dict(sources=implementation(),datasets=[digest(p) for p in args.datasets],
        config=asdict(config),scope=args.scope,limit=args.limit,offset=args.offset,variants=args.variants,
        model=MODEL_ID,revision=REVISION,official_projection_sha256=digest(args.third_party/'flash_dllm/llada/flash_cache_triton.py'),
        seed=51713,stop_policy='Fixed generation work; EOS does not skip later blocks')
    manifest=args.output/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['frozen']!=frozen:raise ValueError('Resume source/config/data hash mismatch')
    else:write(manifest,dict(frozen=frozen,binding=binding,adaptations=adaptations))
    model,tokenizer=load_model(SimpleNamespace(method='flash_verify'),cls)
    forbidden=set(tokenizer.all_special_ids)
    report=dict(scope=args.scope,goal_achieved=False,quality_claim=False,bindings=binding,tasks={},private_controls=[],torch_sdpa_calls=0)
    # Independent dense reference only in controls; production path must remain Triton.
    original=torch.nn.functional.scaled_dot_product_attention
    def sdpa(*a,**kw):
        report['torch_sdpa_calls']+=1;return original(*a,**kw)
    torch.nn.functional.scaled_dot_product_attention=sdpa
    try:
        from .gpu_controls import kernel_controls
        report['kernel_controls']=kernel_controls(model.device)
        for task,path in zip(('humaneval','mbpp','math'),args.datasets):
            samples=select_samples(path,args.limit,args.offset);records=[]
            for index,sample in enumerate(samples):
                ident=str(sample.get('id',sample.get('task_id')))
                target=args.output/'records'/task/f'{index:04d}.json'
                if target.exists():
                    record=json.loads(target.read_text());assert record['id']==ident
                else:
                    ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                    record=dict(id=ident,prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
                    for variant in args.variants:
                        audit=None
                        if args.scope=='mechanism_smoke' and index==0 and variant=='joint':
                            from .gpu_controls import control
                            def audit(runtime,ready,dag):
                                if any(r['task']==task for r in report['private_controls']):return
                                row=control(runtime,ready,dag,tokenizer);row.update(task=task,id=ident)
                                report['private_controls'].append(row)
                        value=generate(model,ids,module.get_rotary_embedding,config,variant=variant,forbidden=forbidden,trace=True)
                        if audit is not None:
                            instrumented=generate(model,ids,module.get_rotary_embedding,config,variant=variant,forbidden=forbidden,trace=True,audit=audit)
                            assert instrumented['token_ids']==value['token_ids'] and instrumented['nfe']==value['nfe']
                            assert instrumented['actions']==value['actions'],'Read-only controls altered online commits'
                            value['audit_seconds_separate']=instrumented['seconds']
                        value['text'],value['output_tokens']=postprocess_output(tokenizer,value['token_ids'],sample,task)
                        value['first_eos']=value['token_ids'].index(126081) if 126081 in value['token_ids'] else None
                        value['truncated']=value['first_eos'] is None
                        record[variant]=value
                    write(target,record)
                records.append(record)
                fraction=(index+1)/len(samples);bar='█'*int(24*fraction)+'░'*(24-int(24*fraction))
                print(f'{task} [{bar}] {index+1}/{len(samples)}',flush=True)
                write(args.output/'progress.json',dict(task=task,completed=index+1,total=len(samples)))
            summary={}
            for variant in args.variants:
                rows=[r[variant] for r in records]
                summary[variant]=dict(mean_seconds=statistics.mean(r['seconds'] for r in rows),
                    mean_nfe=statistics.mean(r['nfe'] for r in rows),
                    total_verified_commits=sum(r['stats']['verified_commits'] for r in rows),
                    total_selected=sum(r['stats']['draft_selected'] for r in rows),
                    mean_planner_seconds=statistics.mean(r['stats']['planner_seconds'] for r in rows),
                    mean_signal_seconds=statistics.mean(r['stats']['signal_seconds'] for r in rows),
                    truncation_rate=statistics.mean(r['truncated'] for r in rows))
            if args.scope!='mechanism_smoke':
                from focus_dllm.tuning.focus_v4_evaluate import score
                scoring=score(task,samples,records,args.variants);write(args.output/f'{task}_scores.json',scoring)
                for variant in args.variants:
                    correct=scoring['correct'][variant];known=[c for c in correct if c is not None]
                    summary[variant].update(accuracy=sum(known)/len(known) if known else None,
                                          scoring_unknown=len(correct)-len(known),quality_scope=args.scope)
            else:
                for row in summary.values():row['accuracy']=None
            report['tasks'][task]=summary
            print('TASK SUMMARY '+json.dumps(dict(task=task,scope=args.scope,methods=summary),ensure_ascii=False),flush=True)
            write(args.output/'summary.json',report)
        assert implementation()==frozen['sources'],'Sources changed during research'
        assert report['torch_sdpa_calls']==0
        write(args.output/'summary.json',report);(args.output/'complete').write_text('0\n')
    finally:torch.nn.functional.scaled_dot_product_attention=original


if __name__=='__main__':main()
