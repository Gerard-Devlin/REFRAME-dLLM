import argparse
from pathlib import Path
import torch
from .core import configuration,read,write,OPTIONS,digest
from .runtime import setup,barrier,gate,gather

def main():
    p=argparse.ArgumentParser(description='Pinned-v2 fixed-trajectory distillation; gated stages')
    p.add_argument('stage',choices=('audit','collect','smoke','train','export','evaluate'))
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit',type=Path)
    p.add_argument('--trajectories',type=Path)
    p.add_argument('--smoke',type=Path)
    p.add_argument('--branch',choices=('basic','release'),default='basic')
    p.add_argument('--resume',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--basic-model',type=Path)
    p.add_argument('--release-model',type=Path)
    p.add_argument('--split',choices=('dev','holdout'),default='dev')
    p.add_argument('--dev-report',type=Path)
    p.add_argument('--time-limit',type=float,default=0,help='Explicit wall seconds; 0 means unlimited')
    args=p.parse_args()
    required={'collect':['audit'],'smoke':['audit','trajectories'],'train':['audit','trajectories','smoke'],
              'export':['checkpoint','trajectories'],
              'evaluate':['basic_model','release_model','trajectories']}
    for name in required.get(args.stage,[]):
        if getattr(args,name) is None: p.error(f'{args.stage} requires --{name.replace("_","-")}')
    rank,world=setup()
    if world>32: p.error('At most 32 independent audit prompts / devices')
    if args.stage=='export' and world!=1: p.error('Export uses one GPU')
    if rank==0:
        if args.output.exists(): raise ValueError('Output already exists; use a new output, resume via --resume')
        args.output.mkdir(parents=True)
    barrier()
    from transformers import AutoTokenizer
    from .model import snapshot
    tokenizer=AutoTokenizer.from_pretrained(snapshot(),trust_remote_code=True,local_files_only=True)
    config=configuration(args.data,args.dataset)
    if args.stage in ('smoke','train'): gate(args.audit,config)
    from .data import run_audit,run_collect,records
    from .training import run_smoke,run_train
    if args.stage=='audit': run_audit(args,config,tokenizer,rank,world)
    elif args.stage=='collect': run_collect(args,config,tokenizer,rank,world)
    elif args.stage=='smoke': run_smoke(args,config,tokenizer,rank,world)
    elif args.stage=='train': run_train(args,config,tokenizer,rank,world)
    elif args.stage=='export':
        from .evaluation import export_checkpoint
        # Export owns its own fresh subdirectory.
        export_checkpoint(args.checkpoint,args.output/'model',tokenizer,records(args.trajectories,'validation')[0])
    else:
        from .evaluation import evaluate_models,fixed_forward_seconds
        from .model import load
        if args.split=='holdout':
            if not args.dev_report: p.error('Holdout requires a passed development report')
            dev=read(args.dev_report)
            if dev.get('config')!=config: raise ValueError('Different development configuration')
            if not any(x['screening_pass'] for x in dev['comparisons'].values()):
                raise ValueError('No candidate passed development screening')
        models={'original':load(),'basic':load(args.basic_model),'release':load(args.release_model)}
        base_cfg=getattr(models['basic'].config,'step_distill',None)
        release_cfg=getattr(models['release'].config,'step_distill',None)
        if not base_cfg or not release_cfg:
            raise ValueError('Exports lack distillation provenance')
        if ({k:v for k,v in base_cfg.items() if k!='branch'} !=
                {k:v for k,v in release_cfg.items() if k!='branch'}):
            raise ValueError('Branches do not have identical data/world/training configuration')
        report=evaluate_models(models,tokenizer,args.dataset,rank,world,(.85,.90,.95),args.split)
        probe=records(args.trajectories,'validation')[0]
        fixed={name:fixed_forward_seconds(model,probe) for name,model in models.items()}
        all_fixed=gather([fixed])
        report['fixed_forward_seconds']={name:sum(r[name] for r in all_fixed)/len(all_fixed)
                                         for name in models}
        report['config']=config
        report['model_paths']={k:str(v) for k,v in [('basic',args.basic_model),('release',args.release_model)]}
        if rank==0: write(args.output/'summary.json',report)
    barrier()

if __name__=='__main__': main()
