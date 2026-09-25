import contextlib
import math
import os
import shutil
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .core import (write,read,digest,batches,rank_microbatches,loss_counts,lr_factor)
from .model import load,add_lora,enable_checkpointing,adapters,load_adapters,Distiller
from .runtime import gather,barrier,save_rng,restore_rng
from .data import records
from .evaluation import evaluate_models,mechanism,merged_copy
from .trajectory import synchronize

def save(path,student,optimizer,step,config,rank,world,best):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    state=dict(adapter=adapters(student),optimizer=optimizer.state_dict(),next_step=step,
        scheduler=dict(next_step=step),rng=save_rng(),world=world,config=config,best=best)
    tmp=path/f'rank{rank}.tmp'; torch.save(state,tmp); tmp.replace(path/f'rank{rank}.pt')
    barrier()
    if rank==0: write(path/'complete.json',dict(world=world,next_step=step,config_hash=digest(config)))
    barrier()

def restore(path,student,optimizer,config,rank,world):
    manifest=read(Path(path)/'complete.json')
    if manifest['world']!=world or manifest['config_hash']!=digest(config):
        raise ValueError('Resume requires identical world size, branch, code and data')
    state=torch.load(Path(path)/f'rank{rank}.pt',map_location='cpu',weights_only=False)
    load_adapters(student,state['adapter']); optimizer.load_state_dict(state['optimizer']); restore_rng(state['rng'])
    return state['next_step'],state['best']

def prune(root):
    root=Path(root).resolve()
    dirs=sorted(root.glob('step_*'))
    for p in dirs[:-2]:
        if p.is_symlink() or p.resolve().parent!=root: raise ValueError('Unsafe checkpoint path')
        shutil.rmtree(p)

def setup_models(branch,rank,world):
    teacher=load(); student=load(); names=add_lora(student); enable_checkpointing(student)
    params=[p for p in student.parameters() if p.requires_grad]
    if any(p.dtype!=torch.float32 for p in params): raise AssertionError('Adapters must be FP32')
    if rank==0: print(dict(total_parameters=sum(p.numel() for p in student.parameters()),
                          trainable_parameters=sum(p.numel() for p in params),lora_modules=len(names)),flush=True)
    module=Distiller(student,teacher,branch)
    runner=DDP(module,device_ids=[torch.cuda.current_device()],broadcast_buffers=False) if world>1 else module
    optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01)
    return teacher,student,runner,optimizer

def update(runner,optimizer,rows,batch,rank,world,step,total):
    optimizer.zero_grad(set_to_none=True)
    counts=loss_counts([rows[i] for i in batch])
    scale=torch.tensor([world/max(1,x) for x in counts],device='cuda')
    metrics=torch.zeros(3,device='cuda'); teacher_seconds=0.; exposure=0
    started=time.perf_counter()
    for j,index in enumerate(rank_microbatches(batch,rank,world)):
        valid=index is not None; record=rows[index] if valid else rows[batch[0]]
        no_sync=runner.no_sync() if isinstance(runner,DDP) and j==0 else contextlib.nullcontext()
        with no_sync,torch.autocast('cuda',dtype=torch.bfloat16):
            terms=runner(record)
            loss=(terms*scale).sum()*(1 if valid else 0)
        loss.backward()
        if valid:
            metrics+=terms.detach(); exposure+=sum(map(len,record['history']))+len(record['canvas'])
        module=runner.module if isinstance(runner,DDP) else runner
        teacher_seconds+=module.teacher_seconds
    if dist.is_initialized(): dist.all_reduce(metrics)
    params=[p for p in runner.parameters() if p.requires_grad]
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    for group in optimizer.param_groups: group['lr']=1e-5*lr_factor(step,total)
    optimizer.step(); synchronize()
    if not torch.isfinite(metrics).all(): raise FloatingPointError('Non-finite training metrics')
    timing=torch.tensor([time.perf_counter()-started,teacher_seconds,exposure],device='cuda',dtype=torch.float64)
    all_time=gather([timing.tolist()])
    return dict(ce=float(metrics[0])/max(1,counts[0]),kl=float(metrics[1])/max(1,counts[1]),
        hinge=float(metrics[2])/max(1,counts[2]),gradient_norm=float(norm),supervised_tokens=counts[0],
        seconds=max(t[0] for t in all_time),teacher_gpu_seconds=sum(t[1] for t in all_time),
        processed_tokens=int(sum(t[2] for t in all_time)),lr=optimizer.param_groups[0]['lr'])

def gradient_consistency(student):
    # Exact full adapter comparison after DDP all-reduce; bounded communication chunks.
    if not dist.is_initialized(): return
    for p in student.parameters():
        if not p.requires_grad: continue
        if p.grad is None: raise AssertionError('Unused trainable parameter')
        flat=p.grad.flatten()
        for i in range(0,flat.numel(),1000000):
            ref=flat[i:i+1000000].clone(); dist.broadcast(ref,0)
            if not torch.equal(ref,flat[i:i+1000000]): raise AssertionError('Ranks have different reduced gradients')

def run_smoke(args,config,tokenizer,rank,world):
    rows=records(args.trajectories,'train')[:32]
    if len(rows)!=32: raise ValueError('Smoke needs 32 real states')
    teacher,student,runner,optimizer=setup_models(args.branch,rank,world)
    before=mechanism(student,teacher,rows,rank,world)
    schedule=batches(len(rows),world); timings=[]; losses=[]
    smoke_config=dict(config,branch=args.branch,world=world,trajectories=digest(read(args.trajectories/'manifest.json')))
    for step in range(50):
        m=update(runner,optimizer,rows,schedule[step%len(schedule)],rank,world,step,50)
        if step>=2: timings.append(m['seconds'])
        losses.append(m['ce'])
        if step==1:
            gradient_consistency(student)
            save(args.output/'resume_probe',student,optimizer,2,smoke_config,rank,world,None)
        if step==2:
            expected=adapters(student)
            restore(args.output/'resume_probe',student,optimizer,smoke_config,rank,world)
            repeat=update(runner,optimizer,rows,schedule[step%len(schedule)],rank,world,step,50)
            if any(not torch.equal(expected[k],v) for k,v in adapters(student).items()):
                raise AssertionError('Save/restore update is not exact')
        if rank==0: print(f'smoke {step+1}/50 ce={m["ce"]:.5f}',flush=True)
    after=mechanism(student,teacher,rows,rank,world)
    fit=(sum(losses[-len(schedule):])/len(schedule)<sum(losses[:len(schedule)])/len(schedule)
         and after['second_release']>=before['second_release'])
    memory=gather([dict(rank=rank,peak_gib=torch.cuda.max_memory_allocated()/2**30)])
    if rank==0:
        summary=dict(stage='smoke',config=smoke_config,pass_=fit,states=32,updates=50,
            before=before,after=after,exact_resume=True,gradient_consistency=True,
            stable_seconds_per_update=sum(timings)/len(timings),memory=memory,
            note='Small-state fitting checks engineering only, not generalization.')
        summary['pass']=summary.pop('pass_'); write(args.output/'summary.json',summary)

def run_train(args,config,tokenizer,rank,world):
    manifest=read(args.trajectories/'manifest.json')
    if manifest['config']!=config: raise ValueError('Trajectory configuration mismatch')
    cfg=dict(config,branch=args.branch,world=world,trajectories=digest(manifest))
    smoke=read(args.smoke)
    if not smoke.get('pass') or smoke['config']!=cfg: raise ValueError('Passed actual-world/branch smoke required')
    rows=records(args.trajectories,'train'); validation=records(args.trajectories,'validation')
    schedule=batches(len(rows),world); total=len(schedule)
    teacher,student,runner,optimizer=setup_models(args.branch,rank,world)
    step=0; best=None
    if args.resume: step,best=restore(args.resume,student,optimizer,cfg,rank,world)
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(args.output/'tensorboard'/args.branch)) if rank==0 else None
    checkpoints=args.output/'checkpoints'; milestones={0,total,*[math.ceil(total*q) for q in (.25,.5,.75)]}
    trained_seconds=teacher_gpu_seconds=eval_seconds=0.; tokens=0; started=time.perf_counter()
    def assess(progress,best):
        nonlocal eval_seconds
        rng=save_rng(); begin=time.perf_counter()
        merged=merged_copy(adapters(student))
        report=evaluate_models({'original':teacher,'student':merged},tokenizer,args.dataset,rank,world)
        report['mechanism']=mechanism(merged,teacher,validation,rank,world)
        eval_seconds+=time.perf_counter()-begin
        score=report['metrics']['student@0.9']['accuracy']
        if rank==0: write(args.output/'evaluations'/f'step_{progress}.json',report)
        del merged; torch.cuda.empty_cache(); restore_rng(rng)
        if best is None or score>best:
            best=score; save(checkpoints/'best',student,optimizer,progress,cfg,rank,world,best)
        if writer:
            for k,v in report['mechanism'].items(): writer.add_scalar('validation/'+k,v,progress)
            writer.add_scalar('gsm8k/accuracy',score,progress)
        return best
    if step in milestones: best=assess(step,best)
    for current in range(step,total):
        m=update(runner,optimizer,rows,schedule[current],rank,world,current,total)
        trained_seconds+=m['seconds']; teacher_gpu_seconds+=m['teacher_gpu_seconds']; tokens+=m['processed_tokens']
        next_step=current+1
        if next_step in milestones: best=assess(next_step,best)
        if writer:
            for k,v in m.items(): writer.add_scalar('train/'+k,v,next_step)
            writer.add_scalar('train/eta_seconds',m['seconds']*(total-next_step),next_step)
            writer.add_scalar('train/tokens_per_second',m['processed_tokens']/m['seconds'],next_step)
            writer.flush()
        if rank==0: print(f'{args.branch} step={next_step}/{total} CE={m["ce"]:.4f} KL={m["kl"]:.4f} sec={m["seconds"]:.2f} ETA={(total-next_step)*m["seconds"]/3600:.2f}h',flush=True)
        stop=torch.tensor(int(args.time_limit>0 and time.perf_counter()-started>=args.time_limit),device='cuda')
        if dist.is_initialized(): dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if next_step%250==0 or next_step==total or bool(stop):
            target=checkpoints/('final' if next_step==total else f'step_{next_step:08d}')
            save(target,student,optimizer,next_step,cfg,rank,world,best)
            if rank==0: prune(checkpoints)
        if bool(stop): break
    memory=gather([dict(rank=rank,peak_gib=torch.cuda.max_memory_allocated()/2**30)])
    if rank==0:
        write(args.output/'summary.json',dict(config=cfg,next_step=next_step,total_steps=total,
            finished_epoch=next_step==total,train_seconds=trained_seconds,eval_seconds=eval_seconds,
            teacher_gpu_seconds=teacher_gpu_seconds,processed_tokens=tokens,memory=memory))
    if writer: writer.close()
