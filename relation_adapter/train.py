"""One resident DDP job for three frozen-v2 coordinate adapters."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from relation_block.common import REVISION, digest, manifest, snapshot, write_json
from relation_block.full_state import epoch_batches
from relation_block.preflight import require_gate
from relation_block.train import batch, loss
from relation_block.model import Model
from relation_block.continuation import milestones
from .codec import specifications, make_codec, coverage
from .model import FrozenChart, bf16_base_copy, evaluation_copy
from .evaluate import evaluate, base_parity


ARMS = ('token', 'random', 'relation')
POINTS = (1000000, 2000000, 5000000, 10000000)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def save_checkpoint(directory, chart, optimizer, info, rank):
    directory = Path(directory)
    if rank == 0:
        if directory.exists():
            raise ValueError(f'Checkpoint already exists: {directory}')
        staging = directory.with_name(directory.name + '.incomplete')
        if staging.exists():
            raise ValueError(f'Incomplete checkpoint needs inspection: {staging}')
        staging.mkdir(parents=True)
        torch.save(chart.trainable_state(), staging/'adapter.pt')
        torch.save(optimizer.state_dict(), staging/'optimizer.pt')
        write_json(staging/'metadata.json', info)
        staging.rename(directory)
    barrier()


def restore_checkpoint(directory, chart, optimizer, expected):
    info = json.loads((directory/'metadata.json').read_text())
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f'Resume metadata differs: {key}')
    chart.load_trainable_state(torch.load(directory/'adapter.pt',map_location='cpu',weights_only=True))
    optimizer.load_state_dict(torch.load(directory/'optimizer.pt',map_location='cpu',weights_only=True))
    return info


def lr_scale(step, steps):
    warmup = max(1, math.ceil(.03 * steps))
    if step < warmup:
        return (step+1)/warmup
    return .5*(1+math.cos(math.pi*(step-warmup+1)/max(1,steps-warmup)))


def log_eval(writer, arm, step, result):
    if writer is None:
        return
    for rounds, values in result['results'].items():
        for name in ('accuracy','truncation_rate','mean_seconds','tokens_per_second'):
            writer.add_scalar(f'{arm}/{name}_{rounds}r',values[name],step)
    writer.flush()


def smoke(args, base, base_bf16, specs, rows, config, rank, world, local, device):
    """Exercise each arm's actual DDP/optimizer and checkpoint path on selected GPUs."""
    from transformers import AutoTokenizer
    from relation_block.evaluate import generate, prompt_ids
    from relation_block.model import clean_mask
    tok=AutoTokenizer.from_pretrained(snapshot(),local_files_only=True)
    indices=list(range(world))
    report={}
    for arm in ARMS:
        codec=make_codec(specs[arm],arm,device)
        torch.manual_seed(args.seed)
        chart=FrozenChart(base,args.rank)
        if arm=='token':
            base_parity(base_bf16,evaluation_copy(base_bf16,chart),tok,config['block_size'])
        trainable=[p for p in chart.parameters() if p.requires_grad]
        wrapped=DDP(chart,device_ids=[local],broadcast_buffers=False)
        optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=.01)
        i=indices[rank]
        raw,response,prefix=batch(rows,[i],config['length'],config['pad_id'],device)
        noise=torch.full((1,config['length']),.4,device=device)
        probabilities=torch.full((1,config['length']//config['block_size']),.5,device=device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            value=loss(wrapped,raw,response,prefix,codec,noise,probabilities,
                       config['mask_id'],config['block_size'])
        value.backward()
        norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
        optimizer.step()
        checkpoint=args.output/arm/'smoke_checkpoint'
        save_checkpoint(checkpoint,chart,optimizer,dict(arm=arm,step=1),rank)
        restore_checkpoint(checkpoint,chart,optimizer,dict(arm=arm,step=1))
        probe=evaluation_copy(base_bf16,chart)
        generated,calls=generate(probe,prompt_ids(tok,'What is 1 plus 1?'),codec,
                                 config['mask_id'],config['eos_id'],4,32)
        torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated()/2**30
        peaks=[None]*world
        dist.all_gather_object(peaks,peak)
        if rank==0:
            report[arm]=dict(loss=float(value),grad_norm=float(norm),
                             trainable_parameters=sum(p.numel() for p in trainable),
                             per_gpu_peak_gib=peaks,generated_tokens=len(generated),calls=calls)
        barrier()
        del probe,wrapped,optimizer,chart,codec
    if rank==0:
        write_json(args.output/'smoke.json',dict(pass_=True,world_size=world,arms=report))
    barrier()


def main():
    import datetime
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--global-batch',type=int,default=0,help='0 = twice the number of GPUs')
    p.add_argument('--rank',type=int,default=64,help='Matched input/output adapter bottleneck')
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--eval-limit',type=int,default=32,help='Development examples at intermediate checkpoints')
    p.add_argument('--final-eval-limit',type=int,default=256)
    p.add_argument('--heldout-limit',type=int,default=32)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true',help='One optimizer step and checkpoint per arm, then stop')
    args=p.parse_args()
    world, rank, local = (int(os.getenv(k,d)) for k,d in (('WORLD_SIZE',1),('RANK',0),('LOCAL_RANK',0)))
    args.global_batch = args.global_batch or 2*world
    if world<2 or args.global_batch%world or args.rank<1 or args.lr<=0 or min(args.eval_limit,args.final_eval_limit,args.heldout_limit)<1:
        raise ValueError('Use at least two GPUs and a divisible positive batch')
    device=torch.device('cuda',local)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl',device_id=device,timeout=datetime.timedelta(hours=6))
    require_gate(args.data)
    config=manifest(args.data)
    data_hash=digest(args.data/'manifest.json')
    driver_hash=digest(Path(__file__))
    if rank==0:
        if not args.resume and any((args.output/name).exists()
                                   for name in ('plan.json','smoke.json','original','token','random','relation')):
            raise ValueError('Experiment results exist; use a new directory or explicit --resume')
        args.output.mkdir(parents=True,exist_ok=True)
    barrier()
    rows=json.loads((args.data/'train.json').read_text())
    heldout=json.loads((args.data/'heldout.json').read_text())
    batches=epoch_batches(len(rows),args.global_batch,args.seed)
    points=milestones(rows,batches,config['length'],POINTS)
    stop=points[-1]['step']
    if stop>len(batches):
        raise ValueError('10M tokens exceed one epoch')
    point_by_step={x['step']:x for x in points}
    specs=specifications(args.data,args.seed)
    coverages={arm:coverage(make_codec(specs[arm],arm,'cpu'),heldout) for arm in ARMS}
    gap=abs(coverages['random']-coverages['relation'])
    if gap>.02:
        raise ValueError(f'Random and relation intervention rates differ by {gap:.4f}; reject unmatched control')
    base=Model.load(snapshot(),device,dtype=torch.float32)
    base.requires_grad_(False)
    base.eval()
    base_bf16=bf16_base_copy(base)
    if args.smoke:
        smoke(args,base,base_bf16,specs,rows,config,rank,world,local,device)
        dist.destroy_process_group()
        return
    writer=None
    if rank==0:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(args.output/'tensorboard'),flush_secs=10)
    if rank==0:
        write_json(args.output/'plan.json',dict(arms=ARMS,points=points,adapter_rank=args.rank,
            learning_rate=args.lr,global_batch=args.global_batch,world_size=world,seed=args.seed,
            data_hash=data_hash,driver_hash=driver_hash,model_revision=REVISION,
            heldout_change_fraction=coverages,random_relation_coverage_gap=gap,
            intermediate_eval_limit=args.eval_limit,final_eval_limit=args.final_eval_limit,
            train_original_tokens=POINTS[-1],learning_rate_schedule='3% warmup then cosine over 10M-token steps',
            note='Frozen base, tied original embedding/head, identical trainable chart size and sampler in all arms.'))
    original = args.output/'original/eval/summary.json'
    if not original.exists():
        identity = make_codec(specs['token'],'token',device)
        result=evaluate(base_bf16,None,identity,args.data,config,original.parent,rank,world,
                        args.final_eval_limit,(4,8,16),512,args.heldout_limit)
        log_eval(writer,'original',0,result)
        del identity
    for arm in ARMS:
        spec=specs[arm]
        codec=make_codec(spec,arm,device)
        cov=coverages[arm]
        if rank==0:
            print('CODEC',arm,'heldout_changed_fraction',cov,flush=True)
            write_json(args.output/arm/'codec.json',spec)
        barrier()
        torch.manual_seed(args.seed)
        chart=FrozenChart(base,args.rank)
        trainable=[p for p in chart.parameters() if p.requires_grad]
        total_trainable=sum(p.numel() for p in trainable)
        if rank==0:
            print('ARM',arm,'base_frozen',all(not p.requires_grad for p in base.parameters()),
                  'trainable',total_trainable,flush=True)
        wrapped=DDP(chart,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
        optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=.01,foreach=False)
        arm_root=args.output/arm
        expected=dict(arm=arm,world_size=world,global_batch=args.global_batch,adapter_rank=args.rank,
                      lr=args.lr,seed=args.seed,data_hash=data_hash,driver_hash=driver_hash,
                      codec=spec,model_revision=REVISION)
        checkpoints=sorted(p for p in (arm_root/'checkpoints').glob('step_*')
                           if p.is_dir() and not p.name.endswith('.incomplete')
                           and all((p/name).is_file() for name in ('adapter.pt','optimizer.pt','metadata.json'))) if args.resume else []
        step0=seen=supervised=0
        if checkpoints:
            info=restore_checkpoint(checkpoints[-1],chart,optimizer,expected)
            step0,seen,supervised=info['step'],info['original_tokens'],info['supervised_tokens']
            if seen != sum(sum(min(len(rows[i]['ids']),config['length']) for i in b) for b in batches[:step0]):
                raise ValueError('Resume token cursor differs')
        if step0>stop:
            raise ValueError('Checkpoint exceeds campaign budget')
        # Original token-space chart must reproduce the frozen model exactly.
        if arm=='token' and step0==0:
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(snapshot(),local_files_only=True)
            base_parity(base_bf16, evaluation_copy(base_bf16,chart),tok,config['block_size'])
        if step0 in point_by_step and not (arm_root/'eval'/f'step_{step0:08d}'/'summary.json').exists():
            dest=arm_root/'eval'/f'step_{step0:08d}'
            result=evaluate(base_bf16,chart,codec,args.data,config,dest,rank,world,
                            args.final_eval_limit if step0==stop else args.eval_limit,
                            (4,8,16),512,args.heldout_limit)
            log_eval(writer,arm,seen,result)
        if step0==0:
            initial=arm_root/'eval/step_00000000/summary.json'
            if not initial.exists():
                result=evaluate(base_bf16,chart,codec,args.data,config,initial.parent,rank,world,
                                args.eval_limit,(4,8,16),512,args.heldout_limit)
                log_eval(writer,arm,0,result)
        for step in range(step0,stop):
            indices=batches[step]
            den=sum(min(len(rows[i]['ids']),config['length'])-rows[i]['prefix'] for i in indices)
            if den<=0:
                raise ValueError('No supervised response tokens')
            original_tokens=sum(min(len(rows[i]['ids']),config['length']) for i in indices)
            seen+=original_tokens
            supervised+=den
            slots=math.ceil(len(indices)/world)*world
            rng=torch.Generator().manual_seed(args.seed+1000003*step)
            noise=torch.rand(slots,config['length'],generator=rng)
            probabilities=.001+.999*torch.rand(slots,config['length']//config['block_size'],generator=rng)
            optimizer.param_groups[0]['lr']=args.lr*lr_scale(step,stop)
            optimizer.zero_grad(set_to_none=True)
            losses=torch.zeros((),device=device)
            tick=time.perf_counter()
            for j in range(slots//world):
                slot=j*world+rank
                active=slot<len(indices)
                i=indices[slot] if active else indices[0]
                raw,response,prefix=batch(rows,[i],config['length'],config['pad_id'],device)
                count=int(response.sum()) if active else 0
                context=wrapped.no_sync() if j<slots//world-1 else nullcontext()
                with context:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        value=loss(wrapped,raw,response,prefix,codec,noise[slot:slot+1].to(device),
                                   probabilities[slot:slot+1].to(device),config['mask_id'],config['block_size'])
                        weighted=value*(count*world/den)
                    if not torch.isfinite(weighted):
                        raise FloatingPointError('Non-finite adapter loss')
                    weighted.backward()
                losses+=value.detach()*count
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            optimizer.step()
            dist.all_reduce(losses)
            torch.cuda.synchronize()
            completed=step+1
            if rank==0:
                rec=dict(arm=arm,step=completed,steps=stop,original_tokens=seen,
                         supervised_tokens=supervised,loss=float(losses/den),
                         lr=optimizer.param_groups[0]['lr'],grad_norm=float(norm),
                         step_seconds=time.perf_counter()-tick)
                with (arm_root/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(rec)+'\n')
                for name in ('loss','grad_norm','lr','step_seconds'):
                    writer.add_scalar(f'{arm}/train_{name}',rec[name],seen)
                if completed%10==0 or completed in point_by_step:
                    print('TRAIN',json.dumps(rec),flush=True)
            if completed%100==0 or completed in point_by_step:
                info=dict(expected,step=completed,original_tokens=seen,supervised_tokens=supervised,
                          trainable_parameters=total_trainable,heldout_change_fraction=cov,
                          optimizer_step=completed)
                save_checkpoint(arm_root/'checkpoints'/f'step_{completed:08d}',chart,optimizer,info,rank)
            if completed in point_by_step:
                dest=arm_root/'eval'/f'step_{completed:08d}'
                if not (dest/'summary.json').exists():
                    result=evaluate(base_bf16,chart,codec,args.data,config,dest,rank,world,
                                    args.final_eval_limit if completed==stop else args.eval_limit,
                                    (4,8,16),512,args.heldout_limit)
                    log_eval(writer,arm,seen,result)
        if step0==stop:
            dest=arm_root/'eval'/f'step_{stop:08d}'
            if not (dest/'summary.json').exists():
                result=evaluate(base_bf16,chart,codec,args.data,config,dest,rank,world,
                                args.final_eval_limit,(4,8,16),512,args.heldout_limit)
                log_eval(writer,arm,seen,result)
        if rank==0:
            write_json(arm_root/'complete.json',dict(expected,step=stop,original_tokens=seen,
                supervised_tokens=supervised,trainable_parameters=total_trainable,
                heldout_change_fraction=cov))
        barrier()
        del wrapped, optimizer, chart, codec
    if rank==0:
        from .report import report
        report(args.output)
        writer.close()
    barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
