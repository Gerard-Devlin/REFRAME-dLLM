import os
import random
import torch
import torch.distributed as dist
from .core import read, write, digest, configuration

def setup():
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if world>1: dist.init_process_group('nccl')
    random.seed(1234); torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    return rank,world

def gather(rows):
    if not dist.is_initialized(): return rows
    all_rows=[None]*dist.get_world_size(); dist.all_gather_object(all_rows,rows)
    return [r for part in all_rows for r in part]

def barrier():
    if dist.is_initialized(): dist.barrier()

def gate(path,config):
    d=read(path)
    if d.get('config')!=config or not d.get('pass') or d.get('stage')!='audit':
        raise ValueError('A passed audit with identical source/data/config is required')
    return d

def save_rng():
    return dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())

def restore_rng(r):
    random.setstate(r['python']); torch.set_rng_state(r['torch']); torch.cuda.set_rng_state(r['cuda'])

def gsm_rows(dataset,split='dev'):
    rows=read(dataset)
    if len(rows)<256: raise ValueError('Need at least 256 GSM8K rows for disjoint screening and holdout')
    keys=[digest(r['question']) for r in rows]
    if len(keys)!=len(set(keys)): raise ValueError('Duplicate GSM8K questions')
    order=list(range(len(rows))); random.Random(1234).shuffle(order)
    picked=order[:128] if split=='dev' else order[128:]
    return [dict(rows[i],sample_id=keys[i]) for i in picked]
