"""One independent request shard per selected GPU; both sizes sequentially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',required=True,help='Physical nvidia-smi indices, e.g. 1,2,3,4')
    p.add_argument('--sizes',default='1.5b,7b')
    p.add_argument('--mode',choices=['probe','generate'],default='probe')
    p.add_argument('--limit',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-free-gib',type=float,default=22)
    args,extra=p.parse_known_args()
    ids=args.gpus.split(',')
    sizes=args.sizes.lower().split(',')
    if not ids or any(not s.isdigit() for s in ids) or len(set(ids))!=len(ids):
        raise ValueError('GPU IDs must be distinct numeric nvidia-smi indices')
    if args.limit<len(ids):
        raise ValueError('LIMIT must be at least the number of GPUs (one prompt shard per GPU)')
    if len(set(sizes))!=len(sizes) or any(s not in ('1.5b','7b') for s in sizes):
        raise ValueError('Sizes must be 1.5b and/or 7b')
    uuids=[]
    for gpu in ids:
        text=subprocess.check_output(['nvidia-smi','-i',gpu,
            '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip()
        uuid,free=[s.strip() for s in text.split(',')]
        if float(free)/1024<args.min_free_gib:
            raise ValueError(f'GPU {gpu} only has {float(free)/1024:.1f} GiB free')
        uuids.append(uuid)
    args.output.mkdir(parents=True,exist_ok=True)
    collected={}
    for size in sizes:
        workers=[]
        model_dir=args.output/size
        if model_dir.exists():
            raise ValueError('Model run directory exists; use a fresh RUN_DIR')
        model_dir.mkdir()
        try:
            for rank,uuid in enumerate(uuids):
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS=os.getenv('OMP_NUM_THREADS','4'))
                log=(model_dir/f'worker_{rank:02d}.log').open('w',encoding='utf-8')
                cmd=[sys.executable,'-u','-m','denoising_dag.benchmark','--size',size,
                     '--mode',args.mode,'--output',str(model_dir/f'worker_{rank:02d}'),
                     '--rank',str(rank),'--world-size',str(len(ids)),'--limit',str(args.limit),*extra]
                process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                workers.append((process,log))
                print(f'{size}: physical GPU {ids[rank]}, log {log.name}',flush=True)
            last_update=time.monotonic()
            while any(p.poll() is None for p,_ in workers):
                failed=[p for p,_ in workers if p.poll() not in (None,0)]
                if failed:
                    raise RuntimeError(f'{size} worker failed; inspect worker logs')
                if time.monotonic()-last_update>=30:
                    for process,log in workers:
                        with open(log.name,'rb') as stream:
                            stream.seek(0,2)
                            stream.seek(max(0,stream.tell()-2048))
                            lines=stream.read().decode('utf-8',errors='replace').splitlines()
                        print(f'{size} pid={process.pid} status={process.poll()} latest: {lines[-1] if lines else "loading"}',flush=True)
                    last_update=time.monotonic()
                time.sleep(1)
            if any(p.returncode for p,_ in workers):
                raise RuntimeError(f'{size} worker failed; inspect worker logs')
        finally:
            for process,log in workers:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                log.close()
        from .benchmark import summarize
        from .backend import write_json
        records=[]
        metadata=[]
        for rank in range(len(ids)):
            root=model_dir/f'worker_{rank:02d}'
            records.extend(json.loads(s) for s in (root/'records.jsonl').read_text(encoding='utf-8').splitlines())
            metadata.append(json.loads((root/'summary.json').read_text(encoding='utf-8')))
        seen=[i for m in metadata for i in m['prompt_ids']]
        if len(set(seen))!=len(seen):
            raise AssertionError('Duplicated prompt assignments')
        if any(m['source_hashes']!=metadata[0]['source_hashes'] for m in metadata):
            raise AssertionError('Workers used different implementations')
        summary=dict(size=size,mode=args.mode,gpu_ids=ids,prompt_ids=seen,
                     groups=summarize(records,args.mode),workers=metadata,
                     timing_scope='Aggregate of per-request paired latencies on independent GPUs, not multi-GPU single-request speedup')
        write_json(model_dir/'summary.json',summary)
        collected[size]=summary
        print(size,json.dumps(summary['groups'],indent=2),flush=True)
    from .backend import write_json
    write_json(args.output/'summary.json',dict(mode=args.mode,models=collected))


if __name__=='__main__':
    main()
