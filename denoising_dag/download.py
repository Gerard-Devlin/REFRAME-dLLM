"""Explicit server-side checkpoint download; inference remains offline."""
import argparse
import json
import time
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from .backend import snapshot_path,MODELS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sizes',default='1.5b,7b')
    p.add_argument('--retries',type=int,default=20)
    args=p.parse_args()
    if args.retries<1:
        raise ValueError('retries must be positive')
    for size in args.sizes.lower().split(','):
        if size not in MODELS:
            raise ValueError(size)
        for attempt in range(1,args.retries+1):
            try:
                path=snapshot_path(size,download=True)
                index=path/'model.safetensors.index.json'
                shards=(set(json.loads(index.read_text())['weight_map'].values()) if index.exists()
                        else {p.name for p in path.glob('*.safetensors')})
                if not shards or any(not (path/name).is_file() or (path/name).stat().st_size==0 for name in shards):
                    raise RuntimeError(f'Incomplete model weights in {path}')
                print(f'COMPLETE {size}: {path} ({len(shards)} weight shards)',flush=True)
                break
            except (ChunkedEncodingError,ConnectionError,Timeout) as exc:
                print(f'{size} interrupted on attempt {attempt}/{args.retries}: {exc}',flush=True)
                if attempt==args.retries:
                    raise
                time.sleep(min(15*attempt,120))


if __name__=='__main__':
    main()
