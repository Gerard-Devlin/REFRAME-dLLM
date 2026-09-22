"""Explicit server-side checkpoint download; inference remains offline."""
import argparse
from .backend import snapshot_path,MODELS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sizes',default='1.5b,7b')
    args=p.parse_args()
    for size in args.sizes.lower().split(','):
        if size not in MODELS:
            raise ValueError(size)
        print(size,snapshot_path(size,download=True),flush=True)


if __name__=='__main__':
    main()
