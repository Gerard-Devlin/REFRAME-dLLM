"""Paired boundary A/B summaries; report differences without causal claims."""
import json
from pathlib import Path
import sys
from .common import write_json


def report(root):
    results = {}
    paths = [sorted((root/mode/'eval').glob('step_*/summary.json')) for mode in ('clean','masked')]
    if len(paths[0]) != 4 or [p.parent.name for p in paths[0]] != [p.parent.name for p in paths[1]]:
        raise ValueError('Expected matched zero/0.25M/0.5M/1M evaluations')
    for a,b in zip(*paths):
        left,right = json.loads(a.read_text()),json.loads(b.read_text())
        if left['ids'] != right['ids']:
            raise ValueError('A/B evaluation IDs differ')
        paired = {}
        for rounds in ('4','8','16'):
            x = [json.loads(s) for s in (a.parent/f'samples_{rounds}.jsonl').read_text().splitlines()]
            y = [json.loads(s) for s in (b.parent/f'samples_{rounds}.jsonl').read_text().splitlines()]
            if [r['id'] for r in x] != [r['id'] for r in y]:
                raise ValueError('Sample ordering differs')
            paired[rounds] = dict(A_only_correct=sum(i['correct'] and not j['correct'] for i,j in zip(x,y)),
                                 B_only_correct=sum(j['correct'] and not i['correct'] for i,j in zip(x,y)))
        results[a.parent.name] = dict(A=left,B=right,paired=paired)
    write_json(root/'summary.json',dict(complete=True,results=results,
        reference='Public 7B modeling.py revision 0661abf5f9f0ee338970d091052a26c8efa51974; boundary masking ablation only'))
    for step,r in results.items():
        print(step,{k:{s:v['accuracy'] for s,v in r[k]['results'].items()} for k in ('A','B')},flush=True)


if __name__ == '__main__':
    report(Path(sys.argv[1]))
