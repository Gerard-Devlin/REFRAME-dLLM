"""Matched-budget adaptation-capacity sweep report."""
import json
from pathlib import Path
import sys
from .common import write_json
from .full_state import resolve_checkpoint


def report(root):
    names = ['full','lora_r8','lora_r32','lora_r64']
    result, reference_meta, reference_steps = {}, None, None
    for name in names:
        checkpoint = resolve_checkpoint(root/name/'train')
        meta = json.loads((checkpoint/'metadata.json').read_text())
        control = {k:meta[k] for k in ('lr','seed','global_batch','micro_batch','world_size','steps',
                   'completed_steps','original_tokens','supervised_tokens','data_hash','evaluation_data_hash','objective')}
        if reference_meta is not None and reference_meta != control:
            raise ValueError('Control settings/token budgets differ')
        reference_meta = control
        paths = sorted((root/name/'eval').glob('step_*/summary.json'))
        steps = [p.parent.name for p in paths]
        if len(paths) != 4 or (reference_steps is not None and reference_steps != steps):
            raise ValueError('Missing/unmatched observation points')
        reference_steps = steps
        result[name] = dict(training=meta,observations={})
        for p in paths:
            d = json.loads(p.read_text())
            paired = {}
            if name != 'full':
                baseline = root/'full/eval'/p.parent.name
                for rounds in ('4','8','16'):
                    left = [json.loads(s) for s in (baseline/f'samples_{rounds}.jsonl').read_text().splitlines()]
                    right = [json.loads(s) for s in (p.parent/f'samples_{rounds}.jsonl').read_text().splitlines()]
                    if [r['id'] for r in left] != [r['id'] for r in right]:
                        raise ValueError('Evaluation IDs differ')
                    paired[rounds] = dict(full_only_correct=sum(a['correct'] and not b['correct'] for a,b in zip(left,right)),
                                         lora_only_correct=sum(b['correct'] and not a['correct'] for a,b in zip(left,right)),
                                         identical_predictions=sum(a['prediction']==b['prediction'] for a,b in zip(left,right)))
            result[name]['observations'][p.parent.name] = dict(evaluation=d,paired_vs_full=paired)
    write_json(root/'summary.json',dict(complete=True,results=result,
        note='Same LR and schedule; ranks are not independently tuned. Retention without measurable learning is inconclusive.'))
    for name,r in result.items():
        for step,d in r['observations'].items():
            print(name,step,{k:v['accuracy'] for k,v in d['evaluation']['results'].items()},flush=True)


if __name__ == '__main__':
    report(Path(sys.argv[1]))
