"""Paired quality curves for token, random, and relation frozen charts."""
import json
from pathlib import Path
from relation_block.common import write_json


def report(root):
    root=Path(root)
    original=json.loads((root/'original/eval/summary.json').read_text())
    rows={}
    reference_ids={}
    reference_steps=None
    for arm in ('token','random','relation'):
        info=json.loads((root/arm/'complete.json').read_text())
        steps=sorted((root/arm/'eval').glob('step_*/summary.json'))
        names=[p.parent.name for p in steps]
        if len(steps)!=5 or (reference_steps is not None and names!=reference_steps):
            raise ValueError('Incomplete or unmatched checkpoint schedule')
        reference_steps=names
        for key in ('global_batch','world_size','adapter_rank','lr','seed','data_hash','driver_hash',
                    'model_revision','original_tokens','supervised_tokens'):
            if arm=='token':
                continue
            if info[key] != rows['token']['training'][key]:
                raise ValueError(f'Control differs: {key}')
        observations={}
        for p in steps:
            data=json.loads(p.read_text())
            reference_ids.setdefault(p.parent.name,data['ids'])
            if data['ids']!=reference_ids[p.parent.name]:
                raise ValueError('Evaluation question IDs differ')
            comparisons={}
            if arm!='token':
                counterpart=root/'token/eval'/p.parent.name
                for rounds in ('4','8','16'):
                    ours=[json.loads(s) for s in (p.parent/f'samples_{rounds}.jsonl').read_text().splitlines()]
                    token=[json.loads(s) for s in (counterpart/f'samples_{rounds}.jsonl').read_text().splitlines()]
                    if [x['id'] for x in ours]!=[x['id'] for x in token]:
                        raise ValueError('Paired sample ordering differs')
                    comparisons[rounds]=dict(ours_only_correct=sum(a['correct'] and not b['correct'] for a,b in zip(ours,token)),
                                              token_only_correct=sum(b['correct'] and not a['correct'] for a,b in zip(ours,token)))
            observations[p.parent.name]=dict(evaluation=data,paired_vs_token=comparisons)
        rows[arm]=dict(training=info,observations=observations)
    # This diagnostic makes the random control's actual intervention size visible.
    delta=abs(rows['random']['training']['heldout_change_fraction']-
              rows['relation']['training']['heldout_change_fraction'])
    final_step=reference_steps[-1]
    if original['ids'] != reference_ids[final_step]:
        raise ValueError('Original baseline and final evaluation question IDs differ')
    summary=dict(complete=True,original_baseline=original,arms=rows,random_relation_coverage_gap=delta,
        note='Fitted relation vs matched-source random code; same adapter architecture and data. A gain here does not prove a causal dependency mechanism or end-to-end speedup.')
    write_json(root/'summary.json',summary)
    for arm,row in rows.items():
        print('RESULT',arm,'coverage',row['training']['heldout_change_fraction'],flush=True)
        for step,obs in row['observations'].items():
            print(' ',step,{k:round(v['accuracy'],4) for k,v in obs['evaluation']['results'].items()},flush=True)
    return summary
