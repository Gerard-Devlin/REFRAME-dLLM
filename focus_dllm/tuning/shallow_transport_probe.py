"""Paid-teacher diagnostic of current-shallow update-field interpolation.

Fresh teacher source KV is an optimistic information control, not a live FOCUS
source or acceleration algorithm. Dropped labels are read only by error_metrics.
No teacher inputs, sampler, formal prefix or support choices are modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .landmark_residual import error_metrics
from .shallow_transport import ShallowTransport


def aggregate(rows):
    totals=[sum(row[name] for row in rows) for name in ('delta_energy','local_error','mean_error','elements')]
    return dict(records=len(rows),delta_energy=totals[0],local_error=totals[1],mean_error=totals[2],
        elements=totals[3],local_relative_error=totals[1]/totals[0] if totals[0] else None,
        mean_relative_error=totals[2]/totals[0] if totals[0] else None)


def summarize(records):
    flat=[v for r in records for v in r['geometry']]
    return dict(states=len(records),pruned_states=sum(bool(r['geometry']) for r in records),
        all=aggregate(flat),by_kind={kind:aggregate([v for v in flat if v['kind']==kind]) for kind in ('k','v')})


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    binding=check_binding(required=True)
    from ..common import sha256,write_json
    from ..llada_backend import LLaDAAttentionBackend
    from ..llada_common import MODEL_ID,REVISION,prompt_ids
    from ..llada_decode import _selected_positions
    from ..llada_pruning import Config
    from . import backend
    from .retention_sweep import sample_slice
    from .run import load_model
    from .static_support import StaticSupportForward
    from .competitors import generation_prompt

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--datasets',type=Path,nargs=3,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    prior={(r['task'],r['id'],r['block'],r['refine']):r
        for r in json.loads((args.reference_root/'allstate_probe.json').read_text())['records']}
    reference=json.loads((args.reference_root/'termination_probe.json').read_text())['records']
    data=dict(zip(('humaneval','mbpp','math'),args.datasets))
    result=dict(model=MODEL_ID,revision=REVISION,gpu_binding=binding,
        datasets={t:sha256(p) for t,p in data.items()},records=[],prompts=[],costs=[],
        implementation={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        configuration=dict(length=256,block=32,layer=4,support=32,regularization=.01,
            seed=51713,offset=0,samples_per_task=2,threshold=.90),
        hypothesis='Current cheap-path shallow hidden changes predict deep KV update fields through landmark ridge interpolation.',
        gate='Weighted squared error relative to no update <0.75 overall and <1 '
            'for both KV kinds in every task. This only warrants a live-source cost/action test, not online evaluation.',
        scope='Six reused dev prompts; all ordinary teacher states. Full same-state teacher computation pays '
            'for fresh source KV. No unavailable deep target labels enter geometry or prediction. Current shallow features are captured from the cheap path, not a future teacher state. '
            'Geometric error is not action preservation, task accuracy, novelty or E2E speed. '
            'First-pruned-state timings measure primitive construction/application only; '
            'not the cost of acquiring fresh source KV or integrating it in attention.')
    write_json(args.output/'diagnostic.json',result)
    model,tokenizer=load_model('cuda:0');torch.set_num_threads(1)
    original=backend.selected_forward
    config=Config(prune_after_layer=4,support_keep_ratio=0.,target_only_head=True)
    for task,path in data.items():
        for sample in sample_slice(json.loads(path.read_text()),0,2):
            ident=sample.get('id',sample.get('task_id'))
            ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
            memo={};timed=False;before=len(result['records']);captured={'calls':0}
            def capture_shallow(_module,_args,out):
                captured['hidden']=out[0].detach().clone()
                captured['calls']+=1
            handle=model.model.transformer.blocks[3].register_forward_hook(capture_shallow)

            def observe(model,x,target,**kwargs):
                nonlocal timed
                if bool(kwargs.get('use_cache')) and kwargs.get('past_key_values') is None:
                    current=original(model,x,target,**kwargs)
                    warm_hidden=captured['hidden']
                    memo.clear();f=StaticSupportForward(model,config,support_count=32)
                    f.reference=current.past_key_values;memo.update(forward=f,refine=0,warm_hidden=warm_hidden)
                    return current
                current=original(model,x,target,**dict(kwargs,use_cache=True))
                memo['refine']+=1;key=(task,ident,(256-x.shape[1])//32,memo['refine'])
                previous=prior[key]
                top=current.logits.argmax(-1)
                conf=current.logits.double().softmax(-1).gather(-1,top.unsqueeze(-1)).squeeze(-1)
                take=_selected_positions(conf,.90)
                assert conf[0].tolist()==previous['teacher_confidence']
                assert target[take].tolist()==previous['teacher_selected']
                assert top[0,take].tolist()==previous['teacher_values']
                f=memo['forward'];versions=[t._version for pair in f.reference+current.past_key_values for t in pair]
                x_version=x._version
                teacher_hidden=captured['hidden']
                capture_count=captured['calls']
                shadow=f(x,target,past_key_values=kwargs['past_key_values'])
                current_hidden=captured['hidden']
                assert captured['calls']==capture_count+1,'Cheap shallow capture is missing or duplicated'
                assert current_hidden.shape[1]==x.shape[1],'Cheap path did not compute the full shallow suffix'
                other=shadow.argmax(-1)
                other_conf=shadow.double().softmax(-1).gather(-1,other.unsqueeze(-1)).squeeze(-1)
                assert other_conf[0].tolist()==previous['shadow_confidence']
                geometry=[];construction=None;shallow_error=None;max_weight_l1=None
                if x.shape[1]>64:
                    prefix=kwargs['past_key_values'][0][0].shape[-2]
                    kept=f.kept.clone();source=kept[kept>=32]+prefix
                    removed=torch.ones(x.shape[1],device=x.device,dtype=torch.bool);removed[kept]=False
                    dropped=torch.arange(x.shape[1],device=x.device)[removed]+prefix
                    assert source.numel()==32 and dropped.numel()>0
                    torch.cuda.synchronize();started=time.perf_counter()
                    shared_plan=ShallowTransport.prepare(memo['warm_hidden'][:,prefix:],current_hidden,
                        source,dropped,prefix_length=prefix,kv_shape=f.reference[4][0].shape)
                    memo['plans']=[shared_plan if layer>=4 else None for layer in range(len(f.reference))]
                    torch.cuda.synchronize();construction=time.perf_counter()-started
                    shallow_error=float((teacher_hidden.float()-current_hidden.float()).abs().max())
                    max_weight_l1=float(shared_plan.weights.abs().sum(-1).max())
                    values=[]
                    for layer,(plan,old_pair,new_pair) in enumerate(zip(memo['plans'],f.reference,current.past_key_values)):
                        if plan is None:continue
                        for kind,old,new in zip(('k','v'),old_pair,new_pair):
                            assert torch.equal(old[:,:,:prefix],new[:,:,:prefix])
                            values.append(error_metrics(plan,old,new))
                    numbers=torch.stack(values).double().cpu().tolist()
                    geometry=[dict(layer=5+i//2,kind=('k','v')[i%2],
                        delta_energy=v[0],local_error=v[1],mean_error=v[2],elements=v[3]) for i,v in enumerate(numbers)]
                    if not timed:
                        def apply():
                            return [plan.predict(a,b) for plan,old,new in zip(memo['plans'],f.reference,current.past_key_values)
                                if plan is not None for a,b in zip(old,new)]
                        for _ in range(2):apply()
                        seconds=[]
                        for _ in range(5):
                            torch.cuda.synchronize();started=time.perf_counter();predictions=apply()
                            torch.cuda.synchronize();seconds.append(time.perf_counter()-started);del predictions
                        result['costs'].append(dict(task=task,id=ident,block=key[2],refine=key[3],
                            construction_seconds=construction,application_seconds=seconds,
                            median_application_seconds=sorted(seconds)[2],
                            scope='Primitive only; paid teacher source computation and attention integration omitted, '
                                'therefore not a deployable forward latency. Outputs allocated in all trials.'))
                        timed=True
                    assert torch.equal(kept,f.kept)
                assert versions==[t._version for pair in f.reference+current.past_key_values for t in pair]
                assert x_version==x._version
                result['records'].append(dict(task=task,id=ident,block=key[2],refine=key[3],
                    suffix_length=x.shape[1],active_masks=len(target),geometry=geometry,
                    current_shallow_geometry_construction_seconds=construction,teacher_shallow_max_error=shallow_error,max_weight_l1=max_weight_l1))
                return current

            backend.selected_forward=observe
            try:
                with LLaDAAttentionBackend(model,'flash') as attention:
                    teacher,actions=backend.generate_active_prefix(model,torch.tensor([ids],device=model.device),
                        gen_length=256,layer=4,keep=1.,pruning=False,trace=True)
            finally:
                backend.selected_forward=original
                handle.remove()
            assert attention.report()['torch_sdpa_calls']==0
            assert len(result['records'])-before+8==teacher.nfe==len(actions)
            ref=next(r for r in reference if r['task']==task and r['id']==ident and r['length']==256 and r['method']=='v1')
            tokens=teacher.output[0,len(ids):].tolist()
            assert ref['nfe']==teacher.nfe and ref['token_sha256']==hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            result['prompts'].append(dict(task=task,id=ident,nfe=teacher.nfe,
                token_nfe_match=True,backend=attention.report()))
            result['summary']=summarize(result['records']);write_json(args.output/'diagnostic.json',result)
            print('SUMMARY',task,ident,json.dumps(result['summary']),flush=True)
    result['by_task']={t:summarize([r for r in result['records'] if r['task']==t]) for t in data}
    result['by_layer']={str(layer):aggregate([g for r in result['records'] for g in r['geometry'] if g['layer']==layer]) for layer in range(5,33)}
    global_ratio=result['summary']['all']['local_relative_error']
    result['gate_passed']=(global_ratio is not None and global_ratio<.75 and all(
        value['local_relative_error'] is not None and value['local_relative_error']<1
        for task in result['by_task'].values() for value in task['by_kind'].values()))
    assert result['implementation']=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    write_json(args.output/'diagnostic.json',result);(args.output/'complete').write_text('OK\n')


if __name__=='__main__':main()
