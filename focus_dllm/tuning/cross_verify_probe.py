"""Small complementary-view diagnostic, never an online sampler or certificate.

The base call is native. After its real release, two independent current states
mask complementary candidate groups. Each own candidate slot remains MASK.
No draft model, future teacher state, answers or teacher-future KV enter views.
Final teacher outputs are retrospective diagnostic references, not gold labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from .cross_verify import make_views,acceptance


def summary(records):
    accepted=sum(len(r['accepted_positions']) for r in records)
    correct=sum(sum(r.get('matches_teacher_final',[])) for r in records)
    empty=sum(len(r.get('oracle_future_ordinary_empty',[])) for r in records)
    base=sum(r['cost']['stable_native_seconds'] for r in records)
    cross=sum(r['cost']['setup_seconds']+r['cost']['stable_batched_verify_seconds'] for r in records)
    return dict(windows=len(records),accepted=accepted,teacher_final_matches=correct,
        teacher_final_agreement=correct/accepted if accepted else None,
        reference_ordinary_action_sets_empty=empty,
        mean_reference_empty_per_window=empty/len(records) if records else None,
        fixed_window_cost_ratio=cross/base if base else None,
        scope='Retrospective reference scheduling and fixed-state cost only; no online saved NFE or E2E accuracy/speed.')


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    binding=check_binding(required=True)
    from ..common import sha256,write_json
    from ..llada_backend import LLaDAAttentionBackend
    from ..llada_common import MODEL_ID,REVISION,MASK_ID,prompt_ids
    from ..llada_decode import _selected_positions
    from . import backend
    from .retention_sweep import sample_slice
    from .run import load_model
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
        datasets={t:sha256(p) for t,p in data.items()},records=[],prompts=[],
        implementation={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        configuration=dict(length=256,block=32,candidates=8,threshold=.90,
            blocks=[0,2,4],first_eligible_ordinary_only=True,seed=51713,offset=0,samples_per_task=2),
        hypothesis='Complementary conditional views can jointly validate enough low-confidence proposals to repay a batched verifier.',
        gate='At least 32 accepted positions; >=.98 agreement with teacher final tokens; '
            'mean >=2 later ordinary reference action sets empty per window; total fixed-window '
            'setup+stable batched verification cost <=2 native post-release calls. '
            'A pass only warrants actual small free-generation testing, never losslessness.',
        scope='Six reused dev prompts. New verification contexts intentionally differ from native decoding. '
            'Two-view agreement is not a distribution/action certificate; self-supporting wrong proposals can pass. '
            'No teacher future logits/KV, reference answers or future actions feed inputs. '
            'Future ledger subtraction does not prove online NFE reduction, because conditioning will change. '
            'Native/verify use common BF16/Flash and consumed-position heads; no SDPA, new model or training.')
    write_json(args.output/'diagnostic.json',result)
    model,tokenizer=load_model('cuda:0');torch.set_num_threads(1)
    forbidden=set(tokenizer.all_special_ids)|{MASK_ID,126081}
    original=backend.selected_forward
    def measured(function):
        torch.cuda.synchronize();started=time.perf_counter();value=function()
        torch.cuda.synchronize();return value,time.perf_counter()-started
    for task,path in data.items():
        for sample in sample_slice(json.loads(path.read_text()),0,2):
            ident=sample.get('id',sample.get('task_id'))
            ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
            kinds=[];seen=set();memo={};windows=[];ordinary_count=0
            def observe(model,x,target,**kwargs):
                nonlocal ordinary_count
                current=original(model,x,target,**kwargs)
                call_index=len(kinds)
                if kwargs.get('past_key_values') is None:
                    kinds.append('warm');memo['refine']=0;return current
                kinds.append('ordinary');ordinary_count+=1;memo['refine']+=1
                block=(256-x.shape[1])//32;key=(task,ident,block,memo['refine'])
                previous=prior[key]
                top=current.logits.argmax(-1)
                conf=current.logits.double().softmax(-1).gather(-1,top.unsqueeze(-1)).squeeze(-1)
                take=_selected_positions(conf,.90)
                assert conf[0].tolist()==previous['teacher_confidence']
                assert target[take].tolist()==previous['teacher_selected']
                assert top[0,take].tolist()==previous['teacher_values']
                if block not in (0,2,4) or block in seen:return current
                eligible=~take
                for token in forbidden:eligible&=top[0]!=token
                if int(eligible.sum())<2:return current
                seen.add(block)
                versions=[t._version for pair in kwargs['past_key_values'] for t in pair]
                x_version=x._version
                torch.cuda.synchronize();started=time.perf_counter()
                remaining_indices=eligible.nonzero().flatten()
                order=conf[0,remaining_indices].argsort(descending=True,stable=True)[:8]
                chosen=remaining_indices[order]
                candidates=target[chosen];drafts=top[0,chosen]
                state=x.clone();state[0,target[take]]=top[0,take]
                torch.cuda.synchronize()
                proposal_seconds=time.perf_counter()-started
                remaining=(state[0,:32]==MASK_ID).nonzero().flatten()
                def native():return original(model,state,remaining,**kwargs).logits
                next_logits,native_first=measured(native)
                next_top=next_logits.argmax(-1)
                next_conf=next_logits.double().softmax(-1).gather(-1,next_top.unsqueeze(-1)).squeeze(-1)
                future=prior[(task,ident,block,memo['refine']+1)]
                assert next_conf[0].tolist()==future['teacher_confidence'], 'Native post-release control changed'
                assert remaining[_selected_positions(next_conf,.90)].tolist()==future['teacher_selected']
                torch.cuda.synchronize();started=time.perf_counter()
                views=make_views(state,candidates,drafts,mask_id=MASK_ID,special_ids=forbidden)
                batch_past=[tuple(t.repeat(2,1,1,1) for t in pair) for pair in kwargs['past_key_values']]
                torch.cuda.synchronize();setup=time.perf_counter()-started+proposal_seconds
                batch_versions=[t._version for pair in batch_past for t in pair]
                def verify():
                    logits=original(model,views.canvas,views.positions,past_key_values=batch_past,use_cache=False).logits
                    accepted,p=acceptance(logits,views)
                    return logits,accepted,p
                (batch_logits,accepted,probability),verify_first=measured(verify)
                # Numerical control: same views in separate batch-one calls.
                # It is diagnostic work, not silently included in an online path.
                row_logits=torch.cat([original(model,views.canvas[row:row+1],views.positions,
                    past_key_values=kwargs['past_key_values'],use_cache=False).logits for row in (0,1)])
                row_accepted,row_probability=acceptance(row_logits,views)
                serial_seconds=[];batch_seconds=[]
                for _ in range(3):
                    _,seconds=measured(native);serial_seconds.append(seconds)
                    _,seconds=measured(verify);batch_seconds.append(seconds)
                assert versions==[t._version for pair in kwargs['past_key_values'] for t in pair]
                assert batch_versions==[t._version for pair in batch_past for t in pair]
                assert x_version==x._version
                absolute=len(ids)+block*32
                record=dict(task=task,id=ident,block=block,refine=memo['refine'],teacher_call_index=call_index,
                    suffix_length=x.shape[1],candidate_positions=(candidates+absolute).tolist(),drafts=drafts.tolist(),
                    base_probabilities=conf[0,chosen].tolist(),verification_probabilities=probability.tolist(),
                    accepted_positions=(candidates[accepted]+absolute).tolist(),accepted_values=drafts[accepted].tolist(),
                    batch_one_acceptance_same=torch.equal(row_accepted,accepted),
                    batch_one_max_logit_error=float((batch_logits.float()-row_logits.float()).abs().max()),
                    batch_one_probability_error=float((probability-row_probability).abs().max()),
                    native_post_release_control_match=True,
                    cost=dict(setup_seconds=setup,native_first_seconds=native_first,verify_first_seconds=verify_first,
                        stable_native_seconds=statistics.median(serial_seconds),
                        stable_batched_verify_seconds=statistics.median(batch_seconds),
                        serial_trials=serial_seconds,batched_trials=batch_seconds,
                        scope='Fixed-state CPU+CUDA wall time with synchronization. No graph optimization. '
                            'Setup includes views/owned KV copies/proposal selection. Base proposal call is paid by native generation. '
                            'No claim of actual online speedup or pure kernel time.'))
                windows.append(record)
                return current
            backend.selected_forward=observe
            try:
                with LLaDAAttentionBackend(model,'flash') as attention:
                    teacher,actions=backend.generate_active_prefix(model,torch.tensor([ids],device=model.device),
                        gen_length=256,layer=4,keep=1.,pruning=False,trace=True)
            finally:backend.selected_forward=original
            assert attention.report()['torch_sdpa_calls']==0
            assert ordinary_count+8==teacher.nfe==len(actions)==len(kinds)
            ref=next(r for r in reference if r['task']==task and r['id']==ident and r['length']==256 and r['method']=='v1')
            tokens=teacher.output[0,len(ids):].tolist()
            assert ref['nfe']==teacher.nfe and ref['token_sha256']==hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            for record in windows:
                positions=record['accepted_positions'];values=record['accepted_values']
                record['matches_teacher_final']=[tokens[p-len(ids)]==v for p,v in zip(positions,values)]
                oracle={p for p,match in zip(positions,record['matches_teacher_final']) if match}
                record['oracle_future_ordinary_empty']=[i for i in range(record['teacher_call_index']+1,len(actions))
                    if kinds[i]=='ordinary' and actions[i][0] and set(actions[i][0])<=oracle]
            result['records'].extend(windows)
            result['prompts'].append(dict(task=task,id=ident,nfe=teacher.nfe,windows=len(windows),
                token_nfe_match=True,backend=attention.report()))
            result['summary']=summary(result['records']);write_json(args.output/'diagnostic.json',result)
            print('SUMMARY',task,ident,json.dumps(result['summary']),flush=True)
    s=result['summary']
    result['gate_passed']=(s['accepted']>=32 and s['teacher_final_agreement']>=.98
        and s['mean_reference_empty_per_window']>=2 and s['fixed_window_cost_ratio']<=2)
    assert result['implementation']=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    write_json(args.output/'diagnostic.json',result);(args.output/'complete').write_text('OK\n')


if __name__=='__main__':main()
