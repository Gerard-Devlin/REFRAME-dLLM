"""Fixed16same-ID clean-coverage and background-age ablations."""
import argparse
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from focus_dllm.tuning.flash_readout import ModelReadout,suppress_official_prints
from focus_v6.audit_probe import forbid_sdpa
from .cache import capture_private_kv
from .quality128 import load_baseline
from .query_budget import generate_budget


CONFIGS={'coverage_recent':'recent','coverage_age':'age'}


def projected_pair(value,layout):
    """Use the compact head's supported virtual slices, never full materialize."""
    logits=value.logits.squeeze(0)
    return logits[layout.clean].detach().clone(),logits[layout.audit].detach().clone()


def maximum_error(a,b,mask_id=126336):
    # Original selection overwrites the ineligible MASK logit with -inf. That
    # policy column is excluded; all eligible logits must still agree exactly.
    delta=a.float()-b.float();delta[:,mask_id]=0
    if not bool(torch.isfinite(delta).all()):raise AssertionError('nonfinite eligible logits')
    return float(delta.abs().max())


def summarize(rows,baseline):
    result={}
    for name in CONFIGS:
        selected=[r for r in rows if r['method']==name]
        if not selected:continue
        n=len(selected);seconds=statistics.mean(r['result']['seconds'] for r in selected)
        result[name]=dict(completed=n,correct=sum(r['correct'] for r in selected),
            accuracy=sum(r['correct'] for r in selected)/n,mean_seconds=seconds,
            mean_nfe=statistics.mean(r['result']['nfe'] for r in selected),
            mean_output_tokens=statistics.mean(r['result']['output_tokens'] for r in selected),
            truncations=sum(r['result']['truncated'] for r in selected),
            mean_clean_coverage=statistics.mean(g['clean'] for r in selected for g in r['result']['query_geometry']),
            mean_audit_width=statistics.mean(g['draft'] for r in selected for g in r['result']['query_geometry']),
            historical_engineering_focus_speedup=statistics.mean(baseline['optimized_seconds'][:n])/seconds,
            scope='16 reused development IDs; not independent accuracy/noninferiority evidence')
    return result


@torch.no_grad()
def main():
    from focus_dllm.common import sha256,write_json
    from focus_dllm.llada_common import MODEL_ID,REVISION,prompt_ids
    from focus_dllm.tuning.competitors import generation_prompt,load_external,load_model,select_samples
    from focus_dllm.tuning.gpu_contract import check_binding
    from dllm_eval.score_humaneval import clean_completion,check
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('third-party','dataset','root','old-revision','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    samples=select_samples(args.dataset,128,0)
    baseline=load_baseline(args.root,args.dataset,samples,MODEL_ID,REVISION)
    samples=samples[:16];write_json(args.output/'baseline.json',baseline)
    cls,external,adaptation=load_external(args.third_party,'flash_verify')
    raw,tokenizer=load_model(SimpleNamespace(method='flash_verify'),cls)
    model=ModelReadout(raw,compact=True,minimum=32)
    torch.set_num_threads(1);torch.manual_seed(1234)
    sources={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    scorer=Path(__file__).parents[1]/'focus_dllm/dllm-eval/dllm_eval'
    scorers={p.name:sha256(p) for p in scorer.glob('score*.py')}
    manifest=dict(model=MODEL_ID,revision=REVISION,binding=check_binding(required=True),task='humaneval',
        length=256,seed=51713,offset=0,limit=16,ids=baseline['ids'][:16],configurations=CONFIGS,
        implementation=sources,scorers=scorers,dataset_sha256=sha256(args.dataset),adaptation=adaptation,
        original_revision_summary_sha256=sha256(args.old_revision/'evaluation/summary.json'),
        query_budget=64,threshold=.9,gamma=.8,online_paid_shadow_calls=0,
        scope='Fixed same16development IDs; separate query-budget and background-age question')
    write_json(args.output/'manifest.json',manifest)
    count=[0]
    def counter(_model,_args):count[0]+=1
    handle=raw.register_forward_pre_hook(counter)
    def run(ids,tracking,*,clean_limit=32,draft_limit=8,observer=None,shadow_calls=0):
        count[0]=0;torch.cuda.synchronize();start=time.perf_counter()
        with forbid_sdpa(),suppress_official_prints():
            result=generate_budget(model,tokenizer,external,ids,length=256,clean_limit=clean_limit,
                                   draft_limit=draft_limit,tracking=tracking,observer=observer)
        torch.cuda.synchronize();result['seconds']=time.perf_counter()-start
        if result['nfe']+shadow_calls!=count[0]:raise AssertionError('NFE/real model count mismatch')
        if not all(g['query_rows']==64 for g in result['query_geometry']):
            raise AssertionError('unreported packet query work')
        result['output_tokens']=len(tokenizer.encode(result['text'],add_special_tokens=False))
        return result
    controls=[]
    def observer(current):
        if current['packets']:return
        layout=current['layout'];bank=current['bank'];saved=[(k.clone(),v.clone()) for k,v in bank]
        def shadow(query):
            with capture_private_kv() as captured:
                value=model(query,use_cache=True,positions=current['positions'],lengths=current['lengths'],
                    focus_head_rows=(layout.clean.start,layout.width-layout.clean.start))
            return projected_pair(value,layout),captured
        repeated,captured=shadow(current['query'])
        original=(current['clean'].detach().clone(),current['audit'].detach().clone())
        max_error=max(maximum_error(a,b) for a,b in zip(repeated,original))
        max_kv=max(float((a-b).abs().max()) for pair,other in zip(current['captured'],captured) for a,b in zip(pair,other))
        assert max_error==0 and max_kv==0
        mutations=[]
        for j in (0,layout.candidates-1):
            query=current['query'].clone();old=int(query[0,layout.draft.start+j])
            alternate=tokenizer.encode('0' if old!=tokenizer.encode('0',add_special_tokens=False)[0] else '1',add_special_tokens=False)[0]
            assert alternate!=old and alternate!=126336
            query[0,layout.draft.start+j]=alternate
            logits,_=shadow(query)
            clean_error=maximum_error(logits[0],original[0])
            protected_error=maximum_error(logits[1][:j+1],original[1][:j+1])
            assert clean_error==0 and protected_error==0
            mutations.append(dict(draft_index=j,clean_max_error=clean_error,own_and_earlier_audit_max_error=protected_error))
        assert all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(bank,saved))
        controls.append(dict(layout=dict(clean=layout.clean_count,draft=layout.candidates,tracked=layout.tracked),
            repeat_max_error=max_error,private_kv_max_error=max_kv,mutations=mutations,
            public_bank_unchanged=True,shadow_model_calls=3))
    try:
        legacy=[]
        for index,sample in enumerate(samples[:2]):
            ids=prompt_ids(tokenizer,generation_prompt(sample),'humaneval',preformatted=True)
            control=run(ids,'recent',clean_limit=16,draft_limit=16)
            old=json.loads((args.old_revision/f'evaluation/records/{index:04d}_admission_only.json').read_text())['result']
            assert all(control[key]==old[key] for key in ('text','raw_token_ids','nfe','packets'))
            legacy.append(dict(id=sample['task_id'],tokens_actions_text_nfe_exact=True,excluded_from_report=True))
        write_json(args.output/'legacy_controls.json',legacy)
        ids=prompt_ids(tokenizer,generation_prompt(samples[0]),'humaneval',preformatted=True)
        warms={name:run(ids,tracking,observer=observer,shadow_calls=3)['seconds'] for name,tracking in CONFIGS.items()}
        write_json(args.output/'gpu_controls.json',controls)
        write_json(args.output/'warmups.json',dict(seconds=warms,excluded_from_report=True,
            shadow_model_calls=sum(c['shadow_model_calls'] for c in controls)))
        rows=[];records=args.output/'records';records.mkdir();names=list(CONFIGS)
        for index,sample in enumerate(samples):
            ids=prompt_ids(tokenizer,generation_prompt(sample),'humaneval',preformatted=True)
            order=names[index%2:]+names[:index%2]
            for name in order:
                result=run(ids,CONFIGS[name]);row=dict(index=index,id=sample['task_id'],method=name,result=result)
                path=records/f'{index:04d}_{name}.json';write_json(path,row)
                code=clean_completion(sample['prompt'],result['text'],sample['entry_point'])
                row['correct']=check(code,sample['test'],sample['entry_point'],6.)
                rows.append(row);write_json(path,row)
                print('SCORE',index,sample['task_id'],name,row['correct'],result['seconds'],result['nfe'],flush=True)
            summary=summarize(rows,baseline)
            write_json(args.output/'summary.json',summary)
            write_json(args.output/'progress.json',dict(completed_questions=index+1,total_questions=16,summary=summary))
        assert sources=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
        assert scorers=={p.name:sha256(p) for p in scorer.glob('score*.py')}
        (args.output/'complete').write_text('OK\n');print('FINAL',json.dumps(summary),flush=True)
    finally:handle.remove()


if __name__=='__main__':main()
