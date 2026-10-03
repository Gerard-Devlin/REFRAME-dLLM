"""Three fixed implementation/admission controls on16same development questions."""
import argparse
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from focus_dllm.tuning.flash_readout import ModelReadout,suppress_official_prints
from focus_v6.audit_probe import forbid_sdpa
from .quality128 import load_baseline
from .revision import generate_revised


CONFIGS={'refresh_only':(True,False),'admission_only':(False,True),'both':(True,True)}


def summarize(rows,baseline,old):
    result={}
    for name in CONFIGS:
        selected=[r for r in rows if r['method']==name]
        if not selected:continue
        n=len(selected);correct=sum(r['correct'] for r in selected)
        mean=statistics.mean(r['result']['seconds'] for r in selected)
        result[name]=dict(completed=n,correct=correct,accuracy=correct/n,mean_seconds=mean,
            mean_nfe=statistics.mean(r['result']['nfe'] for r in selected),
            mean_output_tokens=statistics.mean(r['result']['output_tokens'] for r in selected),
            mean_full_refreshes=statistics.mean(len(r['result']['epoch_barriers']) for r in selected),
            truncations=sum(r['result']['truncated'] for r in selected),
            original_v7_correct=sum(r['correct'] for r in old[:n]),
            focus_correct=sum(baseline['correct'][:n]),
            historical_engineering_focus_speedup=statistics.mean(baseline['optimized_seconds'][:n])/mean,
            scope='Same reused development IDs; historical baseline timing, no quality noninferiority claim')
    return result


@torch.no_grad()
def main():
    from focus_dllm.common import sha256,write_json
    from focus_dllm.llada_common import MODEL_ID,REVISION,prompt_ids
    from focus_dllm.tuning.competitors import generation_prompt,load_external,load_model,select_samples
    from focus_dllm.tuning.gpu_contract import check_binding
    from dllm_eval.score_humaneval import clean_completion,check
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--third-party',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--old-v7',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    samples=select_samples(args.dataset,128,0)
    baseline=load_baseline(args.root,args.dataset,samples,MODEL_ID,REVISION)
    old=[json.loads(p.read_text()) for p in sorted((args.old_v7/'evaluation/records').glob('*.json'))]
    assert len(old)==128 and [r['id'] for r in old]==baseline['ids']
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
        original_v7_summary_sha256=sha256(args.old_v7/'evaluation/summary.json'),
        all_refreshes_own_paid=True,scope='Fixed factorial implementation/admission development screen')
    write_json(args.output/'manifest.json',manifest)
    count=[0]
    def counter(_model,_args):count[0]+=1
    handle=raw.register_forward_pre_hook(counter)
    def run(ids,refresh,admission):
        count[0]=0;torch.cuda.synchronize();start=time.perf_counter()
        with forbid_sdpa(),suppress_official_prints():
            result=generate_revised(model,tokenizer,external,ids,length=256,refresh=refresh,admission=admission)
        torch.cuda.synchronize();result['seconds']=time.perf_counter()-start
        if result['nfe']!=count[0]:raise AssertionError('paid refresh NFE/call count mismatch')
        result['output_tokens']=len(tokenizer.encode(result['text'],add_special_tokens=False))
        return result
    try:
        ids=prompt_ids(tokenizer,generation_prompt(samples[0]),'humaneval',preformatted=True)
        legacy=run(ids,False,False);reference=old[0]['result']
        assert all(legacy[key]==reference[key] for key in ('text','raw_token_ids','nfe','packets'))
        write_json(args.output/'legacy_control.json',dict(exact_tokens_actions_text_nfe=True,
                   seconds=legacy['seconds'],excluded_from_report=True))
        warms={name:run(ids,*flags)['seconds'] for name,flags in CONFIGS.items()}
        write_json(args.output/'warmups.json',dict(seconds=warms,excluded_from_report=True))
        rows=[];records=args.output/'records';records.mkdir()
        names=list(CONFIGS)
        for index,sample in enumerate(samples):
            ids=prompt_ids(tokenizer,generation_prompt(sample),'humaneval',preformatted=True)
            order=names[index%3:]+names[:index%3]
            for name in order:
                result=run(ids,*CONFIGS[name])
                row=dict(index=index,id=sample['task_id'],method=name,result=result)
                path=records/f'{index:04d}_{name}.json';write_json(path,row)
                code=clean_completion(sample['prompt'],result['text'],sample['entry_point'])
                row['correct']=check(code,sample['test'],sample['entry_point'],6.)
                rows.append(row);write_json(path,row)
                print('SCORE',index,sample['task_id'],name,row['correct'],result['seconds'],result['nfe'],flush=True)
            summary=summarize(rows,baseline,old)
            write_json(args.output/'summary.json',summary)
            write_json(args.output/'progress.json',dict(completed_questions=index+1,total_questions=16,summary=summary))
        assert sources=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
        assert scorers=={p.name:sha256(p) for p in scorer.glob('score*.py')}
        (args.output/'complete').write_text('OK\n')
        print('FINAL',json.dumps(summary),flush=True)
    finally:handle.remove()


if __name__=='__main__':main()
