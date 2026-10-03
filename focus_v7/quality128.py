"""User-requested fixed v7 evaluation against saved same-ID original FOCUS."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from focus_dllm.tuning.flash_readout import ModelReadout,suppress_official_prints
from focus_v6.audit_probe import forbid_sdpa
from .generation import generate


def load_baseline(root,dataset,samples,model,revision):
    from focus_dllm.common import sha256
    frozen_path=root/'cpu_frozen_development128_20261003.json'
    frozen=json.loads(frozen_path.read_text())
    data=frozen['datasets']['humaneval']
    ids=[str(s['task_id']) for s in samples]
    if ids!=data['development_ids'] or sha256(dataset)!=data['sha256'] or len(ids)!=128:
        raise ValueError('same128IDs/data are required')
    folder=root/'focus_v4_quality128_20261003/humaneval_256_v4'
    manifest=json.loads((folder/'manifest.json').read_text())
    if (manifest['model']!=model or manifest['revision']!=revision or manifest['length']!=256
            or manifest['sample_seed']!=51713 or manifest['dataset_sha256']!=data['sha256']
            or [str(i) for i in manifest['ids']]!=ids or not (folder/'complete').exists()):
        raise ValueError('saved FOCUS model/config/IDs differ')
    score=json.loads((folder/'scores.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    cell=frozen['cells']['humaneval_256']
    correct=[cell['correctness']['focus'][i] for i in ids]
    if score['correct']['io_borrow']!=correct or sum(correct)!=43:
        raise ValueError('saved FOCUS engineering changed accuracy')
    records=[json.loads(p.read_text()) for p in sorted((folder/'records').glob('*.json'))]
    if [r['id'] for r in records]!=ids:
        raise ValueError('saved FOCUS per-prompt records differ')
    original=[cell['saved']['focus'][i]['seconds'] for i in ids]
    optimized=[r['io_borrow']['seconds'] for r in records]
    paths=[frozen_path,folder/'manifest.json',folder/'scores.json',folder/'summary.json']
    for path,h in frozen['frozen_score_files'].items():
        if '/humaneval_g256_ours/' in path and sha256(Path(path))!=h:
            raise ValueError('original FOCUS record changed')
    return dict(ids=ids,correct=correct,original_seconds=original,optimized_seconds=optimized,
        original_metrics=cell['metrics']['focus'],optimized_metrics=summary['metrics']['io_borrow'],
        frozen_sha256={str(p):sha256(p) for p in paths},dataset_sha256=data['sha256'],
        original_records_sha256={path:h for path,h in frozen['frozen_score_files'].items()
            if '/humaneval_g256_ours/' in path},baseline_regenerated=False,
        scope='Same prompt/model/scoring; historical saved latency, not contemporaneous paired GPU timing')


def summarize(rows,baseline):
    from focus_dllm.tuning.flash_exec_evaluate import paired_intervals
    if not rows:
        return dict(completed=0,total=128)
    n=len(rows)
    outcomes=[r['correct'] for r in rows]
    seconds=[r['result']['seconds'] for r in rows]
    scores=baseline['correct'][:n]
    comparison=paired_intervals(scores,outcomes,baseline['optimized_seconds'][:n],seconds)
    comparison['timing_scope']='Saved historical FOCUS timings on sameIDs; interval omits cross-run hardware/system variation'
    return dict(completed=n,total=128,correct=sum(outcomes),accuracy=sum(outcomes)/n,
        mean_seconds=statistics.mean(seconds),mean_nfe=statistics.mean(r['result']['nfe'] for r in rows),
        mean_committed=statistics.mean(r['result']['committed_count'] for r in rows),
        mean_output_tokens=statistics.mean(r['result']['output_tokens'] for r in rows),
        truncation_rate=statistics.mean(r['result']['truncated'] for r in rows),
        first_eos_mean=statistics.mean(r['result']['first_eos'] for r in rows if r['result']['first_eos'] is not None)
            if any(r['result']['first_eos'] is not None for r in rows) else None,
        comparison_vs_engineering_focus=comparison,
        original_focus_historical_latency_ratio=sum(baseline['original_seconds'][:n])/sum(seconds),
        focus_correct_v7_wrong=[r['id'] for r,b in zip(rows,scores) if b and not r['correct']],
        v7_correct_focus_wrong=[r['id'] for r,b in zip(rows,scores) if r['correct'] and not b],
        scope='User-authorized128 reused development prompts; no independent generalization or noninferiority claim')


@torch.no_grad()
def main():
    from focus_dllm.common import sha256,write_json
    from focus_dllm.llada_common import MODEL_ID,REVISION,prompt_ids
    from focus_dllm.tuning.competitors import generation_prompt,load_external,load_model,select_samples
    from focus_dllm.tuning.gpu_contract import check_binding
    from dllm_eval.score_humaneval import clean_completion,check
    from tqdm import tqdm
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--third-party',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    samples=select_samples(args.dataset,128,0)
    baseline=load_baseline(args.root,args.dataset,samples,MODEL_ID,REVISION)
    write_json(args.output/'baseline.json',baseline)
    sources={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    scorer=Path(__file__).parents[1]/'focus_dllm/dllm-eval/dllm_eval'
    scorer_hashes={p.name:sha256(p) for p in scorer.glob('score*.py')}
    cls,external,adaptation=load_external(args.third_party,'flash_verify')
    raw,tokenizer=load_model(SimpleNamespace(method='flash_verify'),cls)
    compact=ModelReadout(raw,compact=True,minimum=32)
    torch.set_num_threads(1);torch.manual_seed(1234)
    write_json(args.output/'manifest.json',dict(model=MODEL_ID,revision=REVISION,binding=check_binding(required=True),
        task='humaneval',limit=128,offset=0,seed=51713,length=256,k=16,implementation=sources,
        scorers=scorer_hashes,dataset_sha256=sha256(args.dataset),ids=baseline['ids'],adaptation=adaptation,
        scope='Explicit user request expands fixed v7; previous tiny failed gates are preserved',
        common_readout='compact minimum32, original FP64 probability reduction',baseline_generation=False))
    count=[0]
    def hook(_model,_args):count[0]+=1
    handle=raw.register_forward_pre_hook(hook)
    def run(ids):
        count[0]=0;torch.cuda.synchronize();started=time.perf_counter()
        with forbid_sdpa(),suppress_official_prints():
            result=generate(compact,tokenizer,external,ids,length=256)
        torch.cuda.synchronize()
        result['seconds']=time.perf_counter()-started
        if result['nfe']!=count[0]:raise AssertionError('NFE/calls mismatch')
        result['output_tokens']=len(tokenizer.encode(result['text'],add_special_tokens=False))
        return result
    try:
        warm_ids=prompt_ids(tokenizer,generation_prompt(samples[0]),'humaneval',preformatted=True)
        warm=run(warm_ids)
        write_json(args.output/'warmup.json',dict(seconds=warm['seconds'],nfe=warm['nfe'],excluded_from_report=True))
        print('WARMUP',warm['seconds'],flush=True)
        rows=[]
        records=args.output/'records';records.mkdir()
        for index,sample in enumerate(tqdm(samples,desc='FOCUS-v7 HumanEval128',ascii=False)):
            ids=prompt_ids(tokenizer,generation_prompt(sample),'humaneval',preformatted=True)
            result=run(ids)
            code=clean_completion(sample['prompt'],result['text'],sample['entry_point'])
            correct=check(code,sample['test'],sample['entry_point'],6.)
            row=dict(index=index,id=sample['task_id'],result=result,correct=correct)
            rows.append(row);write_json(records/f'{index:04d}.json',row)
            write_json(args.output/'progress.json',dict(completed=index+1,total=128,correct=sum(r['correct'] for r in rows)))
            if (index+1)%16==0:
                print('PARTIAL',json.dumps(summarize(rows,baseline)),flush=True)
        summary=summarize(rows,baseline)
        summary['baselines']=dict(original_focus=baseline['original_metrics'],engineering_focus=baseline['optimized_metrics'])
        summary['warmup_seconds']=warm['seconds']
        assert sources=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
        assert scorer_hashes=={p.name:sha256(p) for p in scorer.glob('score*.py')}
        write_json(args.output/'summary.json',summary)
        (args.output/'complete').write_text('OK\n')
        print('FINAL',json.dumps(summary),flush=True)
    finally:
        handle.remove()


if __name__=='__main__':main()
