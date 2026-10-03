"""Single-GPU v4 component screen. Native LLaDA/v1 results are frozen/reused.

Only old FOCUS and its execution ablations are generated. Reference answers
are accessed exclusively after generation, with the frozen uniform scorers.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from tqdm import tqdm

from ..common import sha256, write_json
from ..llada_backend import LLaDAAttentionBackend
from ..llada_common import MODEL_ID, REVISION, MASK_ID, prompt_ids
from ..llada_decode import Result
from ..llada_evaluate import load_model, postprocess_output
from ..llada_pruning import Config, LLaDABlockForward
from .competitors import select_samples, generation_prompt
from .focus_v4_runtime import Options, BlockEngine, commit, generate_v4


VARIANTS = {
    'device': Options(),
    'io_fused': Options(prepared_rope=True, fused_statistics=True),
    'graph_io': Options(graph=True, prepared_rope=True, fused_statistics=True),
    'device_borrow': Options(borrow_prefix=True),
    'io_borrow': Options(prepared_rope=True, fused_statistics=True, borrow_prefix=True),
    'graph_delayed': Options(graph=True, borrow_prefix=True, graph_start=3,
                            graph_min_remaining=16, graph_warmups=0),
}


def action(logits):
    from ..llada_decode import _selected_positions
    tokens = logits.argmax(-1)
    confidence = logits.double().softmax(-1).gather(-1, tokens[..., None]).squeeze(-1)
    take = _selected_positions(confidence, .90)
    return take, tokens[0, take]


@torch.no_grad()
def reference(model, prompt, length, *, trace=False, observer=None):
    """Same original FOCUS physical pruning and exact final-block branch.

    This is a FOCUS component control, not a rerun of native LLaDA or v1.
    Full warm-up output head, all native confidence reductions, stopping and
    formal cache creation are retained. No changes to the main method files.
    """
    torch.cuda.synchronize();started = time.perf_counter()
    x = torch.full((1, prompt.shape[1]+length), MASK_ID, device=prompt.device, dtype=torch.long)
    x[:, :prompt.shape[1]] = prompt
    forward = LLaDABlockForward(model, Config(support_keep_ratio=.3125, target_only_head=True))
    actions, nfe = [], 0
    for offset in range(0, length, 32):
        start = prompt.shape[1]+offset
        output = model(x, use_cache=True)
        target = torch.arange(start, start+32, device=x.device)
        positions, values = commit(x, target, output.logits[:, start:start+32], .90)
        if trace:actions.append((positions.tolist(), values.tolist()))
        nfe += 1
        past = [tuple(t[:, :, :start] for t in pair) for pair in output.past_key_values]
        del output
        while (x[:, start:start+32] == MASK_ID).any():
            target = (x[0, start:start+32] == MASK_ID).nonzero().flatten()
            logits = forward(x[:, start:], target.tolist(), past_key_values=past, use_cache=True)
            if observer is not None and x.shape[1]-start > 32:
                observer(x[:, start:], target, past, logits)
            positions, values = commit(x, start+target, logits, .90)
            if trace:actions.append((positions.tolist(), values.tolist()))
            nfe += 1
            del logits
    torch.cuda.synchronize()
    return Result(x, nfe, time.perf_counter()-started, torch.cuda.max_memory_allocated()/2**30), actions


def score(task, samples, rows, names):
    if task not in ('math', 'gsm8k'):
        from .suite import score as official
        return dict(correct=official(task, samples, rows, names), policy='Official code execution pass@1')
    from dllm_eval.score_answers import assess, MathComparison, policy_hash
    comparison = MathComparison() if task == 'math' else None
    if task == 'math':
        from dllm_eval.score_math import load_metric, installed_utils
        metric = load_metric(installed_utils())
    details = {name:[] for name in names}
    for sample, row in zip(samples, rows):
        if task == 'gsm8k':gold = sample['answer'].rsplit('####', 1)[-1].strip()
        else:gold = metric['remove_boxed'](metric['last_boxed_only_string'](sample['solution']))
        for name in names:
            # Do not silently turn scorer exceptions into wrong answers.
            try:
                result = assess(row[name]['text'], gold, task, comparison)
            except Exception as error:
                result = dict(correct=None, status='scoring_error',
                              error_type=type(error).__name__, error=str(error))
            details[name].append(result)
    return dict(correct={n:[r['correct'] for r in d] for n,d in details.items()},
                details=details, policy='Frozen final-expression-v3', policy_sha256=policy_hash())


def historical_records(main_results, task, length):
    """Read the finished FOCUS table, without regenerating any baseline."""
    path = main_results/f'{task}_g{length}_ours/output/rank_elastic.jsonl'
    records = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        ident = str(record['id'])
        if ident in records:
            raise ValueError('Duplicated frozen prompt ID')
        records[ident] = record['flash_focus_head']
    return records


def paired_summary(rows, scored, active):
    """Unknown scorer outcomes stay unknown, never silently become failures."""
    metrics = {}
    for name in active:
        outcomes = scored['correct'][name]
        known = [value for value in outcomes if value is not None]
        metrics[name] = dict(examples=len(rows), scored_examples=len(known),
            unresolved_scores=len(outcomes)-len(known),
            accuracy=sum(known)/len(known) if known else None,
            mean_seconds=statistics.mean(r[name]['seconds'] for r in rows),
            mean_nfe=statistics.mean(r[name]['nfe'] for r in rows),
            truncation_rate=statistics.mean(r[name]['truncated'] for r in rows))
    return dict(metrics=metrics, scoring_policy=scored['policy'],
        exact_trajectory={n:all(all(r['parity'][n].values()) and
            all(v['tokens_equal_trace'] and v['nfe']==r[n]['nfe'] for v in r['clean'][n]) for r in rows)
            for n in active[1:]},
        score_agreement={n:scored['correct'][n]==scored['correct']['focus_control'] for n in active[1:]},
        historical_agreement={n:dict(text=sum(r['historical'][n]['text'] for r in rows),
            nfe=sum(r['historical'][n]['nfe'] for r in rows), total=len(rows)) for n in active},
        speedups={n:metrics['focus_control']['mean_seconds']/metrics[n]['mean_seconds'] for n in active[1:]})


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    binding = check_binding(required=True)
    p = argparse.ArgumentParser(description=__doc__)
    for task in ('humaneval', 'mbpp', 'math', 'gsm8k'):
        p.add_argument('--'+task+'-dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--main-results', type=Path, required=True)
    p.add_argument('--limit', type=int, default=2)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--lengths', default='256')
    p.add_argument('--variants', default='device_borrow,io_borrow,graph_delayed')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--validation', action='store_true',
                   help='Fixed io_borrow only, offset>=64, all four tasks and both lengths')
    args = p.parse_args()
    names = args.variants.split(',')
    if len(set(names)) != len(names) or not set(names) <= set(VARIANTS):
        raise ValueError('Known unique v4 execution variants only')
    lengths = [int(v) for v in args.lengths.split(',')]
    if any(v not in (256, 512) for v in lengths) or args.repeats < 1:
        raise ValueError('Fixed generation budget and positive repeats required')
    if args.validation and (names!=['io_borrow'] or args.offset<64 or sorted(lengths)!=[256,512]):
        raise ValueError('Validation freezes one common config on unused IDs and both budgets')
    datasets = {t:getattr(args,t+'_dataset') for t in ('humaneval', 'mbpp', 'math', 'gsm8k')}
    samples = {t:select_samples(path,args.limit,args.offset) for t,path in datasets.items()}
    source = {p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    frozen = {str(p.relative_to(args.main_results)):sha256(p)
              for p in args.main_results.glob('*/output/summary.json')}
    assert len(frozen) == 24, 'Completed main baseline table required'
    main_manifest = json.loads((args.main_results/'elastic/manifest.json').read_text())
    for task, path in datasets.items():
        item = next(i for i in main_manifest['jobs'] if i['name']==f'{task}_g256_ours')
        if sha256(path)!=item['identity']['dataset_sha256']:
            raise ValueError('Dataset differs from the completed table')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(model=MODEL_ID, revision=REVISION, binding=binding, dtype='bfloat16',
        cache='prefix', threshold=.90, block=32, lengths=lengths, options={n:asdict(VARIANTS[n]) for n in names},
        sample_seed=51713, offset=args.offset, limit=args.limit, repeats=args.repeats,
        validation=args.validation,
        implementation=source, dataset_sha256={t:sha256(p) for t,p in datasets.items()},
        frozen_main_summaries=frozen,
        scope='Only FOCUS component controls run. Native LLaDA/v1 are frozen and reused, never regenerated. '+
              ('Independent candidate validation' if args.validation else 'Development screen'),
        evidence_scope='Graph/setup/copies charged to every clean request. Exact actions are finite '
              'regression evidence, not a universal numerical or quality theorem.')
    write_json(args.output/'manifest.json', manifest)
    model, tokenizer = load_model('cuda:0')
    torch.set_num_threads(1);torch.manual_seed(1234)
    all_summaries, audits, passed = {}, [], set(names)
    with LLaDAAttentionBackend(model, 'flash') as backend:
        # Few real same-state controls; never pass gold to the observer/model.
        for task, values in samples.items():
            ids = prompt_ids(tokenizer, generation_prompt(values[0]), task, preformatted=True)
            prompt = torch.tensor([ids], device=model.device)
            seen = [0]
            def inspect(ids, target, past, logits):
                if seen[0] >= 2:return
                seen[0] += 1
                snapshot = [(a._version,b._version) for a,b in past]
                for name in names:
                    engine = BlockEngine(model, ids, past, VARIANTS[name])
                    try:
                        # Exercise actual replay for delayed-capture candidates;
                        # these same-state repeats are private audit calls.
                        for _ in range(VARIANTS[name].graph_start+1):
                            predicted = engine.forward(ids, target)
                        old, new = action(logits), action(predicted)
                        report = dict(task=task, state=seen[0], variant=name,
                            target_count=target.numel(), suffix_length=ids.shape[1],
                            max_logit_error=float((logits.float()-predicted.float()).abs().max()),
                            exact_logits=torch.equal(logits, predicted),
                            exact_actions=all(torch.equal(a,b) for a,b in zip(old,new)),
                            formal_cache_versions_equal=snapshot==[(a._version,b._version) for a,b in past])
                        if not all(report[k] for k in ('exact_logits','exact_actions','formal_cache_versions_equal')):
                            passed.discard(name)
                        audits.append(report)
                        write_json(args.output/'same_state_audit.json', dict(records=audits,passed=sorted(passed)))
                    finally:engine.close()
            reference(model,prompt,256,observer=inspect)
        if not passed:
            raise RuntimeError('No execution variant passed exact BF16/logit/action same-state control')
        # One model/weight warm-up only; not an excluded per-request graph setup.
        for task, values in samples.items():
            for length in lengths:
                rows, active = [], ['focus_control']+[n for n in names if n in passed]
                saved = historical_records(args.main_results,task,length)
                out = args.output/f'{task}_{length}'
                for index,sample in enumerate(tqdm(values,desc=f'FOCUS-v4 {task}-{length}',ascii=False)):
                    ident = str(sample.get('id',sample.get('task_id')))
                    ids = prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                    prompt = torch.tensor([ids],device=model.device)
                    row = dict(id=ident,index=index,methods={})
                    for name in active:
                        torch.cuda.reset_peak_memory_stats()
                        if name == 'focus_control':outcome, actions = reference(model,prompt,length,trace=True);extra={}
                        else:outcome, actions, extra = generate_v4(model,prompt,gen_length=length,options=VARIANTS[name],trace=True)
                        tokens = outcome.output[0,len(ids):].tolist()
                        text,count = postprocess_output(tokenizer,tokens,sample,task)
                        row[name] = dict(text=text,token_ids=tokens,nfe=outcome.nfe,actions=actions,
                                        output_tokens=count,truncated=126081 not in tokens,telemetry=extra)
                    control = row['focus_control']
                    row['parity'] = {n:dict(tokens=row[n]['token_ids']==control['token_ids'],
                        actions=row[n]['actions']==control['actions'],nfe=row[n]['nfe']==control['nfe']) for n in active[1:]}
                    row['historical'] = {n:dict(text=row[n]['text']==saved[ident]['text'],
                        nfe=row[n]['nfe']==saved[ident]['nfe'],
                        saved_seconds=saved[ident]['seconds'],saved_nfe=saved[ident]['nfe']) for n in active}
                    row['input_ids_sha256'] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                    row['clean'] = {n:[] for n in active}
                    for repeat in range(args.repeats):
                        order = active if (index+repeat)%2 == 0 else active[::-1]
                        for name in order:
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.synchronize();began = time.perf_counter()
                            if name == 'focus_control':outcome,_ = reference(model,prompt,length);extra={}
                            else:outcome,_,extra = generate_v4(model,prompt,gen_length=length,options=VARIANTS[name])
                            torch.cuda.synchronize();request_seconds=time.perf_counter()-began
                            tokens=outcome.output[0,len(ids):].tolist()
                            row['clean'][name].append(dict(seconds=request_seconds,nfe=outcome.nfe,
                                tokens_equal_trace=tokens==row[name]['token_ids'],peak_gib=outcome.peak_gib,telemetry=extra))
                            row[name]['seconds']=statistics.median(r['seconds'] for r in row['clean'][name])
                    write_json(out/'records'/f'{index:04d}.json',row);rows.append(row)
                    if args.validation and any(not all(row['parity'][n].values()) or
                            not all(v['tokens_equal_trace'] and v['nfe']==row[n]['nfe']
                                    for v in row['clean'][n]) for n in active[1:]):
                        write_json(out/'validation_failure.json',dict(id=ident,parity=row['parity'],
                            reason='Execution candidate changed a commit, token or NFE; records retained'))
                        raise RuntimeError('FOCUS-v4 exact execution validation failed; diagnose before expanding')
                scored = score(task,values,rows,active)
                write_json(out/'scores.json',scored)
                summary = paired_summary(rows,scored,active)
                write_json(out/'summary.json',summary);(out/'complete').write_text('OK\n')
                all_summaries[f'{task}_{length}']=summary
                write_json(args.output/'summary.json',all_summaries)
                print('Completed',task,length,json.dumps(summary),flush=True)
        assert backend.stats.torch_sdpa_calls == 0,'Unexpected backend fallback'
        write_json(args.output/'backend.json',dict(**backend.report(),
            note='Python counters include setup/capture, not graph replay. Committed NFE is independently logged.'))
    assert source=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},'Tuning source changed'
    assert frozen=={str(p.relative_to(args.main_results)):sha256(p)
                   for p in args.main_results.glob('*/output/summary.json')},'Frozen main summaries changed'
    (args.output/'complete').write_text('OK\n')


if __name__ == '__main__':main()
