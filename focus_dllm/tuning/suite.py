"""Frozen-candidate paired checks on the four paper tasks, with scored logs."""
import argparse
import json
from pathlib import Path
import random
import sys

from tqdm import tqdm

from . import stream_sweep
from . import conservative
from .run import VARIANTS, load_model, summarize
from ..common import write_json, sha256
from ..llada_common import MODEL_ID, REVISION
from ..llada_evaluate import postprocess_output, percentile


def score(task, samples, records, methods):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dllm-eval'))
    if task == 'gsm8k':
        from dllm_eval.score_gsm8k import score as gsm_score
        official = gsm_score({str(s['id']): s for s in samples}, records)
        return {name: [r['methods'][name]['flexible-extract'] for r in official['details']]
                for name in methods}
    values = {name: [] for name in methods}
    if task == 'math':
        from dllm_eval.score_math import load_metric, installed_utils, score_record
        metric = load_metric(installed_utils())
        for sample, record in zip(samples, records):
            result = score_record(record, sample, metric)
            for name in methods:
                values[name].append(result['methods'][name]['exact_match'])
    elif task == 'humaneval':
        from dllm_eval.score_humaneval import clean_completion, check
        for sample, record in zip(samples, records):
            for name in methods:
                code = clean_completion(sample['prompt'], record[name]['text'], sample['entry_point'])
                values[name].append(check(code, sample['test'], sample['entry_point'], 6.))
    else:
        from dllm_eval.score_mbpp import clean_completion, check
        for sample, record in zip(samples, records):
            for name in methods:
                values[name].append(check(clean_completion(record[name]['text']), sample['test_list'], 6.))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', choices=['zero_l2', 'zero_l4', 'stream2_l4', 'stream4_l4', 'stream8_l4',
                    'stream2_l8', 'stream4_l8', 'stream4_l4_t95', 'support16_r4', 'support16_r2'], required=True)
    for task in ('gsm8k', 'math', 'humaneval', 'mbpp'):
        parser.add_argument('--'+task+'-dataset', type=Path, required=True)
    parser.add_argument('--lengths', default='256,512')
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--tasks', default='gsm8k,humaneval,mbpp,math')
    parser.add_argument('--cache-ablation', action='store_true')
    args = parser.parse_args()
    names = ['v1', 'v1_dual', args.candidate]
    if args.cache_ablation:
        names.insert(2, 'stream4_full')
    lengths = [int(x) for x in args.lengths.split(',')]
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = args.tasks.split(',')
    assert tasks and len(set(tasks)) == len(tasks) and set(tasks) <= {'gsm8k','math','humaneval','mbpp'}
    datasets = {t: getattr(args, t+'_dataset') for t in tasks}
    prepared_datasets = {}
    for task, path in datasets.items():
        samples = json.loads(path.read_text(encoding='utf-8'))
        random.Random(51713).shuffle(samples)
        assert 0 < args.limit <= len(samples)
        prepared_datasets[task] = samples[:args.limit]
    identity = dict(model=MODEL_ID, revision=REVISION, candidate=args.candidate, configs={n: VARIANTS[n] for n in names},
                    lengths=lengths, limit=args.limit, seed=51713,
                    datasets={t: sha256(p) for t,p in datasets.items()},
                    ids={t: [s.get('id', s.get('task_id')) for s in rows] for t,rows in prepared_datasets.items()},
                    implementation={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')})
    manifest = args.output / 'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == identity, 'Suite settings changed'
    else:
        write_json(manifest, identity)
    model, tokenizer = load_model('cuda:0')
    all_results = {}
    for task, samples in prepared_datasets.items():
        for length in lengths:
            key = f'{task}_{length}'
            output = args.output / key
            warm_sample = dict(samples[0])
            warm_sample['paper_prompt'] = warm_sample.get('paper_prompt', warm_sample.get('prompt'))
            for name in names:
                conservative.run_one(model, tokenizer, warm_sample, VARIANTS[name], 64)
            rows = []
            for index, sample in enumerate(tqdm(samples, desc=key, ascii=False)):
                record_path = output / 'records' / f'{index:04d}.json'
                ident = sample.get('id', sample.get('task_id'))
                record = json.loads(record_path.read_text()) if record_path.exists() else dict(id=ident,index=index)
                assert record['id'] == ident
                prepared = dict(sample)
                prepared['paper_prompt'] = sample.get('paper_prompt', sample.get('prompt'))
                order = list(names)
                random.Random(index+1234).shuffle(order)
                for name in order:
                    if name in record:
                        continue
                    result = conservative.run_one(model, tokenizer, prepared, VARIANTS[name], length)
                    text, tokens = postprocess_output(tokenizer, result['token_ids'], sample, task)
                    result.update(text=text, output_tokens=tokens)
                    record[name] = result
                    write_json(record_path, record)
                rows.append(record)
            correctness = score(task, samples, rows, names)
            metrics = {}
            for name in names:
                timing = [r[name]['seconds'] for r in rows]
                metrics[name] = dict(examples=len(rows), accuracy=sum(correctness[name])/len(rows),
                    mean_seconds=sum(timing)/len(timing), p50_seconds=percentile(timing,.5),
                    p95_seconds=percentile(timing,.95), mean_nfe=sum(r[name]['nfe'] for r in rows)/len(rows),
                    mean_output_tokens=sum(r[name]['output_tokens'] for r in rows)/len(rows),
                    truncation=sum(r[name]['truncated'] for r in rows)/len(rows),
                    throughput=sum(r[name]['output_tokens'] for r in rows)/sum(timing))
            from .analysis import paired_intervals
            details = [dict(methods={n: {'flexible-extract': correctness[n][i]} for n in names})
                       for i in range(len(rows))]
            confidence = paired_intervals({'metrics':metrics,'official':{'details':details}}, rows)
            result = dict(metrics=metrics, paired_comparisons=confidence,
                          correctness=correctness,
                          scope='Development screen; no claim of statistical losslessness')
            write_json(output / 'summary.json', result)
            all_results[key] = result
            write_json(args.output / 'summary.json', all_results)
            print(f'Completed {key}: '+json.dumps(metrics), flush=True)
    (args.output / 'complete').write_text('OK\n')


if __name__ == '__main__':
    main()
