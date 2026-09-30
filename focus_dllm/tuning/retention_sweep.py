"""Development-only sweep: refresh history sooner and update more future queries.

Uses existing FOCUS execution paths. No training or task-specific decoder rules.
All methods see the same prompts; answers are used only by the scorers.
"""
import argparse
import json
from pathlib import Path
import random

from tqdm import tqdm

from . import conservative
from .analysis import paired_intervals
from .run import VARIANTS, audit, load_model
from .suite import score
from ..common import sha256, write_json
from ..llada_backend import LLaDAAttentionBackend
from ..llada_common import MODEL_ID, REVISION, prompt_ids
from ..llada_evaluate import percentile, postprocess_output


CONFIGS = {
    'stream1_l8': dict(kind='stream', layer=8, keep=0., threshold=.90, refresh=1),
    'support16_r1_l4': dict(kind='support', layer=4, keep=0., threshold=.90, refresh=1, support=16),
    'support32_r1_l4': dict(kind='support', layer=4, keep=0., threshold=.90, refresh=1, support=32),
    'support16_r1_l8': dict(kind='support', layer=8, keep=0., threshold=.90, refresh=1, support=16),
    'support32_r2_l4': dict(kind='support', layer=4, keep=0., threshold=.90, refresh=2, support=32),
}


def sample_slice(samples, offset, limit, seed=51713):
    if offset < 0 or limit < 1 or offset + limit > len(samples):
        raise ValueError('Invalid development/validation sample range')
    copied = list(samples)
    random.Random(seed).shuffle(copied)
    return copied[offset:offset + limit]


def summarize(task, samples, rows, names):
    correctness = score(task, samples, rows, names)
    metrics = {}
    for name in names:
        timings = [r[name]['seconds'] for r in rows]
        metrics[name] = dict(examples=len(rows), accuracy=sum(correctness[name])/len(rows),
            mean_seconds=sum(timings)/len(rows), p50_seconds=percentile(timings, .5),
            p95_seconds=percentile(timings, .95),
            mean_nfe=sum(r[name]['nfe'] for r in rows)/len(rows),
            mean_output_tokens=sum(r[name]['output_tokens'] for r in rows)/len(rows),
            truncation=sum(r[name]['truncated'] for r in rows)/len(rows),
            throughput=sum(r[name]['output_tokens'] for r in rows)/sum(timings))
    details = [dict(methods={n: {'flexible-extract': correctness[n][i]} for n in names})
               for i in range(len(rows))]
    comparisons = {base: paired_intervals(
        {'metrics': metrics, 'official': {'details': details}}, rows, baseline=base)
        for base in ('v1', 'v1_dual')}
    return dict(metrics=metrics, correctness=correctness, paired_comparisons=comparisons,
                scope='Development screening; point estimates are not a quality guarantee')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    for task in ('gsm8k', 'math', 'humaneval', 'mbpp'):
        parser.add_argument('--'+task+'-dataset', type=Path)
    parser.add_argument('--variants', default='stream2_l8,'+','.join(CONFIGS))
    parser.add_argument('--tasks', default='humaneval,mbpp,math')
    parser.add_argument('--lengths', default='256')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--sample-role', choices=['development', 'validation'], default='development')
    args = parser.parse_args()
    VARIANTS.update(CONFIGS)
    candidates = args.variants.split(',')
    names = ['v1', 'v1_dual'] + candidates
    if len(set(names)) != len(names) or any(n not in VARIANTS for n in names):
        parser.error('Unknown or duplicate variants')
    tasks = args.tasks.split(',')
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= {'gsm8k','math','humaneval','mbpp'}:
        parser.error('Invalid tasks')
    lengths = [int(n) for n in args.lengths.split(',')]
    if not lengths or len(set(lengths)) != len(lengths) or any(n < 32 or n % 32 for n in lengths):
        parser.error('Generation lengths must be positive multiples of 32')
    if args.sample_role == 'validation' and args.offset < 64:
        parser.error('The first 64 task prompts were already used for screening')
    prepared, datasets = {}, {}
    for task in tasks:
        path = getattr(args, task+'_dataset')
        if path is None:
            parser.error(f'--{task}-dataset is required')
        datasets[task] = sha256(path)
        prepared[task] = sample_slice(json.loads(path.read_text(encoding='utf-8')), args.offset, args.limit)
    manifest = dict(model=MODEL_ID, revision=REVISION, variants={n: VARIANTS[n] for n in names},
        tasks=tasks, lengths=lengths, offset=args.offset, limit=args.limit, sample_role=args.sample_role,
        seed=51713, datasets=datasets,
        ids={t: [s.get('id', s.get('task_id')) for s in samples] for t,samples in prepared.items()},
        implementation={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
        backend='BF16 FlashAttention for all variants', block_length=32)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume requires unchanged code, data and configuration')
    else:
        write_json(manifest_path, manifest)
    model, tokenizer = load_model('cuda:0')
    first_task = tasks[0]
    first = prepared[first_task][0]
    # Audit the shared selected-output-head path, not an approximation promise.
    ids = prompt_ids(tokenizer, first.get('paper_prompt', first.get('prompt')), first_task, preformatted=True)
    write_json(args.output / 'audit.json', audit(model, ids))
    results = {}
    for task, samples in prepared.items():
        for length in lengths:
            key = f'{task}_{length}'
            folder = args.output / key
            warm = dict(samples[0], paper_prompt=samples[0].get('paper_prompt', samples[0].get('prompt')))
            for name in names:
                conservative.run_one(model, tokenizer, warm, VARIANTS[name], 64)
            rows = []
            for index, sample in enumerate(tqdm(samples, desc=key, ascii=False)):
                path = folder / 'records' / f'{index:04d}.json'
                ident = sample.get('id', sample.get('task_id'))
                row = json.loads(path.read_text()) if path.exists() else dict(id=ident, index=index)
                if row['id'] != ident:
                    raise ValueError('Prompt identity mismatch')
                order = list(names)
                random.Random(1234+index).shuffle(order)
                # Explicit whitelist: no gold answer or reference implementation
                # is passed to a decoder, including on code tasks.
                decoder_sample = dict(paper_prompt=sample.get('paper_prompt', sample.get('prompt')))
                for name in order:
                    if name in row:
                        continue
                    outcome = conservative.run_one(model, tokenizer, decoder_sample, VARIANTS[name], length)
                    text, count = postprocess_output(tokenizer, outcome['token_ids'], sample, task)
                    outcome.update(text=text, output_tokens=count)
                    row[name] = outcome
                    write_json(path, row)
                rows.append(row)
            result = summarize(task, samples, rows, names)
            write_json(folder / 'summary.json', result)
            results[key] = result
            write_json(args.output / 'summary.json', results)
            print('Completed '+key+': '+json.dumps(result['metrics']), flush=True)
    (args.output / 'complete').write_text('OK\n')


if __name__ == '__main__':
    main()
