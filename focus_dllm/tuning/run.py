"""Persistent-model, paired GSM8K development sweep on an assigned GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import torch
from tqdm import tqdm

from ..common import write_json, sha256
from ..llada_backend import LLaDAAttentionBackend
from ..llada_common import MODEL_ID, REVISION, prompt_ids, MASK_ID
from ..llada_evaluate import load_model, run_method, postprocess_output, percentile
from ..llada_decode import _selected_positions
from .backend import selected_forward, generate_active_prefix


VARIANTS = {
    'v1': dict(kind='v1', layer=4, keep=1., threshold=.90),
    'current': dict(kind='current', layer=4, keep=.3125, threshold=.90),
    'active_exact': dict(kind='active', layer=4, keep=1., threshold=.90),
    'keep50_l4': dict(kind='active', layer=4, keep=.5, threshold=.90),
    'keep75_l4': dict(kind='active', layer=4, keep=.75, threshold=.90),
    'keep50_l8': dict(kind='active', layer=8, keep=.5, threshold=.90),
    'keep50_l12': dict(kind='active', layer=12, keep=.5, threshold=.90),
    'keep50_l4_t95': dict(kind='active', layer=4, keep=.5, threshold=.95),
}


def score_rows(samples, records):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dllm-eval'))
    from dllm_eval.score_gsm8k import score
    return score({str(r['id']): r for r in samples}, records)


def prepare_dataset(paper, output, seed=98271):
    from datasets import load_dataset
    train = load_dataset('openai/gsm8k', 'main', split='train')
    reference = json.loads(paper.read_text())[0]
    context = reference['paper_prompt']
    boundary = context.rfind('\n\nQuestion:')
    if boundary < 0:
        raise ValueError('Unknown official five-shot delimiter')
    demonstrations = context[:boundary + 2]
    eligible = [i for i, doc in enumerate(train) if doc['question'] not in demonstrations]
    random.Random(seed).shuffle(eligible)
    rows = []
    for i in eligible[:288]:
        doc = train[i]
        rows.append(dict(id=f'train:{i}', question=doc['question'],
                         answer=doc['answer'].split('####')[-1].strip(),
                         paper_prompt=demonstrations + f"Question: {doc['question']}\nAnswer:",
                         generation_kwargs=reference['generation_kwargs']))
    write_json(output, dict(seed=seed, demonstrations_sha256=hashlib.sha256(
        demonstrations.encode()).hexdigest(), development=rows[:32], validation=rows[32:160],
        confirmation=rows[160:], scope='Training split only; no test-set tuning.'))


@torch.no_grad()
def audit(model, ids):
    x = torch.full((1, len(ids) + 64), MASK_ID, device=model.device)
    x[0, :len(ids)] = torch.tensor(ids, device=model.device)
    target = torch.arange(len(ids), len(ids) + 32, device=model.device)
    errors, matches = [], []
    with LLaDAAttentionBackend(model, 'flash'):
        full = model(x, use_cache=True)
        active = selected_forward(model, x, target, use_cache=True)
        a, b = full.logits[:, target], active.logits
        errors.append(float((a.float() - b.float()).abs().max()))
        matches.append(torch.equal(a.argmax(-1), b.argmax(-1)))
        for p, q in zip(full.past_key_values, active.past_key_values):
            assert all(torch.equal(u, v) for u, v in zip(p, q)), 'Cache changed'
        past = [tuple(t[:, :, :len(ids)] for t in pair) for pair in full.past_key_values]
        del full, active
        target2 = torch.arange(32, device=model.device)
        raw = model(x[:, len(ids):], past_key_values=past, use_cache=True).logits[:, :32]
        short = selected_forward(model, x[:, len(ids):], target2,
                                 past_key_values=past, use_cache=True).logits
        errors.append(float((raw.float() - short.float()).abs().max()))
        matches.append(torch.equal(raw.argmax(-1), short.argmax(-1)))
        for u, v in ((a, b), (raw, short)):
            def action(z):
                top = z.argmax(-1)
                confidence = z.double().softmax(-1).gather(-1, top.unsqueeze(-1)).squeeze(-1)
                take = _selected_positions(confidence, .90)
                return take, top[0, take]
            assert all(torch.equal(m, n) for m, n in zip(action(u), action(v))), 'Head action parity failed'
    assert all(matches), 'Head token parity failed'
    return dict(max_logit_errors=errors, top1_equal=matches, cache_equal=True)


def run_one(model, tokenizer, sample, config, gen_length):
    ids = prompt_ids(tokenizer, sample['paper_prompt'], 'gsm8k', preformatted=True)
    if config['kind'] in {'v1', 'current'}:
        args = SimpleNamespace(prune_after_layer=config['layer'], support_keep_ratio=config['keep'],
              context_dominant_ratio=1., contextual_ratio=0., support_contextual_ratio=0.,
              context_merge_weight=.5, secondary_prune_after_layer=0, secondary_support_ratio=1.,
              cache_mode='prefix', gen_length=gen_length, block_length=32,
              decoding_mode='threshold', threshold=config['threshold'])
        outcome = run_method(model, ids, args, 'flash_native' if config['kind'] == 'v1' else 'flash_focus_head')
        outcome.pop('records')
        token_ids = outcome.pop('token_ids')
    else:
        with LLaDAAttentionBackend(model, 'flash') as backend:
            result, _ = generate_active_prefix(model, torch.tensor([ids], device=model.device),
                        gen_length=gen_length, layer=config['layer'], keep=config['keep'],
                        threshold=config['threshold'], pruning=config['keep'] < 1)
        token_ids = result.output[0, len(ids):].tolist()
        outcome = dict(nfe=result.nfe, seconds=result.seconds, peak_gib=result.peak_gib,
                       backend=backend.report())
    text, output_tokens = postprocess_output(tokenizer, token_ids, sample, 'gsm8k')
    return dict(**outcome, text=text, output_tokens=output_tokens, token_ids=token_ids,
                truncated=126081 not in token_ids, canvas_tokens=gen_length)


def summarize(samples, records, names):
    official = score_rows(samples, records)
    metrics = {}
    for name in names:
        rows = [r[name] for r in records]
        seconds = sum(r['seconds'] for r in rows)
        metrics[name] = dict(examples=len(rows), accuracy=official['results'][name]['flexible-extract'],
                            mean_seconds=seconds/len(rows),
                            p95_seconds=percentile([r['seconds'] for r in rows], .95),
                            mean_nfe=sum(r['nfe'] for r in rows)/len(rows),
                            mean_output_tokens=sum(r['output_tokens'] for r in rows)/len(rows),
                            tokens_per_second=sum(r['output_tokens'] for r in rows)/seconds,
                            truncation=sum(r['truncated'] for r in rows)/len(rows))
    return dict(metrics=metrics, official=official)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--prepare-from', type=Path)
    parser.add_argument('--split', choices=['development', 'validation', 'confirmation'], default='development')
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--gen-length', type=int, default=256)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.prepare_from and not args.dataset.exists():
        prepare_dataset(args.prepare_from, args.dataset)
    names = args.variants.split(',')
    assert names[0] == 'v1' and len(set(names)) == len(names)
    configs = {name: VARIANTS[name] for name in names}
    args.output.mkdir(parents=True, exist_ok=True)
    samples = json.loads(args.dataset.read_text())[args.split][:args.limit]
    manifest = dict(model=MODEL_ID, revision=REVISION, split=args.split,
                    dataset_sha256=sha256(args.dataset), configs=configs, gen_length=args.gen_length,
                    ids=[r['id'] for r in samples],
                    implementation={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')})
    path = args.output / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'Configuration changed during resume'
    else:
        write_json(path, manifest)
    model, tokenizer = load_model('cuda:0')
    torch.set_num_threads(1)
    ids = prompt_ids(tokenizer, samples[0]['paper_prompt'], 'gsm8k', preformatted=True)
    write_json(args.output / 'audit.json', audit(model, ids))
    for kind in ('v1', 'active_exact'):
        run_one(model, tokenizer, samples[0], VARIANTS[kind], 64)
    records = []
    for index, sample in enumerate(tqdm(samples, desc=f'{args.split} paired prompts', ascii=False)):
        record_path = args.output / 'records' / f'{index:04d}.json'
        record = json.loads(record_path.read_text()) if record_path.exists() else dict(id=sample['id'])
        assert record['id'] == sample['id']
        order = list(names)
        random.Random(1234 + index).shuffle(order)
        for name in order:
            if name not in record:
                record[name] = run_one(model, tokenizer, sample, configs[name], args.gen_length)
                write_json(record_path, record)
                print(f"[{index+1}/{len(samples)}] {name}: {record[name]['seconds']:.3f}s nfe={record[name]['nfe']}", flush=True)
        records.append(record)
        if (index+1) % 8 == 0 or index+1 == len(samples):
            result = summarize(samples[:index+1], records, names)
            write_json(args.output / 'summary.json', result)
            print(json.dumps(result['metrics'], ensure_ascii=False), flush=True)
    (args.output / 'complete').write_text('OK\n')


if __name__ == '__main__':
    main()
