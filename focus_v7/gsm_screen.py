"""Fixed coverage-age GSM8K development screen; saved baselines only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from focus_dllm.tuning.flash_readout import ModelReadout, suppress_official_prints
from focus_v6.audit_probe import forbid_sdpa
from .query_budget import generate_budget


def saved_baselines(root, dataset, samples):
    from focus_dllm.common import sha256
    frozen_path = root / 'cpu_frozen_development128_20261003.json'
    frozen = json.loads(frozen_path.read_text())
    ids = [str(s['id']) for s in samples]
    data = frozen['datasets']['gsm8k']
    if (ids != data['development_ids'] or len(ids) != 128
            or sha256(dataset) != data['sha256'] or frozen['seed'] != 51713):
        raise ValueError('Use the frozen same128 GSM8K development IDs/data')
    hashes = {p: h for p, h in frozen['frozen_score_files'].items()
              if '/gsm8k_g256_' in Path(p).as_posix()}
    if len(hashes) != 3 or any(sha256(Path(p)) != h for p, h in hashes.items()):
        raise ValueError('Saved baseline generation records changed')
    cell = frozen['cells']['gsm8k_256']
    return dict(ids=ids, correctness=cell['correctness'], saved=cell['saved'],
                metrics=cell['metrics'], dataset_sha256=data['sha256'],
                frozen_sha256=sha256(frozen_path), original_records_sha256=hashes,
                baseline_regenerated=False,
                scope='Frozen final-expression-v3 scores; historical same-ID latency')


def summarize(rows, baseline):
    from focus_dllm.tuning.flash_exec_evaluate import paired_intervals
    n = len(rows)
    if not n:
        return dict(completed=0, total=128)
    seconds = [r['result']['seconds'] for r in rows]
    outcomes = [r['correct'] for r in rows]
    pairs = {}
    for name in ('llada', 'v1', 'focus'):
        old = [baseline['correctness'][name][r['id']] for r in rows]
        times = [baseline['saved'][name][r['id']]['seconds'] for r in rows]
        pairs[name] = paired_intervals(old, outcomes, times, seconds)
        pairs[name].update(saved_correct=sum(old), historical_latency=True)
    return dict(completed=n, total=128, correct=sum(outcomes), accuracy=sum(outcomes)/n,
                mean_seconds=statistics.mean(seconds),
                mean_nfe=statistics.mean(r['result']['nfe'] for r in rows),
                mean_output_tokens=statistics.mean(r['result']['output_tokens'] for r in rows),
                truncations=sum(r['result']['truncated'] for r in rows),
                answer_statuses=dict(Counter(r['assessment']['status'] for r in rows)),
                comparisons=pairs,
                scope='Fixed GSM8K development screen; not independent holdout/noninferiority evidence')


@torch.no_grad()
def main():
    from focus_dllm.common import sha256, write_json
    from focus_dllm.llada_common import MODEL_ID, REVISION, prompt_ids
    from focus_dllm.tuning.competitors import generation_prompt, load_external, load_model, select_samples
    from focus_dllm.tuning.gpu_contract import check_binding
    from dllm_eval.score_answers import assess, policy_hash
    from dllm_eval.score_gsm8k import score as official_score
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('third-party', 'dataset', 'root', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    samples = select_samples(args.dataset, 128, 0)
    baseline = saved_baselines(args.root, args.dataset, samples)
    write_json(args.output/'baseline.json', baseline)
    assert policy_hash() == '4871dd10d0918d64fbcfa1101cdc512eb310bb88e00775336af529d14933b754'
    files = {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
    scorer_dir = Path(__file__).parents[1]/'focus_dllm/dllm-eval/dllm_eval'
    scorers = {p.name: sha256(p) for p in scorer_dir.glob('score*.py')}
    cls, external, adaptation = load_external(args.third_party, 'flash_verify')
    raw, tokenizer = load_model(SimpleNamespace(method='flash_verify'), cls)
    model = ModelReadout(raw, compact=True, minimum=32)
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    write_json(args.output/'manifest.json', dict(model=MODEL_ID, revision=REVISION,
        binding=check_binding(required=True), task='gsm8k', length=256, limit=128,
        seed=51713, offset=0, ids=baseline['ids'], dataset_sha256=sha256(args.dataset),
        implementation=files, scorers=scorers, primary_policy_sha256=policy_hash(),
        primary_policy='Frozen final-expression-v3, as for saved baselines',
        configuration=dict(clean_limit=32, draft_limit=8, tracking='age', threshold=.9,
                           gamma=.8, query_width=64, maximum_commit=16),
        adaptation=adaptation, baseline_generation=False,
        scope='User requests fast fixed GSM8K screen of unchanged coverage_age algorithm'))
    count = [0]
    def counter(_model, _args):
        count[0] += 1
    handle = raw.register_forward_pre_hook(counter)
    def run(ids):
        count[0] = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        with forbid_sdpa(), suppress_official_prints():
            result = generate_budget(model, tokenizer, external, ids, length=256,
                                     clean_limit=32, draft_limit=8, tracking='age')
        torch.cuda.synchronize()
        result['seconds'] = time.perf_counter()-start
        assert count[0] == result['nfe'], 'Real model-call/NFE mismatch'
        assert all(g['query_rows'] == 64 for g in result['query_geometry'])
        result['output_tokens'] = len(tokenizer.encode(result['text'], add_special_tokens=False))
        return result
    try:
        ids = prompt_ids(tokenizer, generation_prompt(samples[0]), 'gsm8k', preformatted=True)
        warm = run(ids)
        write_json(args.output/'warmup.json', dict(seconds=warm['seconds'], nfe=warm['nfe'],
                                                  excluded_from_report=True))
        print('WARMUP', warm['seconds'], flush=True)
        rows = []
        for index, sample in enumerate(samples):
            ids = prompt_ids(tokenizer, generation_prompt(sample), 'gsm8k', preformatted=True)
            result = run(ids)
            row = dict(index=index, id=str(sample['id']), method='coverage_age', result=result)
            path = args.output/'records'/f'{index:04d}.json'
            # Persist the generated answer before the scoring reference is accessed.
            write_json(path, row)
            row['assessment'] = assess(result['text'], sample['answer'].rsplit('####', 1)[-1].strip(), 'gsm8k')
            row['correct'] = row['assessment']['correct']
            write_json(path, row)
            rows.append(row)
            print('SCORE', index+1, row['id'], row['correct'], result['seconds'], result['nfe'], flush=True)
            if (index+1) % 8 == 0 or index == 127:
                summary = summarize(rows, baseline)
                write_json(args.output/'summary.json', summary)
                write_json(args.output/'progress.json', dict(completed=index+1, total=128, summary=summary))
        # Keep the original lm-eval extraction metric alongside the fixed primary.
        official = official_score({str(s['id']): s for s in samples},
                                 [dict(id=r['id'], coverage_age=r['result']) for r in rows])
        write_json(args.output/'official_scores.json', official)
        assert files == {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
        assert scorers == {p.name: sha256(p) for p in scorer_dir.glob('score*.py')}
        assert baseline == saved_baselines(args.root, args.dataset, samples)
        write_json(args.output/'summary.json', summary)
        (args.output/'complete').write_text('OK\n')
        print('FINAL', json.dumps(summary), flush=True)
    finally:
        handle.remove()


if __name__ == '__main__':
    main()
