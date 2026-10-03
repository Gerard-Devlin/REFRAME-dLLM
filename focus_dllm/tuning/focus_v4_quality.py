"""128-per-task development comparison; never regenerate native LLaDA or v1.

Approximate v4 candidates may change actions, tokens and NFE. Official task
scoring, clean full-request timing and actual layer work decide the tradeoff.
Flash uses its pinned decoder plus the previously validated readout component.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
from tqdm import tqdm

from ..common import sha256, write_json
from ..llada_backend import LLaDAAttentionBackend
from ..llada_common import MODEL_ID, REVISION, MASK_ID, prompt_ids
from ..llada_evaluate import load_model as load_native, postprocess_output
from .competitors import generation_prompt, select_samples
from .focus_v4_budget import generate_budget
from .focus_v4_runtime import DEFAULT_OPTIONS, generate_v4
from .focus_v4_evaluate import score
from .flash_exec_evaluate import paired_intervals


CANDIDATES = {'budget05': .05, 'budget20': .20}
NAMES = ['io_borrow', *CANDIDATES]
TASKS = ['humaneval', 'mbpp', 'math', 'gsm8k']


def validate_protocol(engine, task, limit, offset, length):
    if engine not in ('v4', 'flash') or task not in TASKS:
        raise ValueError('Known isolated engines and paper tasks required')
    if limit < 128 or offset != 0 or length not in (256, 512):
        raise ValueError('At least128 development samples/task, offset0, fixed256/512 required')


def paired_quality(before, after, seconds_before, seconds_after, historical=False):
    if not before or any(len(values)!=len(before) for values in (after, seconds_before, seconds_after)):
        raise ValueError('Complete paired quality/timing arrays required')
    keep = [i for i,(a,b) in enumerate(zip(before, after)) if a is not None and b is not None]
    if not keep:
        return dict(paired_examples=0, unresolved_pairs=len(before))
    result = paired_intervals([before[i] for i in keep], [after[i] for i in keep],
                             [seconds_before[i] for i in keep], [seconds_after[i] for i in keep])
    result.update(paired_examples=len(keep), unresolved_pairs=len(before)-len(keep),
        latency_scope=('Historical frozen baseline timings, not a current paired runtime control'
                       if historical else 'Same GPU/current experiment request timing; development only'))
    return result


def summarize(rows, scores, names, frozen, task, length):
    cell = frozen['cells'][f'{task}_{length}']
    metrics, pairs = {}, {}
    ids = [r['id'] for r in rows]
    for name in names:
        outcomes = scores['correct'][name]
        known = [v for v in outcomes if v is not None]
        method_rows = [r[name] for r in rows]
        metrics[name] = dict(examples=len(rows), scored_examples=len(known),
            unresolved_scores=len(rows)-len(known), accuracy=sum(known)/len(known) if known else None,
            mean_seconds=statistics.mean(r['seconds'] for r in method_rows),
            mean_nfe=statistics.mean(r['nfe'] for r in method_rows),
            mean_output_tokens=statistics.mean(r['output_tokens'] for r in method_rows),
            raw_token_ids_available=all(r['token_ids'] is not None for r in method_rows))
        if metrics[name]['raw_token_ids_available']:
            metrics[name].update(truncation_rate=statistics.mean(r['truncated'] for r in method_rows),
                                first_eos_mean=statistics.mean(r['first_eos'] for r in method_rows if r['first_eos'] is not None)
                                              if any(r['first_eos'] is not None for r in method_rows) else None)
        pairs[name] = {}
        for baseline in ('llada', 'v1', 'focus'):
            pairs[name][baseline] = paired_quality([cell['correctness'][baseline][i] for i in ids], outcomes,
                [cell['saved'][baseline][i]['seconds'] for i in ids], [r['seconds'] for r in method_rows], True)
        if name != names[0]:
            pairs[name]['current_io_borrow'] = paired_quality(scores['correct'][names[0]], outcomes,
                [r[names[0]]['seconds'] for r in rows], [r['seconds'] for r in method_rows])
    return dict(task=task, length=length, metrics=metrics, paired=pairs,
        scoring_policy=scores['policy'], ids=ids, scope='128/task development evidence, not independent holdout. '
        'Token equality with old FOCUS is not a quality gate. Local attention mass is not a losslessness certificate.')


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    binding = check_binding(required=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('v4', 'flash'), required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--frozen-development', type=Path, required=True)
    parser.add_argument('--third-party', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--candidates', nargs='+', choices=NAMES, default=NAMES)
    args = parser.parse_args()
    validate_protocol(args.engine, args.task, args.limit, args.offset, args.length)
    frozen = json.loads(args.frozen_development.read_text())
    specification = frozen['datasets'][args.task]
    if args.limit != len(specification['development_ids']):
        raise ValueError('The frozen development reference must cover every requested prompt')
    assert sha256(args.dataset) == specification['sha256']
    samples = select_samples(args.dataset, args.limit, args.offset)
    ids = [str(s.get('id', s.get('task_id'))) for s in samples]
    assert ids == specification['development_ids']
    for path, expected in frozen['frozen_score_files'].items():
        assert sha256(path) == expected, 'Frozen result or scorer output changed'
    source = {p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    scorer = Path(__file__).resolve().parents[1]/'dllm-eval/dllm_eval'
    scoring_source = {p.name:sha256(p) for p in scorer.glob('score*.py')}
    if len(set(args.candidates))!=len(args.candidates) or args.candidates[0]!='io_borrow':
        raise ValueError('Unique candidates with io_borrow current control first required')
    names = args.candidates if args.engine == 'v4' else ['flash_verify_opt']
    external = None
    original = None
    if args.engine == 'v4':
        model, tokenizer = load_native('cuda:0')
        context = LLaDAAttentionBackend(model, 'flash')
        flash_identity = None
    else:
        from .competitors import load_external, load_model
        from .flash_readout import ModelReadout, generator
        from .flash_statistics import statistics as probability_statistics
        cls, external, adaptations = load_external(args.third_party, 'flash_verify')
        original = Path(inspect.getsourcefile(inspect.unwrap(external)))
        model, tokenizer = load_model(SimpleNamespace(method='flash_verify'), cls)
        proxy = ModelReadout(model, compact=True, minimum=64)
        optimized = generator(external, statistics=probability_statistics)
        context = nullcontext()
        flash_identity = dict(generate_source_sha256=sha256(original), adaptations=adaptations,
            sources=json.loads((args.third_party/'sources.json').read_text()), threshold=.9, gamma=.8,
            track_num=4, mask_num=4, verify=True, head_minimum_rows=64,
            engine='Pinned official fused Triton attention/cache plus validated readout/statistics component')
    assert next(model.parameters()).dtype == torch.bfloat16
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    identity = dict(model=MODEL_ID, revision=REVISION, dtype='bfloat16', binding=binding,
        engine=args.engine, task=args.task, length=args.length, block=32, threshold=.90,
        sample_seed=51713, limit=args.limit, offset=args.offset, ids=ids,
        dataset_sha256=sha256(args.dataset), frozen_development_sha256=sha256(args.frozen_development),
        candidates={n:CANDIDATES[n] for n in names if n in CANDIDATES} if args.engine=='v4' else None,
        common_runtime=asdict(DEFAULT_OPTIONS) if args.engine=='v4' else None,
        implementation=source, scoring_implementation=scoring_source, flash=flash_identity,
        scope='Development first128 per task. Native LLaDA/v1 never generated. '
              'Approximate candidates assessed by actual official quality; clean full-request latency includes '
              'all cache initialization, selection syncs, preparation, allocations and readout. '
              'Weight loading and separately logged kernel warm-up are service startup costs.')
    path = args.output/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == identity, 'Resume source/configuration identity changed'
    else:
        write_json(path, identity)

    def run(input_ids, name, backend):
        prompt = torch.tensor([input_ids], device=model.device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        if args.engine == 'v4':
            before = backend.stats.attention_calls
            if name == 'io_borrow':
                result, _, telemetry = generate_v4(model, prompt, gen_length=args.length)
            else:
                result, _, telemetry = generate_budget(model, prompt, gen_length=args.length, budget=CANDIDATES[name])
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            calls = backend.stats.attention_calls-before
            assert calls == result.nfe*model.model.config.n_layers, 'Actual layer work/NFE mismatch'
            assert backend.stats.torch_sdpa_calls == 0, 'Unexpected SDPA fallback'
            tokens = result.output[0, len(input_ids):].tolist()
            assert len(tokens)==args.length and MASK_ID not in tokens
            # Only prompt and tokens enter decoding. Sample-specific cleanup is outside timing below.
            eos = tokens.index(126081) if 126081 in tokens else None
            return dict(seconds=seconds, nfe=result.nfe, token_ids=tokens, first_eos=eos,
                        truncated=eos is None, telemetry=telemetry, attention_calls=calls,
                        peak_gib=result.peak_gib, backend='FlashAttention; SDPA0')
        from .flash_readout import suppress_official_prints
        responses, iterations, calls = [None], [0], [0]
        def count(_module, _inputs):
            calls[0] += 1
        handle = model.register_forward_pre_hook(count)
        try:
            with suppress_official_prints():
                optimized(proxy, [prompt[0]], [len(input_ids)], 1, responses, iterations,
                    gen_length=args.length, block_length=32, threshold=.9, gamma=.8,
                    track_num=4, mask_num=4, verify=True, tokenizer=tokenizer, stop_tokens=[])
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
        finally:
            handle.remove()
        return dict(seconds=seconds, nfe=calls[0], official_iterations=iterations[0], text=responses[0],
            token_ids=None, first_eos=None, truncated=None, raw_token_ids_available=False,
            output_tokens=len(tokenizer.encode(responses[0], add_special_tokens=False)),
            peak_gib=torch.cuda.max_memory_allocated()/2**30, backend='Official fused Triton attention/cache')

    rows = []
    with context as backend:
        warm_ids = prompt_ids(tokenizer, generation_prompt(samples[0]), args.task, preformatted=True)
        warm_started = time.perf_counter()
        controls = {name:run(warm_ids, name, backend) for name in names}
        write_json(args.output/'preflight.json', dict(warmup_seconds=time.perf_counter()-warm_started,
            nfe={n:r['nfe'] for n,r in controls.items()},
            telemetry={n:r.get('telemetry') for n,r in controls.items()},
            scope='Legitimate first development prompt, no scoring/config selection. '
                  'Cold-start/kernel warm-up recorded separately, not an independent quality sample.'))
        for index, sample in enumerate(tqdm(samples, desc=f'FOCUS-v4-128 {args.task}-{args.length}-{args.engine}', ascii=False)):
            path = args.output/'records'/f'{index:04d}.json'
            if path.exists():
                row = json.loads(path.read_text())
                assert row['id']==ids[index] and set(row['methods'])==set(names)
                rows.append(row)
                continue
            input_ids = prompt_ids(tokenizer, generation_prompt(sample), args.task, preformatted=True)
            row = dict(id=ids[index], index=index,
                input_ids_sha256=hashlib.sha256(json.dumps(input_ids).encode()).hexdigest(), methods=names)
            order = names[index%len(names):]+names[:index%len(names)]
            for name in order:
                result = run(input_ids, name, backend)
                if result['token_ids'] is not None:
                    result['text'], result['output_tokens'] = postprocess_output(tokenizer, result['token_ids'], sample, args.task)
                row[name] = result
            write_json(path, row)
            rows.append(row)
        if args.engine == 'v4':
            write_json(args.output/'backend.json', backend.report())
    scoring_started = time.perf_counter()
    scores = score(args.task, samples, rows, names)
    write_json(args.output/'scores.json', scores)
    summary = summarize(rows, scores, names, frozen, args.task, args.length)
    summary['scoring_seconds'] = time.perf_counter()-scoring_started
    write_json(args.output/'summary.json', summary)
    assert source == {p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}, 'Frozen research source changed'
    assert scoring_source == {p.name:sha256(p) for p in scorer.glob('score*.py')}, 'Frozen scoring source changed'
    if original is not None:
        assert sha256(original) == flash_identity['generate_source_sha256']
    (args.output/'complete').write_text('OK\n')
    print('Completed128', args.task, args.length, args.engine, json.dumps(summary['metrics']), flush=True)


if __name__ == '__main__':
    main()
