"""Falsifiable same-state diagnostic, not an online quality/speedup result.

Run only after a GPU ownership check. Teacher inputs contain only the paper
prompt. Shadow calls never feed back into the native teacher trajectory.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from ..common import sha256, write_json
from ..llada_backend import LLaDAAttentionBackend
from ..llada_common import MODEL_ID, REVISION, prompt_ids
from ..llada_decode import _selected_positions
from ..llada_pruning import Config, LLaDABlockForward
from . import backend
from .focus_v2 import FocusV2Forward, ProxyConfig
from .retention_sweep import sample_slice
from .run import load_model


def prediction(logits):
    top = logits.argmax(-1)
    p = logits.double().softmax(-1).gather(-1, top.unsqueeze(-1)).squeeze(-1)
    return top, p, _selected_positions(p, .90)


def compare(reference, other, targets):
    top, p, take = prediction(reference)
    alt, q, selected = prediction(other)
    shared = take & selected
    wrong = int((top[0, shared] != alt[0, shared]).sum())
    same = torch.equal(take, selected)
    eos = take & (top[0] == 126081)
    other_eos = selected & (alt[0] == 126081)
    return dict(action_match=same and wrong == 0, selection_match=same,
                wrong_shared_values=wrong, selected=targets[selected].tolist(),
                values=alt[0, selected].tolist(), confidence=q[0].tolist(),
                revealed=int(selected.sum()), selected_eos=int(other_eos.sum()),
                extra_eos=int((other_eos & ~eos).sum()),
                threshold_down=int(((p >= .90) & (q < .90)).sum()),
                threshold_up=int(((p < .90) & (q >= .90)).sum()),
                relative_rms=float((other.float() - reference.float()).square().mean().sqrt()
                                   / reference.float().square().mean().sqrt().clamp_min(1e-30)),
                max_logit_error=float((other.float() - reference.float()).abs().max()))


def summarize(rows, names):
    result = {}
    for name in names:
        selected = [row['methods'][name] for row in rows if name in row['methods']]
        if not selected:
            continue
        result[name] = dict(states=len(selected), matches=sum(r['action_match'] for r in selected),
                            selection_mismatches=sum(not r['selection_match'] for r in selected),
                            shared_token_errors=sum(r['wrong_shared_values'] for r in selected),
                            extra_eos=sum(r['extra_eos'] for r in selected),
                            mean_relative_rms=sum(r['relative_rms'] for r in selected) / len(selected))
    return result


def fixed_forward_timing(methods, native, model, ids, targets, past, seed):
    """Synchronized, unlogged fixed-workload forward cost including pooling.

    Original native attention is the denominator. The augmented full control
    is reported separately and never used to inflate the method's speedup.
    """
    functions = {'native': lambda: native(model, ids, targets, past_key_values=past, use_cache=False).logits}
    functions.update({name: (lambda f=f: f(ids, targets.tolist(), past_key_values=past))
                      for name, f in methods.items()})
    names = list(functions)
    rng = random.Random(seed)
    for _ in range(3):
        rng.shuffle(names)
        for name in names:
            functions[name]()
    times = {name: [] for name in names}
    for _ in range(5):
        rng.shuffle(names)
        for name in names:
            torch.cuda.synchronize(ids.device)
            started = time.perf_counter()
            functions[name]()
            torch.cuda.synchronize(ids.device)
            times[name].append(time.perf_counter() - started)
    return {name: dict(seconds=values, median_seconds=sorted(values)[len(values) // 2])
            for name, values in times.items()}


@torch.no_grad()
def run(args):
    if args.output.exists():
        raise ValueError('Refuse to overwrite a diagnostic; keep failed records')
    args.output.mkdir(parents=True)
    known = json.loads((args.reference_root / 'allstate_probe.json').read_text())
    prior = {(r['task'], str(r['id']), r['block'], r['refine']): r for r in known['records']}
    prior_runs = json.loads((args.reference_root / 'termination_probe.json').read_text())['records']
    config = ProxyConfig(mass_implementation=args.mass_implementation)
    model, tokenizer = load_model('cuda:0')
    torch.set_num_threads(1)
    original = backend.selected_forward
    physical = LLaDABlockForward(model, Config(prune_after_layer=4, support_keep_ratio=.3125,
                                             target_only_head=True))
    # Adapter signature only: the unchanged physical wrapper uses use_cache=False.
    class Physical:
        records = physical.records
        def __call__(self, ids, targets, past_key_values=None):
            return physical(ids, targets, prune=True, past_key_values=past_key_values, use_cache=False)
    methods = dict(focus_v1=Physical(),
                   pool_unweighted=FocusV2Forward(model, replace(config, weighted=False)),
                   focus_v2=FocusV2Forward(model, config))
    numerical = FocusV2Forward(model, replace(config, keep_ratio=1, force_mass_kernel=True))
    identity = FocusV2Forward(model, replace(config, keep_ratio=1))
    datasets = {t: getattr(args, t + '_dataset') for t in ('humaneval', 'mbpp', 'math')}
    result = dict(model=MODEL_ID, revision=REVISION, method='focus-v2', configuration=asdict(config),
                  length=256, threshold=.90, seed=51713, offset=0,
                  datasets={t: sha256(p) for t, p in datasets.items()}, records=[], prompts=[],
                  timings=[], numerical_controls=[], implementation={},
                  scope='Six reused development prompts, same-state shadow actions against native teacher. '
                        'Not gold, not online generation, not a quality guarantee or end-to-end speedup. '
                        'All pooling, proportional Flash padding and real RoPE cost is paid. '
                        'No teacher future KV is passed to any candidate.')
    for path in Path(__file__).parent.glob('*.py'):
        result['implementation'][path.name] = sha256(path)
    for task, path in datasets.items():
        samples = sample_slice(json.loads(path.read_text()), 0, 2)
        for sample in samples:
            sid = sample.get('id', sample.get('task_id'))
            ids = prompt_ids(tokenizer, sample.get('paper_prompt', sample.get('prompt')), task, preformatted=True)
            memo = dict(refine=0, audited=False)
            before = len(result['records'])
            def observe(current_model, canvas, targets, **kwargs):
                current = original(current_model, canvas, targets, **kwargs)
                if kwargs.get('use_cache') and kwargs.get('past_key_values') is None:
                    memo['refine'], memo['audited'] = 0, False
                    return current
                memo['refine'] += 1
                key = (task, str(sid), (256 - canvas.shape[1]) // 32, memo['refine'])
                reference = prior[key]
                top, p, take = prediction(current.logits)
                assert p[0].tolist() == reference['teacher_confidence'], 'Teacher numerical path changed'
                assert targets[take].tolist() == reference['teacher_selected']
                assert top[0, take].tolist() == reference['teacher_values']
                past = kwargs['past_key_values']
                versions = [t._version for pair in past for t in pair]
                canvas_version = canvas._version
                row = dict(task=task, id=sid, block=key[2], refine=key[3],
                           suffix_length=canvas.shape[1], active_masks=len(targets), methods={})
                for name, forward in methods.items():
                    forward.records.clear()
                    shadow = forward(canvas, targets.tolist(), past_key_values=past)
                    row['methods'][name] = compare(current.logits, shadow, targets)
                    row['methods'][name]['compression'] = (dict(forward.records[-1]) if forward.records else None)
                    del shadow
                result['records'].append(row)
                if not memo['audited']:
                    full = identity(canvas, targets.tolist(), past_key_values=past)
                    assert torch.equal(full, current.logits), 'Exact identity wrapper changed logits'
                    padded = numerical(canvas, targets.tolist(), past_key_values=past)
                    result['numerical_controls'].append(dict(task=task, id=sid, block=key[2],
                        exact_identity_max_error=0., **compare(current.logits, padded, targets)))
                    del full, padded
                    memo['audited'] = True
                # Fixed initial/middle/later blocks, before inspecting outcomes.
                if key[2] in (0, 3, 5) and key[3] == 1:
                    costs = fixed_forward_timing(methods, original, model, canvas, targets, past,
                                                 1234 + len(result['timings']))
                    result['timings'].append(dict(task=task, id=sid, block=key[2], refine=key[3],
                                                 suffix_length=canvas.shape[1], active_masks=len(targets), methods=costs))
                assert versions == [t._version for pair in past for t in pair], 'Formal prefix mutated'
                assert canvas_version == canvas._version, 'Shadow leaked into teacher canvas'
                return current
            backend.selected_forward = observe
            try:
                with LLaDAAttentionBackend(model, 'flash') as engine:
                    teacher, actions = backend.generate_active_prefix(
                        model, torch.tensor([ids], device=model.device), gen_length=256,
                        keep=1., pruning=False, trace=True)
            finally:
                backend.selected_forward = original
            assert engine.report()['torch_sdpa_calls'] == 0
            assert len(result['records']) - before + 8 == teacher.nfe == len(actions)
            previous = next(r for r in prior_runs if r['task'] == task and r['id'] == sid
                            and r['length'] == 256 and r['method'] == 'v1')
            tokens = teacher.output[0, len(ids):].tolist()
            assert teacher.nfe == previous['nfe']
            assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest() == previous['token_sha256']
            result['prompts'].append(dict(task=task, id=sid, nfe=teacher.nfe,
                token_sha256=previous['token_sha256'], native_token_nfe_parity=True, backend=engine.report()))
            result['summary'] = summarize(result['records'], methods)
            write_json(args.output / 'diagnostic.json', result)
            print('Finished shadow prompt', task, sid, json.dumps(result['summary']), flush=True)
    for name, digest in result['implementation'].items():
        assert sha256(Path(__file__).parent / name) == digest, 'Running source changed'
    result['by_task'] = {t: summarize([r for r in result['records'] if r['task'] == t], methods) for t in datasets}
    write_json(args.output / 'diagnostic.json', result)
    (args.output / 'complete').write_text('OK\n')
    print('FOCUS-v2 mechanism diagnostic complete', json.dumps(result['summary']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mass-implementation', choices=['feature', 'repeat'], default='feature')
    for task in ('humaneval', 'mbpp', 'math'):
        parser.add_argument('--' + task + '-dataset', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
