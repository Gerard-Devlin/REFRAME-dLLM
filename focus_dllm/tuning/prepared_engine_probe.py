"""Verify/measure a common rotary execution optimization on reused dev states."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from ..common import sha256, write_json
from ..llada_backend import LLaDAAttentionBackend
from ..llada_common import MODEL_ID, REVISION, prompt_ids
from . import backend
from .focus_v2 import FocusV2Forward, ProxyConfig
from .focus_v2_probe import prediction
from .prepared_rotary import RotatedPrefixCache
from .retention_sweep import sample_slice
from .run import load_model


def measure(functions, seed):
    rng, names = random.Random(seed), list(functions)
    values = {name: [] for name in names}
    for number in range(8):
        rng.shuffle(names)
        for name in names:
            torch.cuda.synchronize()
            started = time.perf_counter()
            functions[name]()
            torch.cuda.synchronize()
            if number >= 3:
                values[name].append(time.perf_counter() - started)
    return {name:dict(seconds=times, median_seconds=sorted(times)[2]) for name,times in values.items()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    for task in ('humaneval', 'mbpp', 'math'):
        parser.add_argument('--' + task + '-dataset', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Keep previous diagnostic attempts'
    args.output.mkdir(parents=True)
    old = json.loads((args.reference_root/'focus_v2_repeat_probe/diagnostic.json').read_text())
    reference = {(r['task'],str(r['id']),r['block'],r['refine']):r for r in old['records']}
    trajectories = json.loads((args.reference_root/'termination_probe.json').read_text())['records']
    model, tokenizer = load_model('cuda:0')
    torch.set_num_threads(1)
    original = backend.selected_forward
    cache = RotatedPrefixCache()
    config = ProxyConfig(mass_implementation='repeat')
    plain = FocusV2Forward(model, config)
    prepared = FocusV2Forward(model, config, rotary_factory=cache)
    result = dict(model=MODEL_ID, revision=REVISION, length=256, threshold=.90,
        records=[], timings=[], cache_build_seconds=[], prompts=[],
        implementation={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        scope='Shared ENGINE control for native and focus-v2. Six reused development prompts. '
              'Require exact logits/actions/canvas and prefix immutability. Warm-cache per-forward '
              'cost plus separate cache-build cost, not a generation speedup or method gain.')
    for task in ('humaneval', 'mbpp', 'math'):
        samples = sample_slice(json.loads(getattr(args,task+'_dataset').read_text()),0,2)
        for sample in samples:
            ident = sample.get('id',sample.get('task_id'))
            ids = prompt_ids(tokenizer,sample.get('paper_prompt',sample.get('prompt')),task,preformatted=True)
            cache.clear()
            memo = dict(refine=0, cold=True)
            before = len(result['records'])
            def native_prepared(canvas, targets, past, cold=False):
                started = time.perf_counter()
                with cache.suffix_scope(model,past,canvas.shape[1]):
                    if cold:
                        torch.cuda.synchronize()
                        result['cache_build_seconds'].append(dict(task=task,id=ident,
                            block=(256-canvas.shape[1])//32, seconds=time.perf_counter()-started))
                    return original(model,canvas,targets,past_key_values=past,use_cache=False).logits
            def proxy_prepared(canvas, targets, past):
                with cache.suffix_scope(model,past,canvas.shape[1]):
                    return prepared(canvas,targets.tolist(),past_key_values=past)
            def observe(current_model,canvas,targets,**kwargs):
                current = original(current_model,canvas,targets,**kwargs)
                if kwargs.get('use_cache') and kwargs.get('past_key_values') is None:
                    memo.update(refine=0,cold=True)
                    return current
                memo['refine'] += 1
                key = (task,str(ident),(256-canvas.shape[1])//32,memo['refine'])
                past = kwargs['past_key_values']
                versions = [t._version for pair in past for t in pair]
                canvas_version = canvas._version
                top,p,take = prediction(current.logits)
                raw = plain(canvas,targets.tolist(),past_key_values=past)
                other_native = native_prepared(canvas,targets,past,memo['cold'])
                memo['cold'] = False
                other_proxy = proxy_prepared(canvas,targets,past)
                native_error = float((other_native.float()-current.logits.float()).abs().max())
                proxy_error = float((other_proxy.float()-raw.float()).abs().max())
                assert native_error == proxy_error == 0., 'Shared engine changed numerical results'
                _,confidence,selected = prediction(raw)
                previous = reference[key]['methods']['focus_v2']
                assert confidence[0].tolist() == previous['confidence'], 'Unoptimized FOCUS-v2 changed'
                assert targets[selected].tolist() == previous['selected']
                result['records'].append(dict(task=task,id=ident,block=key[2],refine=key[3],
                    native_max_error=native_error,proxy_max_error=proxy_error,
                    native_action_exact=True,proxy_action_exact=True))
                del raw, other_native, other_proxy
                if key[2] in (0,3,5) and key[3] == 1:
                    functions = {
                        'native_original':lambda:original(model,canvas,targets,past_key_values=past,use_cache=False).logits,
                        'native_prepared':lambda:native_prepared(canvas,targets,past),
                        'focus_v2_original':lambda:plain(canvas,targets.tolist(),past_key_values=past),
                        'focus_v2_prepared':lambda:proxy_prepared(canvas,targets,past),
                    }
                    result['timings'].append(dict(task=task,id=ident,block=key[2],
                        suffix_length=canvas.shape[1],methods=measure(functions,1234+len(result['timings']))))
                assert canvas_version == canvas._version and versions == [t._version for pair in past for t in pair]
                return current
            backend.selected_forward = observe
            try:
                with LLaDAAttentionBackend(model,'flash') as engine:
                    teacher,actions = backend.generate_active_prefix(model,torch.tensor([ids],device=model.device),
                        gen_length=256,keep=1.,pruning=False,trace=True)
            finally:
                backend.selected_forward = original
            assert engine.report()['torch_sdpa_calls'] == 0
            assert len(result['records'])-before+8 == teacher.nfe == len(actions)
            prior = next(r for r in trajectories if r['task']==task and r['id']==ident
                         and r['length']==256 and r['method']=='v1')
            tokens = teacher.output[0,len(ids):].tolist()
            assert teacher.nfe == prior['nfe']
            assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest() == prior['token_sha256']
            result['prompts'].append(dict(task=task,id=ident,nfe=teacher.nfe,backend=engine.report()))
            write_json(args.output/'diagnostic.json',result)
            print('Prepared engine prompt complete',task,ident,'states',len(result['records']),flush=True)
    for name,digest in result['implementation'].items():
        assert sha256(Path(__file__).parent/name) == digest
    write_json(args.output/'diagnostic.json',result)
    (args.output/'complete').write_text('OK\n')
    print('Prepared engine diagnostic complete',len(result['records']),flush=True)


if __name__ == '__main__':
    main()
