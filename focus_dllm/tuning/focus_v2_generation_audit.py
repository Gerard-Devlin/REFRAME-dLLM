"""Two reused development prompts: decoder equivalence and trace noninterference."""
import argparse
from pathlib import Path

import torch

from ..llada_backend import LLaDAAttentionBackend
from ..llada_common import prompt_ids, MODEL_ID, REVISION
from .. import llada_decode
from . import focus_v2_decode
from .focus_v2 import ProxyConfig
from .competitors import select_samples, generation_prompt, write, digest
from .run import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not (args.output/'complete').exists(), 'Audit already completed'
    model, tokenizer = load_model('cuda:0')
    report = dict(model=MODEL_ID, revision=REVISION, records=[],
        dataset_sha256=digest(args.dataset), implementation={p.name:digest(p)
        for p in Path(__file__).parent.glob('*.py')},
        scope='Two reused HumanEval development prompts; instrumented timings are not baseline speed')
    old_native, old_active = llada_decode._selected_positions, focus_v2_decode._selected_positions
    def instrument(old, sink):
        def observed(confidence, threshold):
            selected = old(confidence,threshold)
            sink.append(dict(confidence=confidence.cpu(),selected=selected.cpu()))
            return selected
        return observed
    for sample in select_samples(args.dataset,2,0):
        ids = prompt_ids(tokenizer,generation_prompt(sample),'humaneval',preformatted=True)
        prompt = torch.tensor([ids],device=model.device)
        native_actions, active_actions = [], []
        llada_decode._selected_positions = instrument(old_native,native_actions)
        focus_v2_decode._selected_positions = instrument(old_active,active_actions)
        try:
            with LLaDAAttentionBackend(model,'flash') as engine, torch.no_grad():
                native = llada_decode.generate_prefix_cache(model,prompt,gen_length=128)
                active, info = focus_v2_decode.generate(model,prompt,gen_length=128,trace=True)
        finally:
            llada_decode._selected_positions,focus_v2_decode._selected_positions = old_native,old_active
        assert native.nfe == active.nfe == len(native_actions) == len(active_actions) == len(info['actions'])
        assert torch.equal(native.output,active.output), 'Active head changed generation tokens'
        errors = [float((a['confidence']-b['confidence']).abs().max())
                  for a,b in zip(native_actions,active_actions)]
        assert all(torch.equal(a['selected'],b['selected']) for a,b in zip(native_actions,active_actions)), 'Native action changed'
        assert max(errors) == 0., 'Shared head control changed confidence'
        config = ProxyConfig(mass_implementation='repeat')
        with LLaDAAttentionBackend(model,'flash') as pool_engine, torch.no_grad():
            clean, clean_info = focus_v2_decode.generate(model,prompt,gen_length=128,config=config)
            traced, trace_info = focus_v2_decode.generate(model,prompt,gen_length=128,config=config,trace=True)
        assert clean.nfe == traced.nfe == len(trace_info['actions'])
        assert torch.equal(clean.output,traced.output), 'Tracing changed FOCUS-v2 generation'
        assert clean_info['pooling_calls'] == trace_info['pooling_calls']
        for backend,nfe in ((engine,native.nfe+active.nfe),(pool_engine,clean.nfe+traced.nfe)):
            counters = backend.report()
            assert counters['torch_sdpa_calls'] == 0 and counters['flash_calls'] == nfe*32
        report['records'].append(dict(id=sample.get('id',sample.get('task_id')),
            native_active_max_confidence_error=max(errors),native_action_exact=True,
            native_tokens_exact=True,native_nfe=native.nfe,pool_trace_exact=True,
            focus_v2_nfe=clean.nfe,nfe_by_block=clean_info['nfe_by_block'],
            native_backend=engine.report(),pool_backend=pool_engine.report()))
        write(args.output/'diagnostic.json',report)
        print('Verified generation '+str(report['records'][-1]),flush=True)
    assert report['implementation'] == {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    (args.output/'complete').write_text('OK\n')


if __name__ == '__main__':
    main()
