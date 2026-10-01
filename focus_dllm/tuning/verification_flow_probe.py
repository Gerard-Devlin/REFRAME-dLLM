"""Read-only pinned Flash rejection telemetry and private mask-view controls.

Six reused development prompts. Neither alternative mask nor perturbed drafts
are committed to the generation state. Logged timings are not speed baselines.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import time

import torch
import torch.nn.functional as F

from .verification_flow import view_mask, proposal_paths, rejection_record, observe_cumulative


def aggregate(records):
    proposals = sum(len(r['probabilities']) for r in records)
    accepted = sum(r['accepted'] for r in records)
    rejected = [r for r in records if r['first_rejected'] is not None]
    return dict(verification_windows=len(records), proposals=proposals, accepted=accepted,
        rejected_windows=len(rejected), budget_only_windows=sum(r['budget_only_rejection'] for r in rejected),
        first_rejected_argmax_mismatch=sum(r['rejected_argmax_mismatch'] for r in rejected),
        discarded_high_conditional_confidence=sum(len(r['discarded_high_confidence_indices']) for r in records),
        local_failure_with_high_tail=sum(not r['budget_only_rejection'] and bool(r['discarded_high_confidence_indices']) for r in rejected),
        scope='Conditional tail observations are neither valid alternate paths nor saved NFE. No new decoder or quality result.')


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    binding = check_binding(required=True)
    from ..common import sha256, write_json
    from ..llada_common import MODEL_ID, REVISION, prompt_ids
    from .competitors import load_external, load_model, generation_prompt, select_samples
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--third-party', type=Path, required=True)
    parser.add_argument('--datasets', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(exist_ok=False)
    cls, external, adaptation = load_external(args.third_party, 'flash_verify')
    official_file = Path(inspect.getsourcefile(inspect.unwrap(external)))
    before_source = sha256(official_file)
    model, tokenizer = load_model(SimpleNamespace(method='flash_verify'), cls)
    torch.set_num_threads(1)
    forbidden = set(tokenizer.all_special_ids) | {126336, 126081}
    replacement = next(i for i in tokenizer.encode('0', add_special_tokens=False) if i not in forbidden)
    data = dict(zip(('humaneval', 'mbpp', 'math'), args.datasets))
    report = dict(model=MODEL_ID, revision=REVISION, gpu_binding=binding,
        datasets={k:sha256(p) for k,p in data.items()}, official_generate_sha256=before_source,
        implementation={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        configuration=dict(length=256,block=32,threshold=.90,gamma=.80,track_num=4,mask_num=4,
            seed=51713,offset=0,prompts_per_task=2,private_controls_per_prompt=2),
        adaptation=adaptation, prompts=[], records=[], controls=[],
        scope='Read-only official Flash trajectory. Alternative verification masks change conditional contexts '
            'and are evaluated privately, never committed. Structural paths are possible dependencies, '
            'not proof of harmful leakage. Self-label independence is not native-decoder equivalence '
            'or task correctness. No gold, future teacher tokens, logits or KV enter the model.',
        backend=dict(engine='Pinned official fused Triton cache/attention',torch_sdpa_calls=0))
    write_json(args.output/'diagnostic.json', report)
    original_sdpa = F.scaled_dot_product_attention
    def watched_sdpa(*inputs, **kwargs):
        report['backend']['torch_sdpa_calls'] += 1
        return original_sdpa(*inputs, **kwargs)
    F.scaled_dot_product_attention = watched_sdpa
    try:
        for task,path in data.items():
            for sample in select_samples(path,2,0):
                ident = sample.get('id',sample.get('task_id'))
                ids = prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                prompt = torch.tensor(ids,device=model.device)
                flags = dict(busy=False,control_count=0,calls=0,shadow_calls=0,pending=None)
                def count(_model,_inputs,_kwargs):
                    flags['shadow_calls' if flags['busy'] else 'calls'] += 1
                count_handle = model.register_forward_pre_hook(count,with_kwargs=True)
                def generate(function):
                    responses, steps = [None],[0]
                    function(model,[prompt],[len(ids)],1,responses,steps,gen_length=256,
                        block_length=32,threshold=.90,gamma=.8,track_num=4,mask_num=4,
                        verify=True,tokenizer=tokenizer,stop_tokens=[])
                    torch.cuda.synchronize()
                    return responses[0],steps[0]
                try:
                    clean_text,clean_steps = generate(external); clean_calls = flags['calls']
                    flags['calls'] = 0
                    def observer(probability,draft,positions,top1,gamma):
                        # Clone/read-only small arrays; no complete vocab/KV persisted.
                        value = rejection_record(probability.detach().cpu().tolist(),draft.detach().cpu().tolist(),
                            positions.detach().cpu().tolist(),top1.detach().cpu().tolist(),gamma,forbidden)
                        value.update(task=task,id=ident,model_call=flags['calls'])
                        report['records'].append(value)
                    observed = observe_cumulative(external,observer)
                    def pre_hook(_model,inputs,kwargs):
                        if flags['busy'] or flags['control_count']>=2 or not kwargs['lengths'][-1]:return
                        query = inputs[0]; positions = kwargs['positions']; lengths = kwargs['lengths']
                        assert query.shape == (1,64) and lengths[1].shape[0] == 1
                        search = int((query[0] == 126336).sum())
                        if search < 2:return
                        tracked = 64-2*search
                        expected = view_mask(32,search).to(model.device)
                        assert torch.equal(expected,positions[-1]),'Unfamiliar official verification topology'
                        token = int(query[0,tracked])
                        if token in forbidden or token==replacement:return
                        saved = [(b.k_cache.clone(),b.v_cache.clone()) for b in model.model.transformer.blocks]
                        changed = query.clone();changed[0,tracked] = replacement
                        isolated_positions = list(positions)
                        isolated_positions[-1] = view_mask(32,search,isolated=True).to(model.device).contiguous()
                        isolated_kwargs = dict(kwargs,positions=isolated_positions)
                        def measured(q,k):
                            torch.cuda.synchronize();start = time.perf_counter()
                            out = model(q,**k).logits[0,tracked+search].float().clone()
                            torch.cuda.synchronize();return out,time.perf_counter()-start
                        flags['busy'] = True
                        try:
                            original_changed,t1 = measured(changed,kwargs)
                            isolated,t2 = measured(query,isolated_kwargs)
                            isolated_changed,t3 = measured(changed,isolated_kwargs)
                            assert all(torch.equal(b.k_cache.view(torch.int16),k.view(torch.int16)) and
                                torch.equal(b.v_cache.view(torch.int16),v.view(torch.int16))
                                for b,(k,v) in zip(model.model.transformer.blocks,saved)),'Private verifier wrote global KV'
                            assert torch.equal(inputs[0],query)
                        finally:flags['busy'] = False
                        del saved
                        flags['control_count'] += 1
                        flags['pending'] = dict(original_changed=original_changed,isolated=isolated,
                            isolated_changed=isolated_changed,task=task,id=ident,search=search,tracked=tracked,
                            old_label=token,perturbed_label=replacement,model_call=flags['calls'],
                            private_seconds=[t1,t2,t3],global_cache_unchanged=True)
                    def post_hook(_model,inputs,kwargs,output):
                        if flags['busy'] or flags['pending'] is None:return
                        value = flags['pending'];flags['pending'] = None
                        original = output.logits[0,value['tracked']+value['search']].float()
                        a,b,c = value.pop('original_changed'),value.pop('isolated'),value.pop('isolated_changed')
                        value.update(original_own_label_max_logit_change=float((original-a).abs().max()),
                            isolated_own_label_max_logit_change=float((b-c).abs().max()),
                            original_top1=int(original.argmax()),original_changed_top1=int(a.argmax()),
                            isolated_top1=int(b.argmax()),isolated_changed_top1=int(c.argmax()),
                            original_probability=float(original.double().softmax(-1)[value['old_label']]),
                            original_changed_probability=float(a.double().softmax(-1)[value['old_label']]),
                            isolated_probability=float(b.double().softmax(-1)[value['old_label']]))
                        report['controls'].append(value)
                    pre = model.register_forward_pre_hook(pre_hook,with_kwargs=True)
                    post = model.register_forward_hook(post_hook,with_kwargs=True)
                    try:seen_text,seen_steps = generate(observed)
                    finally:pre.remove();post.remove()
                    assert clean_text==seen_text and clean_steps==seen_steps and clean_calls==flags['calls']
                    assert flags['pending'] is None and not flags['busy']
                    report['prompts'].append(dict(task=task,id=ident,text_sha256=hashlib.sha256(clean_text.encode()).hexdigest(),
                        clean_model_calls=clean_calls,observed_model_calls=flags['calls'],official_iterations=seen_steps,
                        private_shadow_calls=flags['shadow_calls'],text_and_nfe_match=True))
                    report['summary'] = aggregate(report['records'])
                    write_json(args.output/'diagnostic.json',report)
                    print(json.dumps(dict(prompt=len(report['prompts']),**report['summary'])),flush=True)
                finally:count_handle.remove()
        assert before_source==sha256(official_file) and report['backend']['torch_sdpa_calls']==0
        assert all(c['isolated_own_label_max_logit_change']==0 for c in report['controls'])
        graph = {}
        for search in (2,4,8,16):
            tracked = 64-2*search
            graph[str(search)] = {name:proposal_paths(view_mask(32,search,isolated=isolated),tracked,search,32)[tracked+search:].tolist()
                for name,isolated in (('official',False),('isolated',True))}
        report['structural_proposal_reachability'] = graph
        write_json(args.output/'diagnostic.json',report)
        (args.output/'complete').write_text('OK\n')
    finally:F.scaled_dot_product_attention = original_sdpa


if __name__ == '__main__':main()
