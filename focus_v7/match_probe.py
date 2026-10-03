"""Cache-producing prefix packet: bounded read-only cost/progress diagnostic."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from focus_v5.relay_joint_probe import TimedModel
from focus_v6.audit_probe import forbid_sdpa
from focus_dllm.tuning.flash_readout import ModelReadout
from .cache import capture_private_kv, promote
from .observe import instrument
from .packet import build_call, promotion_plan
from .greedy import decide


def summarize(records):
    result = {}
    for k in (16,):
        rows = [r for r in records if r['k'] == k]
        count = sum(len(r['tokens']) for r in rows)
        matched = sum(sum(r['teacher_final_matches']) for r in rows)
        task_rates, agreements = {}, {}
        def rate(group):
            ours = sum(len(r['tokens']) for r in group) / sum(r['packet_wall_ms'] for r in group)
            base = sum(r['safe_progress']+r['official_accepted'] for r in group)
            base /= sum(r['regular_ms']+r['verify_ms'] for r in group)
            return ours/base
        for task in ('humaneval', 'mbpp', 'math'):
            group = [r for r in rows if r['task'] == task]
            produced = sum(len(r['tokens']) for r in group)
            agreements[task] = sum(sum(r['teacher_final_matches']) for r in group)/produced if produced else None
            task_rates[task] = rate(group) if produced else 0.
        ratio = rate(rows) if count else 0.
        agreement = matched/count if count else None
        result[str(k)] = dict(windows=len(rows), produced_tokens=count, teacher_final_matches=matched,
            teacher_final_agreement=agreement, task_agreement=agreements, task_rate_ratio=task_rates,
            fixed_state_progress_cost_ratio=ratio,
            median_packet_wall_ms=statistics.median(r['packet_wall_ms'] for r in rows) if rows else None,
            median_packet_gpu_ms=statistics.median(r['packet_gpu_ms'] for r in rows) if rows else None,
            median_prefix=statistics.median(r['accepted'] for r in rows) if rows else None,
            gate_passed=count>=48 and agreement is not None and agreement>=.98
                and all(v is not None and v>=.95 for v in agreements.values())
                and ratio>=1.3 and all(v>=1.0 for v in task_rates.values()),
            scope='Necessary fixed-state screen, common compact head, shadow wall cost versus baseline GPU cost; not achieved speed or task accuracy')
    return result


@torch.no_grad()
def main():
    from focus_dllm.common import sha256, write_json
    from focus_dllm.llada_common import MODEL_ID, REVISION, prompt_ids
    from focus_dllm.tuning.competitors import generation_prompt, load_external, load_model, select_samples
    from focus_dllm.tuning.flash_readout import suppress_official_prints
    from focus_dllm.tuning.gpu_contract import check_binding

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--third-party', type=Path, required=True)
    parser.add_argument('--datasets', type=Path, nargs=3, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    cls, external, adaptation = load_external(args.third_party, 'flash_verify')
    official_source = Path(inspect.getsourcefile(inspect.unwrap(external)))
    binding = check_binding(required=True)
    raw, tokenizer = load_model(SimpleNamespace(method='flash_verify'), cls)
    compact = ModelReadout(raw, compact=True, minimum=32)
    model = TimedModel(compact)
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    forbidden = set(tokenizer.all_special_ids) | {126336, 126081}
    reference = {(r['task'], str(r['id'])): r for r in json.loads(args.reference.read_text())['prompts']}
    report = dict(model=MODEL_ID, revision=REVISION, binding=binding,
        official_source_sha256=sha256(official_source), adaptation=adaptation,
        implementation={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        datasets={str(p):sha256(p) for p in args.datasets},
        configuration=dict(length=256, threshold=.9, gamma=.8, k=[16], windows_per_prompt=2,
                           prompts_per_task=2, seed=51713, offset=0, repeats=3,
                           common_engineering='compact projection, minimum32; original probability reduction'),
        admission_policy='argmax-match, no probability budget; distinct approximate conditional program', records=[], prompts=[], private_controls=[], scope='Paid shadows only; no gold input or online packet commits')
    write_json(args.output/'diagnostic.json', report)
    windows, controlled = [], set()
    cycle, pending, final = [0], [None], [None]

    def packet_run(query, pos, lengths, layout, drafts):
        with capture_private_kv() as kv:
            output = compact(query, use_cache=True, positions=pos, lengths=lengths,
                             focus_head_rows=(layout.clean.start, 3*layout.candidates))
        logits = output.logits.squeeze(0)[layout.audit]
        probability = logits.double().softmax(-1).gather(1, drafts[:,None]).squeeze(1)
        if len(kv) != len(raw.model.transformer.blocks):
            raise AssertionError('missing real native projection buffers')
        return logits, probability, kv

    def before_verify(proxy, state):
        current = cycle[0]; cycle[0] += 1
        if len(windows)>=2 or int(state['num_verify'])<16:
            return
        decoded = int(state['num_decoded'][0])
        positions = state['full_pos'][0,decoded:decoded+16]
        if any(int(t) in forbidden for t in state['x_draft'][positions].tolist()):
            return
        if pending[0] is not None:
            raise AssertionError('unfinished official observer')
        blocks = raw.model.transformer.blocks
        versions = [(b.k_cache._version,b.v_cache._version) for b in blocks]
        public_checks = [(b.k_cache.clone(),b.v_cache.clone()) for b in (blocks[0],blocks[-1])]
        # Persistent private-bank allocation is outside packet timing. A real
        # decoder would own one bank; this diagnostic duplicates it for safety.
        bank = [(b.k_cache.clone(),b.v_cache.clone()) for b in blocks]
        window = {**model.context, 'cycle':current, 'safe_progress':int(state['num_newly_decoded'][0]), 'packets':[]}
        for k in (16,):
            query, pos, lengths, layout, candidates, drafts = build_call(state,k)
            packet_run(query,pos,lengths,layout,drafts) # compile/warm separately
            torch.cuda.synchronize()
            costs, gpu_costs = [], []
            decision, kv, logits = None, None, None
            for _ in range(3):
                torch.cuda.synchronize()
                wall = time.perf_counter()
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                query,pos,lengths,layout,candidates,drafts = build_call(state,k)
                logits,probability,kv = packet_run(query,pos,lengths,layout,drafts)
                decision = decide(probability.tolist(),logits.argmax(-1).tolist(),drafts.tolist(),forbidden=forbidden)
                tracked = pos[0][:layout.tracked]
                rows,dest,dirty = promotion_plan(layout,candidates.tolist(),tracked.tolist(),decision)
                rows_t = torch.tensor(rows,device=query.device,dtype=torch.long)
                dest_t = torch.tensor(dest,device=query.device,dtype=torch.long)
                promote(bank,kv,rows_t,dest_t)
                end.record(); end.synchronize()
                costs.append((time.perf_counter()-wall)*1000)
                gpu_costs.append(start.elapsed_time(end))
            row = {**model.context,'cycle':current,'k':k,'positions':candidates.tolist(),
                'drafts':drafts.tolist(),'probabilities':probability.tolist(),'top1':logits.argmax(-1).tolist(),
                'accepted':decision.accepted,'tokens':list(decision.tokens),'correction':decision.correction,
                'dirty_positions':list(dirty),'promoted_rows':len(rows),'packet_wall_ms':statistics.median(costs),
                'packet_gpu_ms':statistics.median(gpu_costs),'repeats_wall_ms':costs,
                'safe_progress':window['safe_progress'],'public_cache_unchanged':True}
            if k==16 and model.context['task'] not in controlled:
                controls = []
                for index in (0,15):
                    perturbed = query.clone()
                    old = int(perturbed[0,layout.draft.start+index])
                    perturbed[0,layout.draft.start+index] = 15 if old!=15 else 16
                    changed,_,changed_kv = packet_run(perturbed,pos,lengths,layout,drafts)
                    audit_error = float((changed[:index+1].float()-logits[:index+1].float()).abs().max())
                    legal_rows = list(range(layout.clean.stop))+list(range(layout.draft.start,layout.draft.start+index))
                    legal_rows = torch.tensor(legal_rows,device=query.device)
                    cache_error = max(float((a.index_select(0,legal_rows).float()-b.index_select(0,legal_rows).float()).abs().max())
                        for old_layer,new_layer in zip(kv,changed_kv) for a,b in zip(old_layer,new_layer))
                    if audit_error!=0 or cache_error!=0:
                        raise AssertionError(f'own/later identity leaked: audit{audit_error}, KV{cache_error}')
                    controls.append(dict(changed_draft=index,audits_through=index,logit_max_error=audit_error,
                                         all_layer_legal_source_kv_max_error=cache_error))
                report['private_controls'].append({**model.context,'cycle':current,'controls':controls})
                controlled.add(model.context['task'])
            window['packets'].append(row)
        if versions != [(b.k_cache._version,b.v_cache._version) for b in blocks]:
            raise AssertionError('public cache mutation')
        for saved,block in zip(public_checks,(blocks[0],blocks[-1])):
            if any(not torch.allclose(a,b,rtol=0,atol=0,equal_nan=True) for a,b in zip(saved,(block.k_cache,block.v_cache))):
                raise AssertionError('public cache contents changed')
        windows.append(window); pending[0] = window

    def official_accept(probability,drafts,gamma):
        if pending[0] is not None:
            pending[0]['official_accepted'] = int((probability.double().cumprod(0)>=gamma).sum())
            pending[0] = None

    def final_canvas(canvas,prompt_length,max_length,gen_length):
        if final[0] is not None or gen_length!=256:
            raise AssertionError('unexpected final canvas')
        final[0] = canvas[:max_length].detach().cpu().tolist()

    traced = instrument(external,before_verify,official_accept,final_canvas)
    for task,path in zip(('humaneval','mbpp','math'),args.datasets):
        for sample in select_samples(path,2,0):
            ident = str(sample.get('id',sample.get('task_id')))
            model.context = dict(task=task,id=ident)
            windows.clear(); cycle[0]=0; final[0]=None
            event_start = len(model.events)
            ids = prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
            response,steps = [None],[0]
            with forbid_sdpa(),suppress_official_prints():
                traced(model,[torch.tensor(ids,device=raw.device)],[len(ids)],1,response,steps,
                       gen_length=256,block_length=32,threshold=.9,gamma=.8,track_num=4,mask_num=4,
                       verify=True,tokenizer=tokenizer,stop_tokens=[])
            torch.cuda.synchronize()
            events = model.events[event_start:]
            regular = [e for e in events if e['kind']=='regular']
            verify = [e for e in events if e['kind']=='verify']
            if pending[0] is not None or final[0] is None or not len(regular)==len(verify)==cycle[0]==steps[0]:
                raise AssertionError('incomplete official trajectory')
            text_sha = hashlib.sha256(response[0].encode()).hexdigest()
            if text_sha!=reference[(task,ident)]['text_sha256']:
                raise AssertionError('common readout or observer changed official output')
            for window in windows:
                reg = regular[window['cycle']]
                same = [e for e in regular[3:] if e['rows']==reg['rows']] or regular[3:] or regular
                stable_ver = verify[3:] or verify
                for row in window['packets']:
                    row['regular_ms'] = statistics.median(e['start'].elapsed_time(e['end']) for e in same)
                    row['verify_ms'] = statistics.median(e['start'].elapsed_time(e['end']) for e in stable_ver)
                    row['official_accepted'] = window['official_accepted']
                    row['teacher_final_matches'] = [final[0][p]==v for p,v in zip(row['positions'],row['tokens'])]
                    report['records'].append(row)
            report['prompts'].append(dict(task=task,id=ident,windows=len(windows),official_cycles=steps[0],
                text_sha256=text_sha,official_output_unchanged=True,sdpa_calls=0))
            write_json(args.output/'diagnostic.json',report)
            print('PROMPT',task,ident,'windows',len(windows),flush=True)
    report['summary'] = summarize(report['records'])
    report['gate_passed'] = any(r['gate_passed'] for r in report['summary'].values())
    if sha256(official_source)!=report['official_source_sha256']:
        raise AssertionError('official source changed')
    if report['implementation']!={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}:
        raise AssertionError('research source changed')
    write_json(args.output/'diagnostic.json',report)
    (args.output/'complete').write_text('OK\n')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__ == '__main__':
    main()

