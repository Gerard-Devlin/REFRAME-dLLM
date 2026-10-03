"""Offline verification-cost preflight; reference future states are NOT a drafter.

Do not deploy while an existing tuning queue fingerprints this source directory.
No speculative executor, sampler change, or end-to-end speed claim is made here.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F


def validate_window(states, block, mask_id):
    if not states or block <= 0:
        raise ValueError('Nonempty ordinary-state window required')
    first = states[0]
    if first.ndim != 2 or first.shape[0] != 1 or first.shape[1] < block:
        raise ValueError('Expected single-request suffix states')
    previous = None
    for state in states:
        if state.shape != first.shape or state.dtype != torch.long:
            raise ValueError('Mixed state shapes or dtypes')
        if not (state[0, :block] == mask_id).any():
            raise ValueError('Completed blocks are not ordinary denoising states')
        if not torch.equal(state[:, block:], first[:, block:]):
            raise ValueError('Window crosses a block or changes the future canvas')
        if previous is not None:
            fixed = previous != mask_id
            if not torch.equal(state[fixed], previous[fixed]):
                raise ValueError('Previously revealed tokens changed')
            if (state == mask_id).sum() >= (previous == mask_id).sum():
                raise ValueError('Expected a strictly advancing teacher trajectory')
        previous = state


def pack_states(states, past, *, owned=False):
    batch = len(states)
    if batch < 1 or not past:
        raise ValueError('States and formal prefix cache are required')
    prefix = past[0][0].shape[-2]
    packed_past = []
    for pair in past:
        if len(pair) != 2:
            raise ValueError('Malformed K/V pair')
        if any(t.ndim != 4 or t.shape[0] != 1 or t.shape[-2] != prefix for t in pair):
            raise ValueError('Cache version/shape does not describe one fixed prefix')
        # Prefix append is read-only. DualCache instead writes into every active
        # cache row, so its branches must own storage; those copies are timed.
        packed_past.append(tuple(t.repeat(batch,1,1,1) if owned else
                                 t.expand(batch,-1,-1,-1) for t in pair))
    return torch.cat(states, dim=0), packed_past


def decision(logits, state, block, mask_id, threshold):
    target = (state[:block] == mask_id).nonzero().flatten()
    if target.numel() == 0:
        raise ValueError('An ordinary state must contain an active MASK')
    active = logits.index_select(0, target)
    token = active.argmax(-1)
    confidence = F.softmax(active.to(torch.float64), -1).gather(1, token[:, None])[:, 0]
    selected = confidence >= threshold
    selected[confidence.argmax()] = True
    positions, values = target[selected], token[selected]
    next_state = state.clone()
    next_state[positions] = values
    return positions, values, confidence, next_state


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2))
    temp.replace(path)


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    binding = check_binding(required=True)
    from .run import load_model
    from .competitors import select_samples, generation_prompt
    from .padded_head import selected_forward
    from ..llada_backend import LLaDAAttentionBackend
    from ..llada_common import MODEL_ID, REVISION, MASK_ID, prompt_ids
    from ..llada_decode import generate_prefix_cache, generate_dual_cache
    from .native_row_ops import NativeRowOps
    from .reduction_policy import BF16Reduction

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--cache-mode', choices=('prefix','dual'), default='prefix')
    row_parts={'native_rows':('linear','normalization','attention'),
               'linear_rows':('linear',),'attention_rows':('attention',),'norm_rows':('normalization',)}
    parser.add_argument('--operator-mode', choices=('batched',*row_parts), default='batched')
    parser.add_argument('--bf16-reduction',choices=('default','disable_reduced'),default='default')
    parser.add_argument('--workload-indices', type=int, nargs='+', default=None,
        help='Explicit subset of the six predeclared workloads; unavailable states are never replaced')
    args = parser.parse_args()
    if args.repeats < 5:
        raise ValueError('At least five timed repetitions required')
    workload_indices=list(range(6)) if args.workload_indices is None else args.workload_indices
    if len(set(workload_indices))!=len(workload_indices) or any(i not in range(6) for i in workload_indices):
        raise ValueError('Unique predeclared workload indices 0 through 5 required')
    args.output.mkdir(exist_ok=False)
    implementation = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    report = dict(model=MODEL_ID, revision=REVISION, gpu_binding=binding,cache_mode=args.cache_mode,
        operator_mode=args.operator_mode,bf16_reduction=args.bf16_reduction,
        implementation=implementation, records=[], dataset_sha256=[digest(p) for p in args.datasets],
        configuration=dict(repeats=args.repeats, seed=1234, sample_seed=51713, offset=0,
            lengths=[256, 512], block=32, threshold=.90, temperature=0, widths=[1, 2, 4],
            tasks=['humaneval','mbpp','math'], block_indices=[0,2,4], use_cache=True),
        scope='Predeclared family of six reused development workloads; the explicit configured subset is executed. Offline oracle inputs only; '
              'no online drafter, no task score, no measured end-to-end acceleration. '
              'Zero-draft ceilings omit drafting/acceptance losses and are optimistic.',
        measurements='Wall time includes packing, read-only prefix views or owned Dual cache copies, full model, '
                     'head, native decisions, and complete-state edge comparisons. '
                     'Serial comparators exclude verification-only edge comparisons. '
                     'State-copy overhead remains explicit in this fixed-state preflight. '
                     'Selected-head engineering and batch amortization are separate.')
    report['configuration']['workload_indices']=workload_indices
    save(args.output / 'diagnostic.json', report)
    model, tokenizer = load_model('cuda:0')
    native = generate_dual_cache if args.cache_mode=='dual' else generate_prefix_cache
    torch.manual_seed(1234)
    for task_index,(task, dataset, block_index) in enumerate(zip(('humaneval', 'mbpp', 'math'), args.datasets, (0, 2, 4))):
        if not any(i//2==task_index for i in workload_indices):
            continue
        sample = select_samples(dataset, 1, 0)[0]
        ids = prompt_ids(tokenizer, generation_prompt(sample), task, preformatted=True)
        prompt = torch.tensor([ids], device=model.device)
        for length in (256, 512):
            if task_index*2+(length==512) not in workload_indices:
                continue
            wanted_prefix = len(ids) + block_index * 32
            captured, holder, pending = [], {}, {}
            with BF16Reduction(args.bf16_reduction=='disable_reduced') as reduction, LLaDAAttentionBackend(model, 'flash') as backend, (
                NativeRowOps(model, row_parts[args.operator_mode]) if args.operator_mode in row_parts else nullcontext()
            ) as row_ops:
                clean = native(model, prompt, gen_length=length)

                def before(_module, call_args, kwargs):
                    past = kwargs.get('past_key_values')
                    if past is None or len(captured) >= 4:
                        return
                    replace = kwargs.get('replace_position')
                    if args.cache_mode=='dual':
                        if replace is None:
                            return
                        indices=replace[0].nonzero().flatten()
                        if indices.numel()!=32 or int(indices[0])!=wanted_prefix or int(indices[-1])!=wanted_prefix+31:
                            return
                    elif replace is not None or past[0][0].shape[-2]!=wanted_prefix:
                        return
                    state = call_args[0] if call_args else kwargs['input_ids']
                    if not holder:
                        holder['past'] = past
                        holder['pointers'] = [t.data_ptr() for pair in past for t in pair]
                        if replace is not None:
                            holder['replace']=replace.detach().clone()
                    if holder['pointers'] != [t.data_ptr() for pair in past for t in pair]:
                        raise AssertionError('Formal prefix changed inside the window')
                    pending['state'] = state.detach().clone()

                def after(_module, _call_args, _kwargs, output):
                    if pending:
                        active_cache = ([tuple(value[:,:,wanted_prefix:wanted_prefix+32].detach().clone()
                            for value in pair) for pair in output.past_key_values]
                            if args.cache_mode=='dual' else None)
                        captured.append((pending.pop('state'), output.logits[:, :32].detach().clone(),active_cache))

                handles = [model.register_forward_pre_hook(before, with_kwargs=True),
                           model.register_forward_hook(after, with_kwargs=True)]
                try:
                    observed = native(model, prompt, gen_length=length)
                finally:
                    for handle in handles:
                        handle.remove()
                assert torch.equal(clean.output, observed.output) and clean.nfe == observed.nfe
                record = dict(task=task, id=sample.get('id', sample.get('task_id')), length=length,
                    block_index=block_index, prefix_length=wanted_prefix, captured=len(captured),
                    teacher_tokens_match=True, teacher_nfe=clean.nfe, teacher_seconds=clean.seconds,
                    teacher_output_sha256=hashlib.sha256(json.dumps(clean.output[0,len(ids):].tolist()).encode()).hexdigest(),
                    reduction_policy=reduction.audit,variants=[])
                report['records'].append(record)
                if not captured:
                    record['status'] = 'No ordinary states in the predeclared block; not replaced'
                    record['backend']=backend.report()
                    record['row_operator_stats']=row_ops.stats.copy() if row_ops else None
                    assert record['backend']['torch_sdpa_calls']==0
                    save(args.output / 'diagnostic.json', report)
                    print(json.dumps({k:record[k] for k in ('task','length','captured','status')}),flush=True)
                    del clean, observed, captured, holder, pending
                    torch.cuda.empty_cache()
                    continue
                states = [pair[0] for pair in captured]
                validate_window(states, 32, MASK_ID)
                past = holder['past']
                originals = [t.detach().cpu().clone() for pair in past for t in pair]
                versions = [t._version for pair in past for t in pair]
                # Native Dual serial calls reuse one private mutable cache; they
                # do not pay branch duplication every ordinary step. Never use
                # the authoritative teacher cache as this scratch storage.
                serial_cache = ([tuple(value.clone() for value in pair) for pair in past]
                                if args.cache_mode=='dual' else past)

                def verify(width, batched, compact):
                    rows = states[:width]
                    if batched:
                        packed, cache = pack_states(rows, past,owned=args.cache_mode=='dual')
                        kwargs=dict(past_key_values=cache,use_cache=True)
                        if args.cache_mode=='dual':
                            kwargs['replace_position']=holder['replace'].expand(width,-1)
                        result = (selected_forward(model,packed,range(32),**kwargs)
                                  if compact else model(packed,**kwargs))
                        logits = result.logits[:, :32]
                    else:
                        outputs = []
                        kwargs=dict(past_key_values=serial_cache,use_cache=True)
                        if args.cache_mode=='dual':
                            kwargs['replace_position']=holder['replace']
                        for row in rows:
                            result = (selected_forward(model,row,range(32),**kwargs)
                                if compact else model(row,**kwargs))
                            outputs.append(result.logits[:, :32])
                        logits = torch.cat(outputs, dim=0)
                    actions = [decision(logits[i], rows[i][0], 32, MASK_ID, .90) for i in range(width)]
                    edges = ([torch.equal(actions[i][3], rows[i+1][0]) for i in range(width-1)]
                             if batched else [])
                    return logits, actions, edges, cache if batched else serial_cache

                for width in (1, 2, 4):
                    if width > len(states):
                        continue
                    for compact in (False, True):
                        for batched in (False, True):
                            verify(width, batched, compact)  # actual-workload warm-up, excluded
                            elapsed = []
                            torch.cuda.reset_peak_memory_stats()
                            for _ in range(args.repeats):
                                # Do not retain a previous batch's branch caches
                                # or logits while allocating the next sample.
                                output=actions=edges=used_cache=None
                                torch.cuda.synchronize()
                                started = time.perf_counter()
                                output, actions, edges, used_cache = verify(width, batched, compact)
                                torch.cuda.synchronize()
                                elapsed.append(time.perf_counter() - started)
                            reference = torch.cat([pair[1] for pair in captured[:width]], 0)
                            if not batched:
                                # Numeric diagnostics are outside the native serial timing.
                                edges = [torch.equal(actions[i][3], states[i+1][0]) for i in range(width-1)]
                            errors, action_match, confidence_errors = [], [], []
                            for i in range(width):
                                mask = states[i][0, :32] == MASK_ID
                                errors.append(float((output[i, mask] - reference[i, mask]).abs().max()))
                                ref = decision(reference[i], states[i][0], 32, MASK_ID, .90)
                                action_match.append(torch.equal(actions[i][0], ref[0]) and torch.equal(actions[i][1], ref[1]))
                                confidence_errors.append(float((actions[i][2]-ref[2]).abs().max()))
                            record['variants'].append(dict(width=width, batched=batched, compact_head=compact,
                                seconds=elapsed, median_seconds=sorted(elapsed)[len(elapsed)//2],
                                max_logit_error=max(errors), max_confidence_error=max(confidence_errors),
                                action_match=action_match, full_state_edges=edges,
                                peak_gib=torch.cuda.max_memory_allocated()/2**30,
                                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
                            if args.cache_mode=='dual':
                                indices=range(width) if batched else (width-1,)
                                cache_matches=[]
                                outside=~holder['replace'][0]
                                for index in indices:
                                    cache_row=index if batched else 0
                                    cache_matches.append(all(torch.equal(value[cache_row:cache_row+1,:,wanted_prefix:wanted_prefix+32],
                                        captured[index][2][layer][kind]) for layer,pair in enumerate(used_cache)
                                        for kind,value in enumerate(pair)))
                                assert all(torch.equal(value[:,:,outside],past[layer][kind][:,:,outside].expand_as(value[:,:,outside]))
                                    for layer,pair in enumerate(used_cache) for kind,value in enumerate(pair))
                                record['variants'][-1]['active_cache_reference_matches']=cache_matches
                                record['variants'][-1]['outside_block_cache_unchanged']=True
                            assert versions == [t._version for pair in past for t in pair]
                            save(args.output / 'diagnostic.json', report)
                record['cost_ratios'] = []
                for width in (2, 4):
                    for compact in (False, True):
                        variants = {v['batched']:v for v in record['variants']
                                    if v['width']==width and v['compact_head']==compact}
                        if len(variants)!=2:
                            continue
                        serial, parallel = variants[False], variants[True]
                        record['cost_ratios'].append(dict(width=width, compact_head=compact,
                            zero_draft_fixed_state_ratio=serial['median_seconds']/parallel['median_seconds'],
                            serial_actions_equal_in_this_window=all(serial['action_match']),
                            serial_state_edges_equal_in_this_window=all(serial['full_state_edges']),
                            actions_equal_in_this_window=all(parallel['action_match']),
                            state_edges_equal_in_this_window=all(parallel['full_state_edges']),
                            scope='Optimistic fixed-state hardware screen, not a global speed bound or online result'))
                assert all(torch.equal(original, value.detach().cpu()) for original, value in
                    zip(originals, [t for pair in past for t in pair]))
                record['formal_prefix_unchanged'] = True
                record['backend'] = backend.report()
                record['row_operator_stats']=row_ops.stats.copy() if row_ops else None
                assert record['backend']['torch_sdpa_calls'] == 0
                record['status'] = 'Completed offline cost/numeric preflight'
                save(args.output / 'diagnostic.json', report)
                print(json.dumps({k:record[k] for k in ('task','length','captured','status','cost_ratios')}),flush=True)
            del captured, holder, pending, states, past, originals, serial_cache, clean, observed
            del output, actions, edges, used_cache, reference, ref
            torch.cuda.empty_cache()
    assert implementation == {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    save(args.output / 'diagnostic.json', report)
    (args.output / 'complete').write_text('OK\n')


if __name__ == '__main__':
    main()
