"""Literal cross-step phrase drift diagnostics; no canvas edits or decoder.

Candidates use only two past/current predictions and current probabilities.
The final teacher output is an offline reference, never candidate information.
Matched phrases do not establish semantic correctness or safe early release.
"""
from collections import defaultdict, Counter
import math
import random

CONFIG = dict(length=256,block=32,phrase_length=4,max_shift=2,threshold=.9,
              prompts_per_task=2,sample_seed=51713,offset=0,seed=1234,
              shuffled_controls=8,bootstrap_samples=10000)


def spans(tokens, active, forbidden, length=4):
    if len(tokens)!=len(active) or length<2:
        raise ValueError('Aligned local tokens/MASK status required')
    found=defaultdict(list)
    for start in range(len(tokens)-length+1):
        value=tuple(tokens[start:start+length])
        if all(active[start:start+length]) and not forbidden.intersection(value) and len(set(value))>=2:
            found[value].append(start)
    return found


def candidates(previous, current, forbidden, shifted=True):
    if previous['block']!=current['block'] or previous['canvas']==current['canvas']:
        return []
    length=CONFIG['phrase_length'];distance=CONFIG['max_shift']
    old=spans(previous['draft'],previous['active'],forbidden,length)
    new=spans(current['draft'],current['active'],forbidden,length)
    result=[]
    for phrase,starts in new.items():
        # Unique literal occurrence in each eligible local draft. Never select
        # a convenient match among repeated parentheses/whitespace/common words.
        if len(starts)!=1 or len(old.get(phrase,[]))!=1:continue
        a,b=old[phrase][0],starts[0]
        if abs(a-b)>distance or bool(a!=b)!=shifted:continue
        legal=[j for j in range(max(0,b-distance),min(len(current['draft'])-length,b+distance)+1)
               if all(current['active'][j:j+length])]
        if not legal:continue
        scores=current['placement_scores'][b]
        best=max(legal,key=lambda j:(scores[j-b+distance],-abs(j-b),-j))
        confidence=sum(math.log(max(current['confidence'][b+r],1e-300)) for r in range(length))/length
        result.append(dict(tokens=list(phrase),old_start=a,current_start=b,best_current_start=best,
                           confidence=confidence,shift=b-a,call=current['call'],block=current['block']))
    return result


def disjoint(rows, placement='best_current_start'):
    occupied=set();result=[]
    for row in sorted(rows,key=lambda r:(-r['confidence'],r[placement],r['current_start'])):
        span=set(range(row[placement],row[placement]+CONFIG['phrase_length']))
        if not span.intersection(occupied):
            result.append(row);occupied.update(span)
    return result


def _reference(candidate, trace, final, released):
    length=CONFIG['phrase_length'];block=candidate['block'];base=block*CONFIG['block']
    phrase=candidate['tokens'];cur=candidate['current_start'];call=candidate['call']
    eos=next((i for i,t in enumerate(final) if t==126081),len(final))
    local=final[base:base+CONFIG['block']]
    locations=[j for j in range(max(0,cur-2),min(len(local)-length,cur+2)+1)
               if local[j:j+length]==phrase and base+j+length<=eos]
    best=candidate['best_current_start']
    result=dict(candidate,correct_current=local[cur:cur+length]==phrase and base+cur+length<=eos,
                correct_best_current=local[best:best+length]==phrase and base+best+length<=eos,
                content_present_nearby=bool(locations),unique_final_location=len(locations)==1)
    if len(locations)!=1:return result
    final_start=locations[0];state=next(r for r in trace if r['call']==call)
    if not all(state['active'][final_start:final_start+length]):
        result['final_location_still_all_mask']=False;return result
    result['final_location_still_all_mask']=True
    commit=max(released[base+j] for j in range(final_start,final_start+length))
    # Last uninterrupted exact run at the actual final location, ending at
    # the teacher's complete release. Committed tokens are hard context.
    same=[r for r in trace if r['block']==block and r['call']<=commit]
    stable=commit
    for row in reversed(same):
        if row['draft'][final_start:final_start+length]!=phrase:break
        stable=row['call']
    result.update(final_start=final_start,last_native_release_call=commit,
                  absolute_stable_call=stable,lead_to_absolute_stability=stable-call,
                  lead_to_last_release=commit-call)
    return result


def analyze_prompt(trace, final, forbidden):
    if not trace or [r['call'] for r in trace]!=list(range(len(trace))):
        raise ValueError('Complete consecutive normal teacher calls required')
    released={}
    for row in trace:
        for position in row['commit_positions']:
            if position in released:raise ValueError('Teacher release was not irreversible')
            released[position]=row['call']
    if len(released)!=len(final):raise ValueError('Incomplete teacher release map')
    shifted=[];fixed=[];matched=[];shuffled=[]
    for previous,current in zip(trace,trace[1:]):
        move=disjoint(candidates(previous,current,forbidden,True))
        stay=disjoint(candidates(previous,current,forbidden,False),placement='current_start')
        shifted.extend(_reference(c,trace,final,released) for c in move)
        fixed.extend(_reference(c,trace,final,released) for c in stay)
        count=min(len(move),len(stay))
        # EXACTLY the same revealed-token coverage in each paired real state.
        for a,b in zip(move[:count],stay[:count]):
            aa,bb=_reference(a,trace,final,released),_reference(b,trace,final,released)
            matched.append(dict(call=current['call'],shifted=aa,fixed=bb))
        for control in range(CONFIG['shuffled_controls']):
            fake=dict(previous);draft=list(previous['draft'])
            slots=[i for i,on in enumerate(previous['active']) if on]
            values=[draft[i] for i in slots]
            random.Random(CONFIG['seed']+current['call']*100+control).shuffle(values)
            for i,value in zip(slots,values):draft[i]=value
            fake['draft']=draft
            values=disjoint(candidates(fake,current,forbidden,True))
            shuffled.append(dict(call=current['call'],control=control,
                                 candidates=[_reference(c,trace,final,released) for c in values]))
    def opportunities(rows):
        ready={}
        for row in rows:
            if row.get('unique_final_location') and row.get('final_location_still_all_mask'):
                base=row['block']*CONFIG['block']+row['final_start']
                for p in range(base,base+CONFIG['phrase_length']):ready[p]=min(ready.get(p,10**9),row['call'])
        calls=[r for r in trace if r['commit_positions'] and
               all(ready.get(p,10**9)<r['call'] for p in r['commit_positions'])]
        return dict(ready_positions=len(ready),covered_calls=len(calls),
                    covered_forward_seconds=sum(r['forward_seconds'] for r in calls),
                    covered_call_indices=[r['call'] for r in calls],
                    scope='Final-location teacher oracle, not online information or an achievable NFE bound. '
                          'Changing prior reveals changes later states; no calls have been skipped.')
    move_cost=opportunities(shifted);fixed_cost=opportunities(fixed)
    known={(r['block'],tuple(r['tokens'])) for r in fixed}
    early=[r for r in shifted if r.get('lead_to_absolute_stability',0)>=2]
    eos=next((i for i,t in enumerate(final) if t==126081),len(final))
    useful_positions=set()
    for r in early:
        base=r['block']*CONFIG['block']+r['final_start']
        useful_positions.update(range(base,base+CONFIG['phrase_length']))
    return dict(shifted=shifted,fixed=fixed,matched_coverage=matched,shuffled=shuffled,
        summary=dict(calls=len(trace),shifted_candidates=len(shifted),fixed_candidates=len(fixed),
            shifted_correct_current=sum(r['correct_current'] for r in shifted),
            shifted_correct_best_current=sum(r['correct_best_current'] for r in shifted),
            shifted_content_unique=sum(r['unique_final_location'] for r in shifted),
            early_shifted_candidates=len(early),early_unique_positions=len(useful_positions),
            effective_tokens_before_eos=eos,matched_candidate_pairs=len(matched),
            matched_shifted_correct=sum(r['shifted']['correct_best_current'] for r in matched),
            matched_fixed_correct=sum(r['fixed']['correct_current'] for r in matched),
            matched_content_shifted=sum(r['shifted']['content_present_nearby'] for r in matched),
            matched_content_fixed=sum(r['fixed']['content_present_nearby'] for r in matched),
            mean_shuffled_candidates=sum(len(r['candidates']) for r in shuffled)/CONFIG['shuffled_controls'],
            shifted_seen_also_fixed=sum((r['block'],tuple(r['tokens'])) in known for r in shifted),
            teacher_forward_seconds=sum(r['forward_seconds'] for r in trace),
            shifted_cost_oracle=move_cost,fixed_cost_oracle=fixed_cost))


def aggregate(prompts):
    import numpy as np
    keys=['shifted_candidates','fixed_candidates','early_shifted_candidates','early_unique_positions',
          'effective_tokens_before_eos','matched_candidate_pairs','matched_shifted_correct',
          'matched_fixed_correct','shifted_correct_current','shifted_correct_best_current']
    totals={key:sum(p['analysis']['summary'][key] for p in prompts) for key in keys}
    n=totals['matched_candidate_pairs']
    delta=(totals['matched_shifted_correct']-totals['matched_fixed_correct'])/n if n else None
    interval=None
    if prompts and n:
        rng=np.random.default_rng(CONFIG['seed']);idx=rng.integers(0,len(prompts),(CONFIG['bootstrap_samples'],len(prompts)))
        denominators=np.array([p['analysis']['summary']['matched_candidate_pairs'] for p in prompts])[idx].sum(1)
        numerators=np.array([p['analysis']['summary']['matched_shifted_correct']-p['analysis']['summary']['matched_fixed_correct'] for p in prompts])[idx].sum(1)
        valid=denominators>0
        interval=(np.percentile(numerators[valid]/denominators[valid],[2.5,97.5])*100).tolist()
    forward=sum(p['analysis']['summary']['teacher_forward_seconds'] for p in prompts)
    move=sum(p['analysis']['summary']['shifted_cost_oracle']['covered_forward_seconds'] for p in prompts)
    fixed=sum(p['analysis']['summary']['fixed_cost_oracle']['covered_forward_seconds'] for p in prompts)
    # Offline nuisance control: literal phrases shared by different prompts.
    # This dictionary never changes which candidates were constructed online.
    frequency=defaultdict(set)
    for index,prompt in enumerate(prompts):
        for row in prompt['trace']:
            for phrase in spans(row['draft'],row['active'],set(prompt['forbidden'])):
                frequency[phrase].add(index)
    common={phrase for phrase,owners in frequency.items() if len(owners)>=2}
    early=[r for p in prompts for r in p['analysis']['shifted'] if r.get('lead_to_absolute_stability',0)>=2]
    noncommon_positions=set()
    for index,prompt in enumerate(prompts):
        for row in prompt['analysis']['shifted']:
            if row.get('lead_to_absolute_stability',0)>=2 and tuple(row['tokens']) not in common:
                base=row['block']*CONFIG['block']+row['final_start']
                noncommon_positions.update((index,p) for p in range(base,base+CONFIG['phrase_length']))
    pairs=[r for p in prompts for r in p['analysis']['matched_coverage']]
    rare_pairs=[r for r in pairs if tuple(r['shifted']['tokens']) not in common and tuple(r['fixed']['tokens']) not in common]
    common_control=dict(shared_literal_phrases=len(common),
        early_shifted_shared_phrases=sum(tuple(r['tokens']) in common for r in early),
        early_noncommon_unique_positions=len(noncommon_positions),noncommon_matched_pairs=len(rare_pairs),
        noncommon_matched_shifted_correct=sum(r['shifted']['correct_best_current'] for r in rare_pairs),
        noncommon_matched_fixed_correct=sum(r['fixed']['correct_current'] for r in rare_pairs),
        scope='Cross-prompt literal frequency nuisance analysis, not an online phrase dictionary or gold filter.')
    return dict(**totals,matched_precision_delta_pp=None if delta is None else delta*100,
        matched_precision_delta_95ci_pp=interval,bootstrap_unit='Prompt',
        early_unique_token_coverage=totals['early_unique_positions']/max(1,totals['effective_tokens_before_eos']),
        shifted_reference_covered_forward_fraction=move/max(forward,1e-12),
        fixed_reference_covered_forward_fraction=fixed/max(forward,1e-12),
        common_phrase_control=common_control,
        mean_shuffled_candidates=sum(p['analysis']['summary']['mean_shuffled_candidates'] for p in prompts),
        matched_confidence_means={key:sum(r[key]['confidence'] for r in pairs)/len(pairs) if pairs else None for key in ('shifted','fixed')},
        initial_gate_pass=bool(len(prompts)==6 and totals['early_shifted_candidates']>=16 and
            totals['early_unique_positions']/max(1,totals['effective_tokens_before_eos'])>=.1 and
            move/max(forward,1e-12)>=.2 and interval is not None and interval[0]>=-2),
        scope='Small reused-state diagnostic. Reference agreement is not task correctness. '
              'Forward intervals are synchronized measured model-call costs, not clean request latency. '
              'Oracle coverage does not establish causal savings or lossless execution.')
