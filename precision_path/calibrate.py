"""Fixed-score request-level split conformal; no per-step independence assumption.

This calibrates a REFERENCE-PATH diagnostic. The first-divergence implication
requires a future executor to preserve all BF16 persistent state and decisions.
No online executor, per-request certificate or empirical accuracy claim here.
"""
import argparse
from decimal import Decimal, ROUND_CEILING
import json
import math
from pathlib import Path

from .common import sha256,write_json


def threshold(scores,alpha):
    if not scores or not 0<alpha<1 or any(not math.isfinite(x) or x<0 for x in scores):
        raise ValueError('Need nonnegative finite request scores and alpha in (0,1)')
    k=int((Decimal(len(scores)+1)*(1-Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))
    return (None if k>len(scores) else sorted(scores)[k-1]),k


def evaluate(rows,cutoff):
    accepted=wrong=total=bad_requests=0
    bf=low=score=fallback=0.0
    for row in rows:
        bad=False
        for state in row['states']:
            allow=cutoff is not None and state['score']>cutoff
            total+=1; accepted+=allow
            wrong+=allow and not state['equal']
            bad|=allow and not state['equal']
            bf+=state['bf16_seconds']; low+=state['low_seconds']; score+=state['score_seconds']
            if not allow: fallback+=state['bf16_seconds']
        bad_requests+=bad
    return dict(requests=len(rows),ordinary_calls=total,accepted_calls=accepted,
        call_coverage=accepted/total if total else 0,
        mismatching_accepted_calls=wrong,local_risk=wrong/accepted if accepted else None,
        requests_with_reference_path_acceptance_error=bad_requests,
        empirical_request_error_rate=bad_requests/len(rows) if rows else None,
        time_weighted_fallback=fallback/bf if bf else None,
        ordinary_cost_ratio=(low+score+fallback)/bf if bf else None,
        cost_note='Low forward always paid, plus score, plus BF16 on rejection. '
                  'Same-state diagnostic timings; excludes fixed calls and online dispatch, not measured speedup.')


def development_curve(rows):
    """Exploratory thresholds on DEVELOPMENT only; never choose on final eval."""
    scores=sorted(s['score'] for r in rows for s in r['states'])
    if not scores:
        raise ValueError('No complete development states')
    cutoffs=sorted({0.0,*[scores[round((len(scores)-1)*i/20)] for i in range(21)]})
    return [dict(threshold=c,accept_none=c is None,**evaluate(rows,c)) for c in [None,*cutoffs]]


def validate(calibration,evaluation):
    for report,role in ((calibration,'calibration'),(evaluation,'evaluation')):
        if report.get('status')!='complete' or report.get('stage')!='audit' or report.get('role')!=role:
            raise ValueError(f'Need completed {role} full audit, not development/cost traces')
        ids=[]
        for row in report['rows']:
            if not row['complete'] or len(row['states'])!=row['normal_calls']:
                raise ValueError('Incomplete trajectory')
            ids.append(row['prompt_id'])
            actual=max((s['score'] for s in row['states'] if not s['equal']),default=0.0)
            if actual!=row['risk_score']:
                raise ValueError('Stored request maximum differs from complete trajectory')
        if len(ids)!=len(set(ids)):
            raise ValueError('Duplicate requests')
    if calibration['config_hash']!=evaluation['config_hash']:
        raise ValueError('Frozen score/model/quantization/runtime/decode config mismatch')
    if {r['prompt_id'] for r in calibration['rows']} & {r['prompt_id'] for r in evaluation['rows']}:
        raise ValueError('Calibration/evaluation prompt leakage')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--alpha',type=float,default=.01)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new calibration output file')
    cal=json.loads(args.calibration.read_text()); test=json.loads(args.evaluation.read_text())
    validate(cal,test)
    cutoff,k=threshold([r['risk_score'] for r in cal['rows']],args.alpha)
    result=dict(alpha=args.alpha,n=len(cal['rows']),order_statistic_k=k,
        threshold=cutoff,accept_none=cutoff is None,strict_acceptance='score > threshold',
        config_hash=cal['config_hash'],evaluation=evaluate(test['rows'],cutoff),
        calibration_sha256=sha256(args.calibration),evaluation_sha256=sha256(args.evaluation),
        guarantee_scope='Marginal rank bound under exchangeable requests, fixed score/backend/settings '
                        'and a state-preserving executor. No conditional/subgroup/per-request guarantee.',
        online_state_preservation_validated=False,task_accuracy_measured=False)
    write_json(args.output,result); print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
