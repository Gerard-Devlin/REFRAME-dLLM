"""Task accuracy and paired comparisons; frozen baselines are never generated."""
import hashlib
import json
from pathlib import Path
import random
import statistics


def accuracy(outcomes):
    if not outcomes or any(x not in (True, False, None, 0, 1) for x in outcomes):
        raise ValueError('Invalid or empty scoring outcomes')
    known = [x for x in outcomes if x is not None]
    correct = sum(known)
    unknown = len(outcomes)-len(known)
    return dict(examples=len(outcomes), scored_examples=len(known), correct=correct,
                scoring_unknown=unknown, accuracy=correct/len(outcomes) if not unknown else None,
                known_accuracy=correct/len(known) if known else None,
                accuracy_bounds=[correct/len(outcomes), (correct+unknown)/len(outcomes)])


def paired(candidate, baseline, *, draws=10000, seed=1234):
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError('Paired scoring requires the same nonempty prompt set')
    delta = [int(a)-int(b) for a,b in zip(candidate,baseline) if a is not None and b is not None]
    if not delta:
        return dict(paired_examples=0, excluded_unknown=len(candidate), accuracy_delta=None, ci95=None)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(draws))
    return dict(paired_examples=len(delta), excluded_unknown=len(candidate)-len(delta),
                accuracy_delta=statistics.mean(delta), ci95=[means[int(.025*draws)], means[int(.975*draws)]],
                bootstrap_draws=draws, bootstrap_seed=seed,
                scope='Paired development prompts; not an independent non-inferiority test')


def load_baselines(root, task, samples, dataset, length, model, revision, policy_hash=None):
    root, dataset = Path(root), Path(dataset)
    frozen = json.loads((root/'cpu_frozen_development128_20261003.json').read_text())
    ids = [str(s.get('id',s.get('task_id'))) for s in samples]
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    info = frozen['datasets'][task]
    if ids != info['development_ids'] or dataset_hash != info['sha256']:
        raise ValueError('Frozen baseline IDs/data do not match this development evaluation')
    folder = root/'focus_v4_quality128_20261003'/f'{task}_{length}_flash'
    if not (folder/'complete').exists():
        raise ValueError('Flash baseline is incomplete; never use a partial result')
    manifest = json.loads((folder/'manifest.json').read_text())
    scoring = json.loads((folder/'scores.json').read_text())
    summary = json.loads((folder/'summary.json').read_text())
    if ([str(i) for i in manifest['ids']] != ids or manifest['dataset_sha256'] != dataset_hash
            or manifest['model'] != model or manifest['revision'] != revision or manifest['length'] != length
            or manifest['sample_seed'] != 51713):
        raise ValueError('Flash baseline prompt/model/config alignment failed')
    if policy_hash is not None and scoring.get('policy_sha256') != policy_hash:
        raise ValueError('Mathematics scorer differs from the frozen baseline')
    flash = scoring['correct']['flash_verify_opt']
    if len(flash) != len(samples):
        raise ValueError('Frozen Flash scoring count differs')
    paths = [root/'cpu_frozen_development128_20261003.json',folder/'manifest.json',folder/'scores.json',folder/'summary.json']
    return dict(ids=ids, dataset_sha256=dataset_hash, flash_correct=flash,
                metrics=dict(frozen['cells'][f'{task}_{length}']['metrics'],
                             flash=summary['metrics']['flash_verify_opt']),
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                policy=scoring['policy'], policy_sha256=scoring.get('policy_sha256'),
                baseline_generation=False)


def compare(scoring, names, baseline):
    return dict(baselines={k:v for k,v in baseline.items() if k!='flash_correct'},
                vs_flash={name:paired(scoring['correct'][name],baseline['flash_correct']) for name in names},
                timing_scope='PACT latency includes fixed EOS work and possible first-shape compilation; '
                             'shared GPU timing cannot establish dedicated-card speedup')
