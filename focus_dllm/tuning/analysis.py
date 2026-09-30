"""Paired uncertainty and a fixed-input latency control for development runs."""
import numpy as np


def paired_intervals(summary, records, baseline='v1', repeats=10000):
    details = summary['official']['details']
    names = list(summary['metrics'])
    n = len(records)
    rng = np.random.default_rng(1234)
    indices = rng.integers(0, n, size=(repeats, n))
    base_score = np.array([r['methods'][baseline]['flexible-extract'] for r in details], float)
    base_time = np.array([r[baseline]['seconds'] for r in records])
    output = {}
    for name in names:
        if name == baseline:
            continue
        score = np.array([r['methods'][name]['flexible-extract'] for r in details], float)
        time = np.array([r[name]['seconds'] for r in records])
        delta = (score-base_score)[indices].mean(1)
        ratio = base_time[indices].mean(1)/time[indices].mean(1)
        output[name] = dict(accuracy_delta=float((score-base_score).mean()),
            accuracy_delta_ci95=np.quantile(delta, [.025, .975]).tolist(),
            latency_speedup=float(base_time.mean()/time.mean()),
            latency_speedup_ci95=np.quantile(ratio, [.025, .975]).tolist(),
            wins=int(((score == 1) & (base_score == 0)).sum()),
            losses=int(((score == 0) & (base_score == 1)).sum()))
    return output
