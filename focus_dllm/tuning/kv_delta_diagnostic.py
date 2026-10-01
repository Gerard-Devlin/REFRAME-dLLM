"""Offline geometry of same-state future KV changes, never an online update.

Both ``fresh`` and the optional fitted scalar require a full teacher forward.
The source mean is an optimistic information control: live FOCUS KV need not
equal these teacher values. No result here establishes action preservation.
"""
import torch


@torch.no_grad()
def delta_geometry(old, fresh, source_positions, dropped_positions):
    """Measure whether source deltas share the direction of dropped deltas.

    Positions are absolute indices in the same [batch, head, position, dim]
    canvas. Sum of squared error is retained so aggregation remains weighted
    by actual delta energy rather than averaging unstable per-state ratios.
    """
    if old.ndim != 4 or old.shape != fresh.shape:
        raise ValueError('Expected matching full teacher KV tensors')
    if old.dtype != fresh.dtype or old.device != fresh.device:
        raise ValueError('Teacher KV must share dtype and device')
    for positions in (source_positions, dropped_positions):
        if positions.ndim != 1 or positions.dtype != torch.long:
            raise ValueError('Positions must be one-dimensional int64 tensors')
        if positions.device != old.device:
            raise ValueError('Positions and KV must share a device')
        if positions.numel() and (positions.min() < 0 or positions.max() >= old.shape[-2]):
            raise ValueError('Position outside the canvas')
        if positions.unique().numel() != positions.numel():
            raise ValueError('Repeated position')
    if torch.isin(source_positions, dropped_positions).any():
        raise ValueError('Dropped positions cannot supply their own prediction')
    if not source_positions.numel() or not dropped_positions.numel():
        return None
    source = fresh.index_select(-2, source_positions).float() - old.index_select(-2, source_positions).float()
    target = fresh.index_select(-2, dropped_positions).float() - old.index_select(-2, dropped_positions).float()
    mean = source.mean(-2, keepdim=True)
    energy = target.square().sum()
    norm = mean.square().sum() * dropped_positions.numel()
    dot = (target * mean).sum()
    mean_error = (target - mean).square().sum()
    # Best scalar is fitted using unavailable dropped labels: upper information
    # control only. It must never be reused as an executable correction.
    fitted_error = energy - dot.square() / norm.clamp_min(1e-30)
    values = torch.stack((energy, mean_error, norm, dot, fitted_error.clamp_min(0))).double().cpu().tolist()
    return dict(elements=target.numel(), source_positions=source_positions.numel(),
                dropped_positions=dropped_positions.numel(), delta_energy=values[0],
                source_mean_error=values[1], source_mean_energy=values[2],
                cross_dot=values[3], offline_fitted_error=values[4])


def aggregate_geometry(records):
    """Aggregate additive errors; report undefined ratios for zero changes."""
    totals = {name: sum(row[name] for row in records) for name in
              ('elements', 'delta_energy', 'source_mean_error', 'source_mean_energy',
               'cross_dot', 'offline_fitted_error')}
    energy = totals['delta_energy']
    return dict(records=len(records), **totals,
                source_mean_relative_error=totals['source_mean_error'] / energy if energy else None,
                offline_fitted_relative_error=totals['offline_fitted_error'] / energy if energy else None)
