"""Paid full-spectrum diagnostics of native state differences; no delta executor."""
from contextlib import AbstractContextManager
import math
import time

import numpy as np
import torch


class CaptureComponents(AbstractContextManager):
    """Observe projected attention, MLP and residual outputs without replacing them.

    In eval mode LLaDA's attn_out and ff_out are the two residual contributions.
    CPU activation copies are diagnostic overhead, never an online cached state.
    """
    def __init__(self, model):
        self.model, self.rows, self.hooks = model, [], []

    def __enter__(self):
        try:
            if self.model.training:
                raise ValueError('Component capture requires eval mode')
            for block in self.model.model.transformer.blocks:
                row = {}
                self.rows.append(row)
                def pre(module, inputs, _row=row):
                    _row['input'] = inputs[0].detach().cpu().clone()
                def post(module, inputs, output, _row=row):
                    _row['hidden'] = output[0].detach().cpu().clone()
                def attention(module, inputs, output, _row=row):
                    _row['attention'] = output.detach().cpu().clone()
                def mlp(module, inputs, output, _row=row):
                    _row['mlp'] = output.detach().cpu().clone()
                self.hooks.extend([block.register_forward_pre_hook(pre),
                    block.register_forward_hook(post),
                    block.attn_out.register_forward_hook(attention),
                    block.ff_out.register_forward_hook(mlp)])
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return False


def remove_energetic_rows(delta, fraction=.25):
    """Oracle row budget; ties by original position, remaining rows keep order."""
    if delta.ndim != 2 or not 0 <= fraction <= 1:
        raise ValueError('Expected a matrix and a fraction in [0,1]')
    energy = delta.double().square().sum(-1).cpu().numpy()
    count = math.ceil(len(energy) * fraction)
    removed = np.argsort(-energy, kind='stable')[:count]
    keep = np.ones(len(energy), dtype=bool)
    keep[removed] = False
    retained = np.flatnonzero(keep)
    total = float(energy.sum())
    return delta[torch.as_tensor(retained, device=delta.device)], dict(
        removed_positions=removed.tolist(), removed_rows=count, total_rows=len(energy),
        tail_energy_fraction=float(energy[keep].sum()/total) if total else 0.,
        removed_energy_fraction=float(energy[~keep].sum()/total) if total else 0.)


def spectrum_summary(energy, rows, columns):
    """Ranks describe Frobenius energy, not exact algebraic rank or a certificate."""
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 1 or np.any(~np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError('Invalid squared singular spectrum')
    if np.any(np.diff(energy) > 0):
        raise ValueError('Spectrum must be descending')
    total = float(energy.sum())
    weights = energy / total if total else np.zeros_like(energy)
    cumulative = np.cumsum(weights)
    ranks = {f'r{pct}': min(len(energy), int(np.searchsorted(cumulative, pct/100)+1))
             if total else 0 for pct in (90, 95, 99)}
    positive = weights[weights > 0]
    return dict(rows=int(rows), columns=int(columns), energy=total, **ranks,
        stable_rank=float(total/energy[0]) if total else 0.,
        entropy_effective_rank=float(np.exp(-(positive*np.log(positive)).sum())) if total else 0.,
        energy_at_rank={str(r): float(weights[:r].sum()) for r in (1,2,4,8,16,32,64)},
        normalized_top64_energy=weights[:64].tolist(),
        r99_over_rows=ranks['r99']/rows if rows else 0.)


@torch.no_grad()
def full_spectrum(delta, device='cpu'):
    """All singular energies via the smaller FP64 Gram matrix and eigvalsh.

    No randomized or truncated SVD. FP64 inputs are exact conversions of the
    observed BF16 differences. Negative eigenmass and trace error are audited;
    clipping only roundoff at the numerical floor does not certify exact rank.
    """
    if delta.ndim != 2 or not torch.isfinite(delta).all():
        raise ValueError('Expected a finite matrix')
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    began = time.perf_counter()
    matrix = delta.to(device=device, dtype=torch.float64)
    rows, columns = matrix.shape
    total = float(matrix.square().sum())
    if min(rows, columns) == 0 or total == 0:
        energy = np.zeros(min(rows, columns), dtype=np.float64)
        negative, trace_error = 0., 0.
    else:
        gram = matrix @ matrix.T if rows <= columns else matrix.T @ matrix
        values = torch.linalg.eigvalsh(gram).cpu().numpy()
        negative = float(-values[values < 0].sum()/total)
        trace_error = abs(float(values.sum())-total)/total
        if negative > 1e-10 or trace_error > 1e-10:
            raise ArithmeticError(f'FP64 spectrum audit failed: {negative=}, {trace_error=}')
        energy = np.maximum(values[::-1], 0).copy()
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    result = spectrum_summary(energy, rows, columns)
    result.update(negative_eigenmass_fraction=negative, trace_relative_error=trace_error,
                  spectral_seconds=time.perf_counter()-began, method='full_FP64_Gram_eigvalsh')
    return result, energy


def residual_audit(previous, current):
    """Audit actual BF16 residual arithmetic, and measure its delta-rounding term."""
    for state in (previous, current):
        reconstructed = (state['input'] + state['attention']) + state['mlp']
        if not torch.equal(reconstructed, state['hidden']):
            raise AssertionError('Captured components do not reconstruct the native residual output')
    d = {k: current[k].double()-previous[k].double() for k in previous}
    remainder = d['hidden'] - d['input'] - d['attention'] - d['mlp']
    energy = float(d['hidden'].square().sum())
    return dict(bf16_residual_reconstruction_bitwise=True,
        delta_rounding_energy_fraction=float(remainder.square().sum())/energy if energy else 0.,
        delta_rounding_max_absolute=float(remainder.abs().max()))


def audit_legacy_row_energy(previous, current, saved):
    """Compare the old FP32 statistic in its own arithmetic; spectra stay FP64."""
    delta = current.float()-previous.float()
    legacy = delta.norm(dim=-1).square().numpy()
    if not np.array_equal(legacy, np.asarray(saved, dtype=np.float32)):
        raise AssertionError('Native hidden row energies differ in the original FP32 arithmetic')
    precise = (current.double()-previous.double()).square().sum(-1).numpy()
    gap = np.abs(precise-legacy)/np.maximum(precise,1e-300)
    return dict(original_fp32_row_energy_bitwise_equal=True,
                fp32_vs_fp64_row_energy_max_relative_gap=float(gap.max()))
