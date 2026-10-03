"""Candidate-evidence Gaussian copula reference, not a greedy-equivalent decoder.

Within-position Gaussian covariance is identity. Cross-position dependence
comes from normalized h * output-weight evidence. Full-vocabulary categorical
marginals are preserved in exact arithmetic by an independent TAIL class.
This does not preserve a joint distribution, trajectory, or task accuracy.
"""
from dataclasses import dataclass
import math

import torch


@dataclass
class Factors:
    shared: torch.Tensor       # positions, candidates, channels
    residual: torch.Tensor     # positions, candidates, candidates


def factors(evidence, gamma=.8, regularization=.001):
    if evidence.ndim != 3 or not 0 <= gamma <= 1 or regularization <= 0:
        raise ValueError('Finite position/candidate/channel evidence and valid coupling required')
    if min(evidence.shape) < 1 or not torch.isfinite(evidence).all():
        raise ValueError('Empty/nonfinite evidence')
    value = evidence.double()
    gram = value @ value.transpose(-1, -2)
    # Equivalent to lambda = regularization * mean row squared norm, with a
    # floor for zero evidence. No learned scale or task-specific parameter.
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    lam = regularization * scale
    eigenvalues, q = torch.linalg.eigh(gram)
    if (eigenvalues < -1e-8 * scale[:, None]).any():
        raise ValueError('Invalid evidence Gram spectrum')
    eigenvalues = eigenvalues.clamp_min(0)
    invsqrt = (eigenvalues + lam[:, None]).rsqrt()
    shared = gamma * ((q * invsqrt[:, None, :]) @ q.transpose(-1, -2)) @ value
    remaining = (1 - gamma**2 * eigenvalues / (eigenvalues + lam[:, None])).clamp_min(0)
    residual = (q * remaining.sqrt()[:, None, :]) @ q.transpose(-1, -2)
    return Factors(shared, residual)


def shuffled_factors(value, permutations):
    """Shuffle candidate correspondence AND residual covariance consistently."""
    p, k, _ = value.shared.shape
    if permutations.shape != (p, k) or permutations.dtype != torch.long:
        raise ValueError('One bijective candidate permutation per position required')
    if not torch.equal(permutations.sort(-1).values, torch.arange(k, device=permutations.device).expand(p, -1)):
        raise ValueError('Not a candidate permutation')
    r = value.shared.gather(1, permutations[:, :, None].expand_as(value.shared))
    c = value.residual.gather(1, permutations[:, :, None].expand_as(value.residual))
    c = c.gather(2, permutations[:, None, :].expand_as(c))
    return Factors(r, c)


def category_probabilities(logits, k=4, temperature=1.):
    if logits.ndim != 2 or logits.shape[0] < 1 or not 1 <= k <= logits.shape[1]:
        raise ValueError('Nonempty full vocabulary logits required')
    if not math.isfinite(temperature) or temperature <= 0 or not torch.isfinite(logits).all():
        raise ValueError('Copula requires positive finite temperature and finite logits')
    probabilities = (logits.double() / temperature).softmax(-1)
    _, tokens = logits.topk(k, dim=-1)
    mass = probabilities.gather(1, tokens)
    remainder = probabilities.clone().scatter_(1, tokens, 0.)
    # Summing the complement avoids catastrophic 1 - top-mass cancellation.
    categories = torch.cat((mass, remainder.sum(-1, keepdim=True)), -1)
    categories /= categories.sum(-1, keepdim=True)
    return probabilities, tokens, categories


def gumbels(normals):
    # Finite FP64 CDF clipping is explicitly a numerical approximation of the
    # ideal copula. Never advertise universal exact sampling from this code.
    uniforms = torch.special.ndtr(normals.double())
    eps = torch.finfo(torch.float64).eps
    return -torch.log(-torch.log(uniforms.clamp(min=eps, max=1-eps)))


def categorical(normals, mass):
    if normals.ndim != 3 or mass.ndim != 2 or normals.shape[1:] != mass.shape:
        raise ValueError('Aligned draw/position/category arrays required')
    if (mass < 0).any() or not torch.isfinite(mass).all() or not torch.allclose(mass.sum(-1), torch.ones_like(mass[:, 0])):
        raise ValueError('Normalized nonnegative category probabilities required')
    return (mass.log()[None] + gumbels(normals)).argmax(-1)


def draw_categories(value, mass, draws, generator):
    """Literal shared-xi construction; TAIL has its own independent Gaussian."""
    p, k, d = value.shared.shape
    if mass.shape != (p, k+1) or draws < 1:
        raise ValueError('Include exactly one independent TAIL category')
    options = dict(dtype=torch.float64, device=value.shared.device, generator=generator)
    xi = torch.randn(draws, d, **options)
    eta = torch.randn(draws, p, k, **options)
    u = torch.einsum('pkd,nd->npk', value.shared, xi) + torch.einsum('pkj,npj->npk', value.residual, eta)
    tail = torch.randn(draws, p, 1, **options)
    return categorical(torch.cat((u, tail), -1), mass)


def draw_tokens(logits, evidence, draws, generator, *, gamma=.8, regularization=.001, temperature=1., permutations=None):
    k = evidence.shape[1]
    full, tokens, mass = category_probabilities(logits, k, temperature)
    value = factors(evidence, gamma, regularization)
    if permutations is not None:
        value = shuffled_factors(value, permutations)
    choices = draw_categories(value, mass, draws, generator)
    result = tokens.gather(1, choices.clamp_max(k-1).T).T
    for position in range(logits.shape[0]):
        tail = choices[:, position] == k
        count = int(tail.sum())
        if count:
            remainder = full[position].clone().scatter_(0, tokens[position], 0.)
            if not remainder.sum() > 0:
                raise RuntimeError('Impossible zero-mass TAIL selected')
            result[tail, position] = torch.multinomial(remainder, count, replacement=True, generator=generator)
    return result, choices


def pair_table(value, mass, draws, seed):
    """Paid offline Monte Carlo with the equivalent 2K-dimensional covariance.

Avoid materializing draws x hidden-width for mechanism statistics. This is
not the implementation whose online construction/sampling time is reported.
"""
    if value.shared.shape[0] != 2 or mass.shape != (2, value.shared.shape[1]+1):
        raise ValueError('Exactly two preselected positions required')
    k = value.shared.shape[1]
    covariance = torch.eye(2*k, dtype=torch.float64, device=mass.device)
    cross = value.shared[0] @ value.shared[1].T
    covariance[:k, k:] = cross
    covariance[k:, :k] = cross.T
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    if eigenvalues.min() < -1e-9:
        raise ValueError('Invalid joint covariance')
    square_root = vectors * eigenvalues.clamp_min(0).sqrt()[None]
    rng = torch.Generator(device=mass.device).manual_seed(seed)
    u = torch.randn(draws, 2*k, dtype=torch.float64, device=mass.device, generator=rng) @ square_root.T
    u = u.reshape(draws, 2, k)
    tail = torch.randn(draws, 2, 1, dtype=torch.float64, device=mass.device, generator=rng)
    choices = categorical(torch.cat((u, tail), -1), mass)
    flat = choices[:, 0] * (k+1) + choices[:, 1]
    return torch.bincount(flat, minlength=(k+1)**2).reshape(k+1, k+1).double() / draws


def interaction(raw, mass):
    """Remove row/column effects under the ORIGINAL categorical marginals.

TAIL interactions are initially neutral (not evaluated with teacher calls).
The centering ensures independent expectation zero and prevents changes in
single-position preferences from masquerading as dependence information.
"""
    if raw.shape != (mass.shape[1], mass.shape[1]) or mass.shape[0] != 2:
        raise ValueError('Full top-K plus TAIL compatibility table required')
    row = raw @ mass[1]
    col = mass[0] @ raw
    return raw - row[:, None] - col[None] + mass[0] @ raw @ mass[1]


def choose_pair(logits, positions, forbidden):
    """Pre-draw ambiguity selection; no sampled token, gold, or future output."""
    if positions.ndim != 1 or logits.shape[0] != len(positions):
        raise ValueError('Current active MASK rows only')
    if len(positions) < 2:
        return None
    top = logits.argmax(-1)
    eligible = torch.ones(len(positions), dtype=torch.bool, device=top.device)
    for token in forbidden:
        eligible &= top != token
    indices = eligible.nonzero().flatten()
    if len(indices) < 2:
        return None
    logp = logits.double().log_softmax(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    order = torch.argsort(entropy.index_select(0, indices), descending=True, stable=True)
    return indices.index_select(0, order[:2])
