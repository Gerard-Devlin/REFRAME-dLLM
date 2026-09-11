"""Threshold/factor decoding with reference caches spanning several blocks."""
import time

import torch

from generate import get_transfer_index, get_transfer_index_dynamic
from .model import ReframeConfig, ReframeSession
from .transport import relative_error


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def generate_reframe(model, prompt, gen_length=256, block_length=32,
                     threshold=0.9, factor=None, mask_id=126336,
                     config=None, trace=None, audit_every=0):
    """Returns sequence, actual attempted NFE, per-request diagnostics.

    Last completed block is recomputed with final token identities in the next
    block's first partial forward. No invisible extra commit pass is omitted
    from timing. Failed partial forwards count toward NFE and elapsed time.
    Model load, dataset load and external oracle probes are excluded.
    """
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("Use batch_size=1")
    if gen_length <= 0 or block_length <= 0 or gen_length % block_length:
        raise ValueError("Positive generation length must be divisible by block length")
    if threshold is None and factor is None:
        raise ValueError("Prototype requires threshold or factor decoding")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    cfg = config or ReframeConfig()
    if audit_every < 0:
        raise ValueError("audit_every must be nonnegative")
    synchronize(prompt.device)
    started = time.perf_counter()
    session = ReframeSession(model, cfg)
    if prompt.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(prompt.device)
    plen = prompt.shape[1]
    x = torch.full((1, plen + gen_length), mask_id, dtype=prompt.dtype, device=prompt.device)
    x[:, :plen] = prompt
    last_refresh_block = -cfg.refresh_blocks
    decisions, audits = 0, []
    for block in range(gen_length // block_length):
        start, end = plen + block * block_length, plen + (block + 1) * block_length
        pending = torch.arange(start - block_length, start, device=x.device) if block else None
        iteration = 0
        while bool((x[:, start:end] == mask_id).any().item()):
            force = iteration == 0 and block - last_refresh_block >= cfg.refresh_blocks
            logits, refreshed = session.step(x, start, end, pending, force)
            if refreshed:
                last_refresh_block = block
            pending = None
            old = x[:, start:end]
            mask = old == mask_id
            if factor is None:
                pred, transfer = get_transfer_index(logits, 0.0, "low_confidence", mask, old, None, threshold)
            else:
                pred, transfer = get_transfer_index_dynamic(logits, 0.0, "low_confidence", mask, old, None, factor)
            if audit_every and decisions % audit_every == 0:
                # Read-only same-state full model: never feeds back into the
                # approximate trajectory. Probe time remains in elapsed time,
                # so audited runs are explicitly ineligible as speed results.
                full_logits, _ = session.full(x, refresh=False)
                target = full_logits[:, start:end]
                if factor is None:
                    fp, ft = get_transfer_index(target, 0.0, "low_confidence", mask, old, None, threshold)
                else:
                    fp, ft = get_transfer_index_dynamic(target, 0.0, "low_confidence", mask, old, None, factor)
                audits.append(dict(block=block, iteration=iteration,
                                   logits_error=relative_error(logits[mask], target[mask]),
                                   top1_agreement=float((pred[mask] == fp[mask]).float().mean().item()),
                                   commit_set_agreement=bool(torch.equal(transfer, ft)),
                                   committed_token_agreement=bool(torch.equal(pred[transfer], fp[transfer]))))
            decisions += 1
            # A model proposing MASK is not making decoding progress.
            if bool((pred[transfer] == mask_id).any().item()):
                raise RuntimeError("Model selected MASK; cannot count it as a committed token")
            if not bool(transfer.any().item()):
                raise RuntimeError("Decoder made no progress")
            if trace is not None:
                trace(dict(block=block, iteration=iteration, full_refresh=refreshed,
                           positions=(transfer[0].nonzero().flatten() + start).tolist(),
                           tokens=pred[transfer].tolist()))
            x[:, start:end] = torch.where(transfer, pred, old)
            iteration += 1
            if iteration > block_length:
                raise RuntimeError("Exceeded one-token-per-step fallback budget")
    synchronize(prompt.device)
    stats = dict(session.stats)
    stats.update(elapsed_seconds=time.perf_counter() - started,
                 nfe=stats["full_forwards"] + stats["partial_forwards"],
                 generated_slots=gen_length, prompt_tokens=plen,
                 peak_memory_bytes=(torch.cuda.max_memory_allocated(prompt.device)
                                    if prompt.device.type == "cuda" else None))
    stats.update(diagnostic_run=bool(audit_every), audit_full_forwards=len(audits), audits=audits)
    # Slots/s is deliberately distinguished from useful output tokens/s.
    stats["slots_per_second"] = gen_length / stats["elapsed_seconds"]
    return x, stats["nfe"], stats
