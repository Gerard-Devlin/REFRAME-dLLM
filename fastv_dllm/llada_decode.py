"""Original LLaDA parallel decoder with an optional physically-pruned forward."""

from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from .llada_common import MASK_ID


@dataclass
class Result:
    output: torch.Tensor
    nfe: int
    seconds: float
    peak_gib: float


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _selected_positions(probs, threshold):
    """Official confidence decoding, including the one-token control."""
    if threshold is None:
        selected = torch.zeros_like(probs[0], dtype=torch.bool)
    else:
        selected = probs[0] >= threshold
    selected[probs[0].argmax()] = True
    return selected


@torch.no_grad()
def generate(model, prompt, gen_length=256, block_length=32, threshold=0.9,
             mask_id=MASK_ID, block_forward=None, prune=False):
    if gen_length % block_length:
        raise ValueError("block_length must divide gen_length")
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=prompt.device)
    x[:, :prompt.shape[1]] = prompt
    nfe = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync(); started = time.perf_counter()
    for block in range(gen_length // block_length):
        start = prompt.shape[1] + block * block_length
        end = start + block_length
        while (x[:, start:end] == mask_id).any():
            target = (x[0, start:end] == mask_id).nonzero().flatten() + start
            positions = target.tolist()
            if block_forward is None:
                logits = model(x).logits.index_select(1, target)
            else:
                logits = block_forward(x, positions, prune=prune)
            nfe += 1
            tokens = logits.argmax(-1)
            probs = F.softmax(logits.to(torch.float64), dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            selected = _selected_positions(probs, threshold)
            chosen_positions = target[selected]
            x[0, chosen_positions] = tokens[0, selected]
    _sync(); elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return Result(x, nfe, elapsed, peak)


@torch.no_grad()
def generate_prefix_cache(model, prompt, gen_length=256, block_length=32, threshold=0.9,
                          mask_id=MASK_ID, block_forward=None, prune=False):
    """Fast-dLLM prefix cache with an optional FastV suffix forward.

    The once-per-block warm-up remains the unmodified model so every layer gets
    an exact prefix cache.  Only ordinary refinement calls physically prune
    untouched future MASK positions; the cached prefix and current block are
    always retained.
    """
    if gen_length % block_length:
        raise ValueError("block_length must divide gen_length")
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=prompt.device)
    x[:, :prompt.shape[1]] = prompt
    nfe = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync(); started = time.perf_counter()
    for block in range(gen_length // block_length):
        start = prompt.shape[1] + block * block_length
        end = start + block_length

        # Exact Fast-dLLM warm-up and formal prefix-cache construction.
        output = model(x, use_cache=True)
        nfe += 1
        target = (x[0, start:end] == mask_id).nonzero().flatten() + start
        logits = output.logits.index_select(1, target)
        tokens = logits.argmax(-1)
        probs = F.softmax(logits.to(torch.float64), dim=-1).gather(
            -1, tokens.unsqueeze(-1)
        ).squeeze(-1)
        selected = _selected_positions(probs, threshold)
        x[0, target[selected]] = tokens[0, selected]

        past_key_values = [
            tuple(value[:, :, :start] for value in layer)
            for layer in output.past_key_values
        ]
        while (x[:, start:end] == mask_id).any():
            suffix = x[:, start:]
            target = (suffix[0, :block_length] == mask_id).nonzero().flatten()
            positions = target.tolist()
            if block_forward is None:
                logits = model(
                    suffix, past_key_values=past_key_values, use_cache=True
                ).logits.index_select(1, target)
            else:
                logits = block_forward(
                    suffix, positions, prune=prune,
                    past_key_values=past_key_values, use_cache=True,
                )
            nfe += 1
            tokens = logits.argmax(-1)
            probs = F.softmax(logits.to(torch.float64), dim=-1).gather(
                -1, tokens.unsqueeze(-1)
            ).squeeze(-1)
            selected = _selected_positions(probs, threshold)
            x[0, start + target[selected]] = tokens[0, selected]
    _sync(); elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return Result(x, nfe, elapsed, peak)


@torch.no_grad()
def generate_dual_cache(model, prompt, gen_length=256, block_length=32, threshold=0.9,
                        mask_id=MASK_ID, block_forward=None, prune=False):
    """Fast-dLLM DualCache under the same timing and metric contract.

    DualCache evaluates only the current block after its once-per-block full
    warm-up.  Consequently, the safe FastV policy has no untouched future MASK
    canvas to prune during refinement.  Passing ``block_forward`` remains useful
    as an executable check of that structural overlap: the pruning wrapper must
    take its exact-forward branch and preserve the official output.
    """
    if gen_length % block_length:
        raise ValueError("block_length must divide gen_length")
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=prompt.device)
    x[:, :prompt.shape[1]] = prompt
    nfe = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync(); started = time.perf_counter()
    for block in range(gen_length // block_length):
        start = prompt.shape[1] + block * block_length
        end = start + block_length

        output = model(x, use_cache=True)
        nfe += 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, start:end] = True

        target = (x[0, start:end] == mask_id).nonzero().flatten() + start
        logits = output.logits.index_select(1, target)
        tokens = logits.argmax(-1)
        probs = F.softmax(logits.to(torch.float64), dim=-1).gather(
            -1, tokens.unsqueeze(-1)
        ).squeeze(-1)
        selected = _selected_positions(probs, threshold)
        x[0, target[selected]] = tokens[0, selected]

        past_key_values = output.past_key_values
        while (x[:, start:end] == mask_id).any():
            block_ids = x[:, start:end]
            target = (block_ids[0] == mask_id).nonzero().flatten()
            positions = target.tolist()
            if block_forward is None:
                logits = model(
                    block_ids, past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position,
                ).logits.index_select(1, target)
            else:
                logits = block_forward(
                    block_ids, positions, prune=prune,
                    past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position,
                )
            nfe += 1
            tokens = logits.argmax(-1)
            probs = F.softmax(logits.to(torch.float64), dim=-1).gather(
                -1, tokens.unsqueeze(-1)
            ).squeeze(-1)
            selected = _selected_positions(probs, threshold)
            x[0, start + target[selected]] = tokens[0, selected]
    _sync(); elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return Result(x, nfe, elapsed, peak)
