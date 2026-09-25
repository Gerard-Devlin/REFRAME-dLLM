from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from .core import BLOCK_LENGTH, GEN_LENGTH, MASK_ID


@dataclass
class DecodeResult:
    output: torch.Tensor
    nfe: int
    seconds: float
    peak_gib: float
    calls_per_block: list[int]


def transfer_schedule(mask_count: int, steps: int) -> list[int]:
    base, remainder = divmod(mask_count, steps)
    return [base + (index < remainder) for index in range(steps)]


@torch.no_grad()
def generate_fixed_quota(
    model,
    prompt: torch.Tensor,
    *,
    steps_per_block: int,
    gen_length: int = GEN_LENGTH,
    block_length: int = BLOCK_LENGTH,
    mask_id: int = MASK_ID,
) -> DecodeResult:
    if gen_length % block_length:
        raise ValueError("block_length must divide gen_length")
    if not 1 <= steps_per_block <= block_length:
        raise ValueError("steps_per_block must be between 1 and block_length")
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device)
    x[:, : prompt.shape[1]] = prompt
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(prompt.device)
        torch.cuda.synchronize(prompt.device)
    started = time.perf_counter()
    calls = []
    for block in range(gen_length // block_length):
        start = prompt.shape[1] + block * block_length
        end = start + block_length
        schedule = transfer_schedule(block_length, steps_per_block)
        block_calls = 0
        for amount in schedule:
            masked = (x[0, start:end] == mask_id).nonzero().flatten() + start
            if not len(masked):
                break
            logits = model(x, use_cache=False).logits[0].index_select(0, masked)
            predicted = logits.argmax(-1)
            confidence = F.softmax(logits.float(), dim=-1).gather(1, predicted[:, None]).squeeze(1)
            take = min(int(amount), len(masked))
            selected = confidence.topk(take).indices
            x[0, masked[selected]] = predicted[selected]
            block_calls += 1
        if (x[0, start:end] == mask_id).any():
            raise AssertionError("fixed quota schedule did not complete block")
        calls.append(block_calls)
    if torch.cuda.is_available():
        torch.cuda.synchronize(prompt.device)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(prompt.device) / 2**30 if torch.cuda.is_available() else 0.0
    return DecodeResult(x, sum(calls), seconds, peak, calls)

