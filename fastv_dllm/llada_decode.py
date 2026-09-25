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
            selected = probs[0] >= threshold
            selected[probs[0].argmax()] = True
            chosen_positions = target[selected]
            x[0, chosen_positions] = tokens[0, selected]
    _sync(); elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return Result(x, nfe, elapsed, peak)
