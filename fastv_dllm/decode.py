"""Pinned v2 decoder with an optional FastV block-forward implementation."""

from dataclasses import dataclass
import time

import torch

from .common import EOS_ID, MASK_ID


@dataclass
class DecodeResult:
    output: torch.Tensor
    logical_forwards: int
    ordinary_denoise: int
    cache_writes: int
    prefill: int
    seconds: float
    peak_gib: float


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens=512,
    mask_id=MASK_ID,
    threshold=0.90,
    small_block_size=8,
    block_size=32,
    stop_token=EOS_ID,
    top_p=0.95,
    temperature=0,
    use_block_cache=False,
    block_forward=None,
    prune=False,
):
    if temperature != 0:
        raise ValueError("FastV experiment is defined only for deterministic temperature=0 decoding")
    if max_new_tokens < block_size or max_new_tokens % block_size or block_size % small_block_size:
        raise ValueError("Use whole output blocks and a sub-block divisor")
    if prune and use_block_cache:
        raise ValueError("Physical full-block pruning and official sub-block cache are separate baselines")
    if block_forward is not None and use_block_cache:
        raise ValueError("Probe/custom block forward does not run with block cache")

    counters = dict(logical_forwards=0, ordinary_denoise=0, cache_writes=0, prefill=0)

    def native(**kwargs):
        counters["logical_forwards"] += 1
        if kwargs.get("update_past_key_values"):
            if counters["logical_forwards"] == 1:
                counters["prefill"] += 1
            else:
                counters["cache_writes"] += 1
        else:
            counters["ordinary_denoise"] += 1
        return model.forward(**kwargs)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync()
    started = time.perf_counter()
    num_blocks = max_new_tokens // block_size
    original_input_length = input_ids.shape[1]

    if input_ids.shape[1] > block_size:
        output = native(input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)],
                        use_cache=True, update_past_key_values=True, block_size=block_size)
        logits, past_key_values = output.logits, output.past_key_values
        if input_ids.shape[1] % block_size == 0:
            input_ids = torch.cat([input_ids, logits[:, -1:, :].argmax(dim=-1)], dim=1)
    else:
        past_key_values = None

    num_small_blocks = block_size // small_block_size
    for _block_idx in range(num_blocks):
        if stop_token in input_ids[:, original_input_length:]:
            break
        prompt_length = input_ids.shape[1]
        x_init = mask_id * torch.ones(
            (input_ids.shape[0], block_size - prompt_length % block_size), device=model.device, dtype=torch.long
        )
        x_t = torch.cat([input_ids, x_init], dim=1)
        block_past_key_values = None
        while True:
            if stop_token in x_t[:, prompt_length:]:
                stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                if (x_t[:, prompt_length:prompt_length + stop_token_idx] == mask_id).sum() == 0:
                    break
            mask_idx = x_t[:, -block_size:] == mask_id
            if mask_idx.sum() == 0:
                output = native(input_ids=x_t[:, -block_size:], use_cache=True,
                                past_key_values=past_key_values, update_past_key_values=True,
                                block_size=block_size)
                logits, past_key_values = output.logits, output.past_key_values
                x_t = torch.cat([x_t, logits[:, -1:, :].argmax(dim=-1)], dim=1)
                break
            for small_block_idx in range(num_small_blocks):
                small_start = small_block_idx * small_block_size
                small_end = small_start + small_block_size
                start = -block_size + small_start
                end = None if small_end == block_size else -block_size + small_end
                while True:
                    mask_idx = x_t[:, -block_size:] == mask_id
                    if mask_idx[:, start:end].sum() == 0:
                        break
                    if stop_token in x_t[:, prompt_length:]:
                        stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                        if (x_t[:, prompt_length:prompt_length + stop_token_idx] == mask_id).sum() == 0:
                            break
                    if use_block_cache:
                        if block_past_key_values is None or (x_t[:, -block_size + small_start] == mask_id).any():
                            output = native(input_ids=x_t[:, -block_size:], use_cache=True,
                                            past_key_values=past_key_values, update_past_key_values=False,
                                            use_block_cache=True)
                            logits, block_past_key_values = output.logits, output.block_past_key_values
                            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)[:, start:end]
                        else:
                            logits = native(input_ids=x_t[:, start:end], use_cache=True,
                                            past_key_values=past_key_values, update_past_key_values=False,
                                            use_block_cache=True, block_past_key_values=block_past_key_values,
                                            replace_position=small_start).logits
                            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                    elif block_forward is not None:
                        counters["logical_forwards"] += 1
                        counters["ordinary_denoise"] += 1
                        logits = block_forward(x_t[:, -block_size:], past_key_values, small_start, prune=prune)
                        if not prune:
                            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)[:, start:end]
                    else:
                        logits = native(input_ids=x_t[:, -block_size:], use_cache=True,
                                        past_key_values=past_key_values, update_past_key_values=False).logits
                        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)[:, start:end]

                    x_1, p_1t = model.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                    x1_p = torch.gather(p_1t, -1, x_1.unsqueeze(-1)).squeeze(-1)
                    x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)
                    unmask_idx = x1_p > threshold
                    maximum = x1_p.argmax(dim=-1)
                    unmask_idx[torch.arange(x_1.shape[0], device=x_1.device), maximum] = True
                    unmask_idx &= mask_idx[:, start:end]
                    x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
        input_ids = x_t

    if stop_token in input_ids[:, original_input_length:]:
        stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
        input_ids = input_ids[:, :stop_token_idx + original_input_length + 1]
    _sync()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return DecodeResult(input_ids, seconds=elapsed, peak_gib=peak, **counters)
