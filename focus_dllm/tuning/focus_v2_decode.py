"""Free generation for isolated FOCUS-v2, with the native PrefixCache sampler.

The active-output-head control and both pooling ablations use the same warm
calls, formal prefix, full canvas and release rule. No EOS early stopping,
teacher, answer input or prepared-RoPE optimization is introduced here.
"""
import time

import torch
import torch.nn.functional as F

from ..llada_common import MASK_ID
from ..llada_decode import Result, _selected_positions, _sync
from .backend import selected_forward
from .focus_v2 import FocusV2Forward, ProxyConfig


@torch.no_grad()
def generate(model, prompt, *, gen_length=256, block_length=32, threshold=.90,
             config=None, trace=False):
    if prompt.ndim != 2 or prompt.shape[0] != 1 or not prompt.shape[1]:
        raise ValueError('Expected one nonempty prompt')
    if block_length <= 0 or gen_length <= 0 or gen_length % block_length:
        raise ValueError('Positive block length must divide the canvas')
    if config is not None and config.block_length != block_length:
        raise ValueError('Decoder/pooling block mismatch')
    forward = None if config is None else FocusV2Forward(model, config)
    canvas = torch.full((1, prompt.shape[1] + gen_length), MASK_ID,
                        dtype=torch.long, device=prompt.device)
    canvas[:, :prompt.shape[1]] = prompt
    by_block, actions = [], []
    if canvas.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    _sync()
    started = time.perf_counter()
    for offset in range(0, gen_length, block_length):
        start, end = prompt.shape[1] + offset, prompt.shape[1] + offset + block_length
        target = (canvas[0, start:end] == MASK_ID).nonzero().flatten() + start
        output = selected_forward(model, canvas, target, use_cache=True)
        past = [tuple(t[:, :, :start] for t in pair) for pair in output.past_key_values]
        logits = output.logits
        del output
        calls = 0
        while True:
            calls += 1
            tokens = logits.argmax(-1)
            confidence = F.softmax(logits.double(), dim=-1).gather(
                -1, tokens.unsqueeze(-1)).squeeze(-1)
            selected = _selected_positions(confidence, threshold)
            # Otherwise the unchanged native argmax fallback could loop forever.
            if not (tokens[0, selected] != MASK_ID).any():
                raise RuntimeError('The native release action made no progress')
            canvas[0, target[selected]] = tokens[0, selected]
            if trace:
                actions.append(dict(block=offset // block_length, call=calls,
                    positions=target[selected].tolist(), tokens=tokens[0, selected].tolist()))
            del logits, confidence
            if not (canvas[0, start:end] == MASK_ID).any():
                break
            if calls >= block_length:
                raise RuntimeError('Release calls exceeded the progress bound')
            local = (canvas[0, start:end] == MASK_ID).nonzero().flatten()
            suffix = canvas[:, start:]
            logits = (selected_forward(model, suffix, local, past_key_values=past,
                                       use_cache=True).logits if forward is None else
                      forward(suffix, local.tolist(), past_key_values=past))
            target = local + start
        by_block.append(calls)
    _sync()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**30 if canvas.is_cuda else 0.
    if not torch.equal(canvas[:, :prompt.shape[1]], prompt):
        raise RuntimeError('Prompt mutated')
    return Result(canvas, sum(by_block), elapsed, peak), dict(
        nfe_by_block=by_block, actions=actions if trace else None,
        refinement_calls=sum(by_block) - len(by_block), warm_calls=len(by_block),
        pooling_calls=[] if forward is None else forward.records)
