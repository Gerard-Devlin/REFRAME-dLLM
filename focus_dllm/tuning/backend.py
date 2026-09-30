"""Project only consumed positions, retaining the original Transformer and KV."""
from contextlib import contextmanager
import time

import torch
import torch.nn.functional as F

from ..llada_common import MASK_ID
from ..llada_decode import Result, _selected_positions, _sync
from ..llada_pruning import Config, LLaDABlockForward


@contextmanager
def active_output(model, target):
    # Norm is token-local. All Transformer layers and cache writes finish first.
    # This also handles checkpoints with tied embedding/output weights.
    norm = model.model.transformer.ln_f
    handle = norm.register_forward_pre_hook(
        lambda _module, args: (args[0].index_select(1, target),)
    )
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def selected_forward(model, ids, target, **kwargs):
    with active_output(model, target):
        return model(ids, **kwargs)


class ActiveForward(LLaDABlockForward):
    """Keep the current pruning rule, optimize its exact final-block path."""
    def __call__(self, ids, positions, prune=True, past_key_values=None,
                 use_cache=False, **kwargs):
        target = torch.as_tensor(positions, device=ids.device)
        if self.config.support_keep_ratio == 1 or ids.shape[1] <= 32:
            return selected_forward(self.model, ids, target,
                                    past_key_values=past_key_values,
                                    use_cache=use_cache).logits
        return super().__call__(ids, positions, prune=prune,
                                past_key_values=past_key_values,
                                use_cache=use_cache, **kwargs)


@torch.no_grad()
def generate_active_prefix(model, prompt, *, gen_length=256, block_length=32,
                           threshold=.90, layer=4, keep=.5, pruning=True,
                           trace=False):
    if gen_length % block_length or prompt.shape[0] != 1:
        raise ValueError('Expected batch=1 and divisible canvas length')
    x = torch.full((1, prompt.shape[1] + gen_length), MASK_ID,
                   device=prompt.device, dtype=torch.long)
    x[:, :prompt.shape[1]] = prompt
    forward = ActiveForward(model, Config(prune_after_layer=layer,
                            support_keep_ratio=keep, target_only_head=True))
    nfe, actions = 0, []
    if x.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    _sync()
    started = time.perf_counter()
    for offset in range(0, gen_length, block_length):
        start, end = prompt.shape[1] + offset, prompt.shape[1] + offset + block_length
        target = (x[0, start:end] == MASK_ID).nonzero().flatten() + start
        output = selected_forward(model, x, target, use_cache=True)
        past = [tuple(t[:, :, :start] for t in pair) for pair in output.past_key_values]
        logits = output.logits
        del output
        while True:
            nfe += 1
            tokens = logits.argmax(-1)
            confidence = F.softmax(logits.double(), dim=-1).gather(
                -1, tokens.unsqueeze(-1)).squeeze(-1)
            selected = _selected_positions(confidence, threshold)
            x[0, target[selected]] = tokens[0, selected]
            if trace:
                actions.append((target[selected].tolist(), tokens[0, selected].tolist()))
            del logits, confidence
            if not (x[0, start:end] == MASK_ID).any():
                break
            local = (x[0, start:end] == MASK_ID).nonzero().flatten()
            suffix = x[:, start:]
            if pruning and keep < 1:
                logits = forward(suffix, local.tolist(), past_key_values=past,
                                 use_cache=False)
            else:
                logits = selected_forward(model, suffix, local,
                                          past_key_values=past, use_cache=False).logits
            target = local + start
    _sync()
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**30 if x.is_cuda else 0.
    return Result(x, nfe, seconds, peak), actions
