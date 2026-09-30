"""Refresh the completed block while reading older history from its cache."""
import time

import torch

from ..llada_common import MASK_ID
from ..llada_decode import Result, _sync, _selected_positions
from ..llada_pruning import Config
from .backend import selected_forward
from .zero_support import ZeroForward


@torch.no_grad()
def generate_stream(model, prompt, *, gen_length=256, layer=4,
                    threshold=.90, refresh_every=4, keep=0., trace=False,
                    forward_type=ZeroForward):
    if gen_length % 32 or refresh_every < 1:
        raise ValueError('Canvas must be divisible by 32; refresh interval >=1')
    if keep not in (0.,1.):
        raise ValueError('Streaming ablation supports keep=0 (freeze future queries) or keep=1 (full suffix)')
    x = torch.full((1, prompt.shape[1]+gen_length), MASK_ID,
                   dtype=torch.long, device=prompt.device)
    x[:, :prompt.shape[1]] = prompt
    forward = forward_type(model, Config(prune_after_layer=layer,
                          support_keep_ratio=keep, target_only_head=True))
    past, nfe, actions, warm_query_tokens = None, 0, [], 0
    if x.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    _sync()
    started = time.perf_counter()
    for block in range(gen_length//32):
        start, end = prompt.shape[1]+block*32, prompt.shape[1]+(block+1)*32
        warm_start = 0 if block % refresh_every == 0 else start-32
        warm_past = None if warm_start == 0 else past
        positions = (x[0, start:end] == MASK_ID).nonzero().flatten()+start
        output = selected_forward(model, x[:, warm_start:], positions-warm_start,
                                  past_key_values=warm_past, use_cache=True)
        forward.reference = output.past_key_values
        past = [tuple(t[:, :, :start] for t in pair) for pair in output.past_key_values]
        warm_query_tokens += x.shape[1]-warm_start
        logits = output.logits
        del output
        while True:
            nfe += 1
            tokens = logits.argmax(-1)
            probabilities = logits.double().softmax(-1).gather(-1,tokens.unsqueeze(-1)).squeeze(-1)
            take = _selected_positions(probabilities, threshold)
            x[0, positions[take]] = tokens[0,take]
            if trace:
                actions.append((positions[take].tolist(),tokens[0,take].tolist()))
            del logits, probabilities
            if not (x[0,start:end] == MASK_ID).any():
                break
            local = (x[0,start:end] == MASK_ID).nonzero().flatten()
            logits = forward(x[:,start:],local,past_key_values=past)
            positions = local+start
    _sync()
    elapsed = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated()/2**30 if x.is_cuda else 0.
    return Result(x,nfe,elapsed,peak), actions, warm_query_tokens
