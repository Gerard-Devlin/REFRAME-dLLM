"""Observe actual adjacent native forwards, never synthesize teacher futures."""

import random
import time

import torch

from .data import coarsen


def shifted(x):
    return torch.cat((x[:, :1], x[:, :-1]), dim=1)


def current_candidates(logits, k):
    """Always include greedy current argmax first, even for a top-K tie."""
    best = logits.argmax(-1, keepdim=True)
    candidates = logits.topk(k, dim=-1).indices
    # Move the already-included argmax to the end before taking K-1 others.
    order = (candidates == best).to(torch.int32).argsort(dim=-1, stable=True)
    rest = candidates.gather(-1, order)[..., :k-1]
    return torch.cat((best, rest), dim=-1)


def active_mask(ids, mask_id, small_block_size):
    masked = ids == mask_id
    positions = torch.arange(ids.shape[-1], device=ids.device)
    first = torch.where(masked, positions, ids.shape[-1]).min(-1).values
    return masked & (positions // small_block_size == first[:, None] // small_block_size)


def transition(previous, current, mask_id):
    """Accept monotone reveals only, without any already-filled token changing."""
    if previous.shape != current.shape:
        return None
    changed = previous != current
    if not bool(changed.any()) or bool((changed & (previous != mask_id)).any()):
        return None
    return changed & (current != mask_id)


class NativeObserver:
    """A wrapper around official model.forward; official generate stays untouched.

    We reset on prefill/cache writes. A training pair is two successive normal
    full-block calls in the same cache interval, connected by observed reveals.
    """

    def __init__(self, model, *, block_size=32, small_block_size=8, mask_id=151665,
                 stop_id=151645, threshold=0.9, top_k=16, max_pairs=32, seed=1234):
        self.model = model
        self.block_size, self.small_block_size = block_size, small_block_size
        self.mask_id, self.stop_id, self.threshold = mask_id, stop_id, threshold
        self.top_k, self.max_pairs = top_k, max_pairs
        self.rng = random.Random(seed)
        self.previous = None
        self.records = []
        self.seen = self.calls = 0
        self.hidden = None
        self.trace = []
        self.capture_trace = False
        self.original = None

    def __enter__(self):
        self.original = self.model.forward
        # The pinned LM head receives final normalized hidden states; this hook
        # neither requests extra layers nor changes the official return value.
        self.hook = self.model.get_output_embeddings().register_forward_pre_hook(self._capture_hidden)
        self.model.forward = self.forward
        return self

    def __exit__(self, *_):
        self.model.forward = self.original
        self.hook.remove()

    def _capture_hidden(self, module, args):
        self.hidden = args[0].detach()

    def forward(self, *args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        self.calls += 1
        if self.capture_trace:
            self.trace.append((ids.detach().cpu().tolist(), kwargs.get("update_past_key_values"),
                               kwargs.get("use_block_cache"), kwargs.get("replace_position")))
        normal = (ids is not None and ids.shape == (1, self.block_size)
                  and kwargs.get("update_past_key_values") is False
                  and not kwargs.get("use_block_cache", False))
        self.hidden = None
        if normal and ids.is_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        started = time.perf_counter()
        output = self.original(*args, **kwargs)
        if normal and ids.is_cuda:
            end.record()
            end.synchronize()
            seconds = start.elapsed_time(end) / 1000
        else:
            seconds = time.perf_counter() - started
        if not normal:
            self.previous = None
            self.hidden = None
            return output
        if self.hidden is None or self.hidden.shape[:2] != ids.shape:
            raise RuntimeError("Pinned model LM-head hidden capture failed; stop before collecting")
        logits, hidden = shifted(output.logits), shifted(self.hidden)
        previous = self.previous
        committed = transition(previous["state"], ids, self.mask_id) if previous is not None else None
        eligible = active_mask(ids, self.mask_id, self.small_block_size)
        if committed is not None and bool(eligible.any()) and not bool((ids == self.stop_id).any()):
            teacher_probs = coarsen(logits, previous["candidates"])
            teacher_top1 = logits.argmax(-1)
            # Use the same BF16 native probabilities and forced-argmax rule as
            # the real decoder for the next-step commit-selection diagnostic.
            native_ids, native_probs = self.model.sample_with_top_p(logits, top_p=0.95, temperature=0)
            confidence = native_probs.gather(-1, native_ids.unsqueeze(-1)).squeeze(-1)
            scores = torch.where(eligible, confidence, -torch.inf)
            selected = (scores > self.threshold) & eligible
            selected[torch.arange(len(ids), device=ids.device), scores.argmax(-1)] = True
            record = dict(hidden=previous["hidden"][0], candidates=previous["candidates"][0],
                          base_log_probs=previous["base_log_probs"][0],
                          commit_ids=torch.where(committed, ids, 0)[0], committed=committed[0],
                          eligible=eligible[0], teacher_probs=teacher_probs[0],
                          teacher_top1=teacher_top1[0], teacher_selected=selected[0],
                          teacher_forward_seconds=torch.tensor(seconds))
            # Deterministic reservoir: do not keep only the earliest transitions.
            self.seen += 1
            replace = self.seen - 1 if self.seen <= self.max_pairs else self.rng.randrange(self.seen)
            if replace < self.max_pairs:
                saved = {key: value.detach().cpu().clone() for key, value in record.items()}
                if replace < len(self.records):
                    self.records[replace] = saved
                else:
                    self.records.append(saved)
        candidates = current_candidates(logits, self.top_k)
        self.previous = dict(state=ids.detach().clone(), hidden=hidden.detach().clone(),
                             candidates=candidates, base_log_probs=coarsen(logits, candidates).log())
        self.hidden = None
        return output
