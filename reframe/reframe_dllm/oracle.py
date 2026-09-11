"""Held-out oracle diagnostics on native Fast-dLLM block-boundary states.

These probes use current full KV to fit pilots and cannot establish online
speed. They neither change baseline decoding nor count as benchmark timing.
"""
from contextlib import contextmanager
from types import MethodType

import torch

from .model import spaced_indices, split_pilots
from .transport import attention_lse, fit_transport, grouped_attention, relative_error


class OracleProbe:
    def __init__(self, emit, layers=(0, 7, 15, 31), pilots=(8, 16, 32),
                 ages=(1, 2, 4), kinds=("stale", "shift", "scale", "pair"), ridge=1e-3):
        self.emit, self.layers, self.pilots = emit, set(layers), pilots
        self.ages, self.kinds, self.ridge = ages, kinds, ridge
        if not ages or min(ages) < 1:
            raise ValueError("Reference ages must be positive block counts")
        self.history = []

    @torch.no_grad()
    def observe(self, x, snapshots, block, start, end):
        for old_block, old_ids, old in self.history:
            age = block - old_block
            if age not in self.ages:
                continue
            same_identity = x[0] == old_ids[0]
            all_ids = torch.arange(x.shape[1], device=x.device)
            for layer, (q, k, v) in snapshots.items():
                k0, v0 = old[layer]
                for side, region in (("prefix", all_ids[:start]), ("suffix", all_ids[end:])):
                    region = region[same_identity[region]]
                    for n in self.pilots:
                        pilots = spaced_indices(region, n)
                        train, valid = split_pilots(pilots)
                        remaining = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
                        remaining[pilots] = False
                        heldout = region[remaining[region]]
                        if not train.numel() or not heldout.numel():
                            continue
                        # All non-held-out positions use true current KV so
                        # attention error isolates generalization to this side.
                        exact = torch.ones_like(remaining)
                        exact[heldout] = False
                        exact = all_ids[exact]
                        target = attention_lse(q, k, v)[0]
                        for kind in self.kinds:
                            t = fit_transport(k0[:, :, train], v0[:, :, train],
                                              k[:, :, train], v[:, :, train], kind, self.ridge)
                            groups = [(k0[:, :, heldout], v0[:, :, heldout], t),
                                      (k[:, :, exact], v[:, :, exact], None)]
                            transported = grouped_attention(q, groups)
                            materialized = grouped_attention(q, groups, materialize=True)
                            self.emit(dict(block=block, reference_block=old_block, age_blocks=age,
                                           layer=layer, side=side, kind=kind, pilots=n,
                                           train_tokens=train.numel(), heldout_tokens=heldout.numel(),
                                           key_error=relative_error(t.key(k0[:, :, heldout]), k[:, :, heldout]),
                                           value_error=relative_error(t.value(v0[:, :, heldout]), v[:, :, heldout]),
                                           attention_error=relative_error(transported, target),
                                           execution_error=relative_error(transported, materialized),
                                           safe_transform=t.is_safe(),
                                           oracle=True))
        # Native DualCache modifies its raw values in place; history owns copies.
        self.history.append((block, x.clone(), {layer: (k.clone(), v.clone())
                                               for layer, (_, k, v) in snapshots.items()}))
        self.history = [entry for entry in self.history if block - entry[0] < max(self.ages)]


class NativeOracleWrapper:
    """Duck-typed model wrapper for the repository's generate_with_dual_cache."""
    def __init__(self, model, probe, prompt_length, block_length):
        self.model, self.probe = model, probe
        self.prompt_length, self.block_length = prompt_length, block_length
        self.block = 0

    @property
    def device(self):
        return self.model.device

    @contextmanager
    def _capture(self, start, end, snapshots):
        originals = []
        try:
            for block in self.model.model.transformer.blocks:
                if block.layer_id not in self.probe.layers:
                    continue
                if "_scaled_dot_product_attention" in block.__dict__:
                    raise RuntimeError("Attention backend already overridden")
                original = block._scaled_dot_product_attention

                def capture(this, q, k, v, *args, _original=original, **kwargs):
                    snapshots[this.layer_id] = (q[:, :, start:end].clone(), k.clone(), v.clone())
                    return _original(q, k, v, *args, **kwargs)

                block._scaled_dot_product_attention = MethodType(capture, block)
                originals.append(block)
            yield
        finally:
            for block in originals:
                del block._scaled_dot_product_attention

    @torch.no_grad()
    def __call__(self, x, **kwargs):
        if kwargs.get("past_key_values") is not None:
            return self.model(x, **kwargs)
        start = self.prompt_length + self.block * self.block_length
        end = start + self.block_length
        captured = {}
        with self._capture(start, end, captured):
            result = self.model(x, **kwargs)
        self.probe.observe(x, captured, self.block, start, end)
        self.block += 1
        return result
