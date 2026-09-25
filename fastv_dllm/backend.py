"""Measured SDPA/FlashAttention backend switch for the pinned remote model.

The checkpoint hard-codes ``ALL_ATTENTION_FUNCTIONS['sdpa']``.  Installing
flash-attn therefore does not change its execution.  This context patches that
single registry entry and records every true FlashAttention call and fallback.
"""

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass

import torch


@dataclass
class BackendStats:
    calls: int = 0
    flash_calls: int = 0
    fallback_calls: int = 0
    verified_masks: int = 0
    query_tokens: int = 0


class AttentionBackend(AbstractContextManager):
    def __init__(self, name, block_size=32, verify_masks=True):
        if name not in {"sdpa", "flash"}:
            raise ValueError(f"Unknown attention backend: {name}")
        self.name = name
        self.block_size = int(block_size)
        self.verify_masks = bool(verify_masks)
        self.stats = BackendStats()
        self._verified = set()
        self._registry = None
        self._original = None

    def __enter__(self):
        if self.name == "sdpa":
            return self
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from flash_attn import flash_attn_func

        self._registry = ALL_ATTENTION_FUNCTIONS
        self._original = ALL_ATTENTION_FUNCTIONS["sdpa"]

        def dispatch(module, query, key, value, attention_mask, **kwargs):
            self.stats.calls += 1
            self.stats.query_tokens += int(query.shape[-2])
            qlen, klen = int(query.shape[-2]), int(key.shape[-2])
            full_visibility = attention_mask is None
            # For v2's newest 32-token block, cached history ends on a block
            # boundary and eval_block_diff_mask is identically true.
            semantic_full = qlen <= self.block_size and (klen - qlen) % self.block_size == 0
            if not full_visibility and semantic_full and self.verify_masks:
                shape = (qlen, klen)
                if shape not in self._verified:
                    full_visibility = bool(attention_mask.all().item())
                    if full_visibility:
                        self._verified.add(shape)
                        self.stats.verified_masks += 1
                else:
                    full_visibility = True
            elif not full_visibility and semantic_full:
                full_visibility = True
            sliding = kwargs.get("sliding_window")
            if (not full_visibility or query.device.type != "cuda" or query.dtype not in (torch.float16, torch.bfloat16)
                    or sliding not in (None, -1)):
                self.stats.fallback_calls += 1
                return self._original(module, query, key, value, attention_mask, **kwargs)
            output = flash_attn_func(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                dropout_p=float(kwargs.get("dropout", 0.0)),
                softmax_scale=float(kwargs.get("scaling", module.scaling)),
                causal=False,
            )
            self.stats.flash_calls += 1
            return output, None

        ALL_ATTENTION_FUNCTIONS["sdpa"] = dispatch
        return self

    def __exit__(self, *exc):
        if self._registry is not None:
            self._registry["sdpa"] = self._original
        return False

    def report(self):
        return asdict(self.stats)
