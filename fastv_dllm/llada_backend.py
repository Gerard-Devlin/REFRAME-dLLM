"""Explicit torch-SDPA versus flash-attn execution for original LLaDA."""

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass


@dataclass
class Stats:
    attention_calls: int = 0
    flash_calls: int = 0
    torch_sdpa_calls: int = 0
    query_tokens: int = 0


class LLaDAAttentionBackend(AbstractContextManager):
    def __init__(self, model, name):
        if name not in {"torch", "flash"}:
            raise ValueError("backend must be torch or flash")
        self.model, self.name = model, name
        self.stats = Stats()
        self.saved = []

    def __enter__(self):
        flash = None
        if self.name == "flash":
            from flash_attn import flash_attn_func
            flash = flash_attn_func
        for block in self.model.model.transformer.blocks:
            original_method = block._scaled_dot_product_attention
            original_flash = block.flash_attn_func
            block.flash_attn_func = flash

            def counted(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                        _original=original_method, _block=block):
                self.stats.attention_calls += 1
                self.stats.query_tokens += int(q.shape[-2])
                uses_flash = _block.flash_attn_func is not None and attn_mask is None
                if uses_flash:
                    self.stats.flash_calls += 1
                else:
                    self.stats.torch_sdpa_calls += 1
                return _original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

            block._scaled_dot_product_attention = counted
            self.saved.append((block, original_method, original_flash))
        return self

    def __exit__(self, *exc):
        for block, method, flash in self.saved:
            block._scaled_dot_product_attention = method
            block.flash_attn_func = flash
        return False

    def report(self):
        return asdict(self.stats)
