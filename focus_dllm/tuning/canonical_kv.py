"""Isolate numerical changes caused by permuting retained attention keys.

The support policy, cached values, query positions and decoder stay unchanged.
This ablation puts K/V back in original token order before attention. It incurs
extra gathers; neither faster execution nor improved quality is assumed.
"""
from functools import partial

import torch

from .rotation import RotatedForward
from .static_support import StaticSupportForward
from .zero_support import ZeroForward


class CanonicalOrder:
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        saved = []
        # Every mapped layer in one call uses the same retained-position map.
        # Cache only within this call: dynamic support can change on the next.
        order = None
        for block in self.model.model.transformer.blocks:
            previous = block._scaled_dot_product_attention

            def attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=False,
                          _block=block, _previous=previous):
                nonlocal order
                positions = getattr(_block.rotary_emb, 'key_positions', None)
                if positions is not None:
                    if attn_mask is not None or is_causal:
                        raise ValueError('Canonical K/V ablation requires unmasked bidirectional attention')
                    if len(positions) != k.shape[-2] or k.shape[-2] != v.shape[-2]:
                        raise ValueError('Attention key map length mismatch')
                    if order is None:
                        order = positions.argsort()
                    k = k.index_select(-2, order)
                    v = v.index_select(-2, order)
                return _previous(q, k, v, attn_mask=attn_mask,
                                 dropout_p=dropout_p, is_causal=is_causal)

            block._scaled_dot_product_attention = attention
            saved.append((block, previous))
        try:
            return super().__call__(*args, **kwargs)
        finally:
            for block, previous in saved:
                block._scaled_dot_product_attention = previous


class CanonicalZero(CanonicalOrder, ZeroForward):
    pass


class CanonicalSupport(CanonicalOrder, StaticSupportForward):
    pass


class CanonicalReference(CanonicalOrder, RotatedForward):
    pass


def generate_canonical(model, prompt, *, kind, support=32, refresh=1, **kwargs):
    if kind == 'support':
        from .streaming import generate_stream
        return generate_stream(model, prompt, refresh_every=refresh,
                               forward_type=partial(CanonicalSupport, support_count=support),
                               **kwargs)[0]
    if kind not in {'zero', 'reference'}:
        raise ValueError('Unknown canonical K/V family')
    from . import tensor_pruning
    previous = tensor_pruning.ReferenceForward
    tensor_pruning.ReferenceForward = CanonicalZero if kind == 'zero' else CanonicalReference
    try:
        return tensor_pruning.generate_tensor(model, prompt, reference=True, **kwargs)[0]
    finally:
        tensor_pruning.ReferenceForward = previous


def main():
    # Reuse the existing prompt partition, scoring and paired timing contract.
    from . import conservative, retention_sweep
    from ..llada_backend import LLaDAAttentionBackend
    from ..llada_common import prompt_ids

    conservative.run.VARIANTS.update({
        'canonical_zero_l1': dict(kind='canonical_zero', layer=1, keep=0., threshold=.90),
        'canonical_support32_r1_l4': dict(kind='canonical_support', layer=4, keep=0.,
                                        threshold=.90, support=32, refresh=1),
        'canonical_support32_r2_l4': dict(kind='canonical_support', layer=4, keep=0.,
                                        threshold=.90, support=32, refresh=2),
    })
    previous = conservative.run_one

    def run_one(model, tokenizer, sample, config, gen_length):
        if not config['kind'].startswith('canonical_'):
            return previous(model, tokenizer, sample, config, gen_length)
        ids = prompt_ids(tokenizer, sample['paper_prompt'], 'gsm8k', preformatted=True)
        prompt = torch.tensor([ids], device=model.device)
        with LLaDAAttentionBackend(model, 'flash') as backend:
            result = generate_canonical(model, prompt, kind=config['kind'][10:],
                gen_length=gen_length, layer=config['layer'], keep=config['keep'],
                threshold=config['threshold'], support=config.get('support', 32),
                refresh=config.get('refresh', 1))
        tokens = result.output[0, len(ids):].tolist()
        return dict(token_ids=tokens, nfe=result.nfe, seconds=result.seconds,
                    peak_gib=result.peak_gib, backend=backend.report(),
                    truncated=126081 not in tokens, canvas_tokens=gen_length)

    conservative.run_one = run_one
    try:
        retention_sweep.main()
    finally:
        conservative.run_one = previous


if __name__ == '__main__':
    main()
