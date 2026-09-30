"""Replace a focused refinement with a full suffix refresh within a block.

The native commit rule is unchanged. Refreshes replace, rather than add, model
calls and never overwrite the persistent history cache. This is a development
ablation of stale future K/V, not a claim of lossless generation.
"""
import torch

from .backend import selected_forward
from .static_support import StaticSupportForward


class RefreshingSupport(StaticSupportForward):
    def __init__(self, model, config, support_count=32, period=0, stall_limit=0):
        super().__init__(model, config, support_count=support_count)
        if period < 0 or stall_limit < 0 or not (period or stall_limit):
            raise ValueError('Enable a nonnegative refresh period or stall limit')
        self.period, self.stall_limit = period, stall_limit
        self.refresh_source = None
        self.refresh_calls = self.focused_calls = self.tail_calls = 0

    @torch.no_grad()
    def __call__(self, ids, positions, past_key_values=None, **kwargs):
        # The generator installs a new reference at every block boundary.
        # Internally replacing the reference must not reset the block clock.
        if self.reference is not self.refresh_source:
            self.refresh_source = self.reference
            self.block_calls = self.stalled = 0
            self.previous_count = None
            self.last_focused = False
        count = len(positions)
        if self.previous_count is not None and self.last_focused:
            if count > self.previous_count:
                raise ValueError('Active MASK count increased inside a block')
            self.stalled = self.stalled + 1 if self.previous_count - count <= 1 else 0
        self.block_calls += 1
        self.previous_count = count
        due = ((self.period and self.block_calls % self.period == 0) or
               (self.stall_limit and self.stalled >= self.stall_limit))
        if ids.shape[1] <= 32:
            self.last_focused = False
            self.tail_calls += 1
            return super().__call__(ids, positions, past_key_values=past_key_values, **kwargs)
        if due:
            target = torch.as_tensor(positions, device=ids.device)
            output = selected_forward(self.model, ids, target,
                                      past_key_values=past_key_values, use_cache=True)
            # Only the temporary full suffix snapshot changes. The caller's
            # clean prefix remains read-only, as in an ordinary refinement.
            self.reference = self.refresh_source = output.past_key_values
            self.stalled = 0
            self.last_focused = False
            self.refresh_calls += 1
            return output.logits
        self.last_focused = True
        self.focused_calls += 1
        return super().__call__(ids, positions, past_key_values=past_key_values, **kwargs)


def generate_refresh(model, prompt, *, support=32, period=0, stall_limit=0, **kwargs):
    from .streaming import generate_stream
    forwards = []

    def create(model, config):
        forward = RefreshingSupport(model, config, support_count=support,
                                    period=period, stall_limit=stall_limit)
        forwards.append(forward)
        return forward

    result, actions, queries = generate_stream(model, prompt, refresh_every=1,
                                               forward_type=create, **kwargs)
    forward = forwards[0]
    counters = dict(within_refresh_calls=forward.refresh_calls,
                    focused_calls=forward.focused_calls, tail_calls=forward.tail_calls,
                    warm_query_tokens=queries)
    return result, actions, counters


CONFIGS = {
    'within_period4_s32_l4': dict(kind='within_refresh', layer=4, keep=0.,
        threshold=.90, support=32, period=4, stall_limit=0),
    'within_stall2_s32_l4': dict(kind='within_refresh', layer=4, keep=0.,
        threshold=.90, support=32, period=0, stall_limit=2),
}


def main():
    from . import conservative, retention_sweep
    from ..llada_backend import LLaDAAttentionBackend
    from ..llada_common import prompt_ids

    conservative.run.VARIANTS.update(CONFIGS)
    previous = conservative.run_one

    def run_one(model, tokenizer, sample, config, gen_length):
        if config['kind'] != 'within_refresh':
            return previous(model, tokenizer, sample, config, gen_length)
        ids = prompt_ids(tokenizer, sample['paper_prompt'], 'gsm8k', preformatted=True)
        with LLaDAAttentionBackend(model, 'flash') as backend:
            result, _, counters = generate_refresh(model,
                torch.tensor([ids], device=model.device), gen_length=gen_length,
                layer=config['layer'], keep=config['keep'], threshold=config['threshold'],
                support=config['support'], period=config['period'], stall_limit=config['stall_limit'])
        tokens = result.output[0, len(ids):].tolist()
        return dict(token_ids=tokens, nfe=result.nfe, seconds=result.seconds,
                    peak_gib=result.peak_gib, backend=backend.report(), **counters,
                    truncated=126081 not in tokens, canvas_tokens=gen_length)

    conservative.run_one = run_one
    try:
        retention_sweep.main()
    finally:
        conservative.run_one = previous


if __name__ == '__main__':
    main()
