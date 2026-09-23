"""Fast-dLLM v2 greedy decoder with a single instrumented commit decision.

Derived from the pinned Fast_dLLM_v2_1.5B modeling.py `generate` method
(Apache-2.0, NVIDIA CORPORATION & AFFILIATES, 2025). Only the decision after
`unmask_idx &= mask_idx[:, start:end]` is extended. Prefill, token shift,
sub-blocks, caches, stop handling, and the number of model calls are retained.
"""

import torch

from .budget import decide


@torch.no_grad()
def generate(
    self,
    input_ids,
    max_new_tokens,
    mask_id=151665,
    threshold=1,
    small_block_size=8,
    block_size=32,
    stop_token=151645,
    stopping_criteria=None,
    top_p=0.95,
    temperature=0,
    use_block_cache=False,
    block_cache_refresh_interval=16,
    *,
    policy="native",
    margin=0.0,
    observer=None,
    **kwargs,
):
    if policy not in ("native", "observe", "budget"):
        raise ValueError(f"Unknown policy: {policy}")
    if policy != "native" and temperature != 0:
        raise ValueError("Competitor certificate requires greedy temperature=0")
    if max_new_tokens < block_size or max_new_tokens % block_size or block_size % small_block_size:
        raise ValueError("Use whole output blocks and a sub-block divisor")
    num_blocks = max_new_tokens // block_size
    original_input_length = input_ids.shape[1]

    if input_ids.shape[1] > block_size:
        output = self.forward(input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)],
                              use_cache=True, update_past_key_values=True, block_size=block_size)
        logits, past_key_values = output.logits, output.past_key_values
        if input_ids.shape[1] % block_size == 0:
            next_token = logits[:, -1:, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    else:
        past_key_values = None

    num_small_blocks = block_size // small_block_size
    for block_idx in range(num_blocks):
        if stop_token in input_ids[:, original_input_length:]:
            break
        prompt_length = input_ids.shape[1]
        x_init = mask_id * torch.ones((input_ids.shape[0], block_size - prompt_length % block_size),
                                      device=self.device, dtype=torch.long)
        x_init = torch.cat([input_ids, x_init], dim=1)
        x_t = x_init.clone()
        block_past_key_values = None
        while True:
            if stop_token in x_t[:, prompt_length:]:
                stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                if (x_t[:, prompt_length:prompt_length + stop_token_idx] == mask_id).sum() == 0:
                    break
            mask_idx = (x_t[:, -block_size:] == mask_id)
            if mask_idx.sum() == 0:
                output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True,
                                      past_key_values=past_key_values, update_past_key_values=True,
                                      block_size=block_size)
                logits, past_key_values = output.logits, output.past_key_values
                next_token = logits[:, -1:, :].argmax(dim=-1)
                x_t = torch.cat([x_t, next_token], dim=1)
                break
            for small_block_idx in range(num_small_blocks):
                small_block_start_idx = small_block_idx * small_block_size
                small_block_end_idx = small_block_start_idx + small_block_size
                start = -block_size + small_block_start_idx
                end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                while True:
                    mask_idx = (x_t[:, -block_size:] == mask_id)
                    if mask_idx[:, start:end].sum() == 0:
                        break
                    if stop_token in x_t[:, prompt_length:]:
                        stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                        if (x_t[:, prompt_length:prompt_length + stop_token_idx] == mask_id).sum() == 0:
                            break
                    if use_block_cache:
                        if block_past_key_values is None or (x_t[:, -block_size + small_block_start_idx] == mask_id).any():
                            output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True,
                                                  past_key_values=past_key_values, update_past_key_values=False,
                                                  use_block_cache=True)
                            logits, block_past_key_values = output.logits, output.block_past_key_values
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]
                        else:
                            logits = self.forward(input_ids=x_t[:, start:end], use_cache=True,
                                                  past_key_values=past_key_values, update_past_key_values=False,
                                                  use_block_cache=True, block_past_key_values=block_past_key_values,
                                                  replace_position=small_block_start_idx).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                    else:
                        logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=True,
                                              past_key_values=past_key_values, update_past_key_values=False).logits
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]

                    x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                    x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                    x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)
                    unmask_idx = (x1_p > threshold)
                    max_prob_idx = x1_p.argmax(dim=-1)
                    unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                    unmask_idx = unmask_idx & mask_idx[:, start:end]

                    if policy != "native":
                        decision = decide(p_1t, x_1, mask_idx[:, start:end], threshold, margin,
                                          include_top1_bound=observer is not None)
                        if not torch.equal(decision.baseline, unmask_idx):
                            raise AssertionError("Commit logic differs from the pinned v2 decoder")
                        if observer is not None:
                            offset = x_t.shape[1] + start
                            for row in range(x_1.shape[0]):
                                positions = mask_idx[row, start:end].nonzero().flatten()
                                extras = (decision.proposed[row] & ~unmask_idx[row]).nonzero().flatten()
                                observer(dict(
                                    row=row, block=block_idx, subblock=small_block_idx,
                                    positions=(positions + offset).tolist(),
                                    top1=decision.top1[row, positions].tolist(),
                                    top2=decision.top2[row, positions].tolist(),
                                    baseline=(unmask_idx[row].nonzero().flatten() + offset).tolist(),
                                    proposed=(decision.proposed[row].nonzero().flatten() + offset).tolist(),
                                    plusplus=(decision.plusplus[row].nonzero().flatten() + offset).tolist(),
                                    extras=[dict(position=int(offset + pos), token=int(x_1[row, pos]))
                                            for pos in extras],
                                ))
                        if policy == "budget":
                            unmask_idx = decision.proposed

                    x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
        input_ids = x_t

    if stop_token in input_ids[:, original_input_length:]:
        stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
        input_ids = input_ids[:, :stop_token_idx + original_input_length + 1]
    return input_ids
