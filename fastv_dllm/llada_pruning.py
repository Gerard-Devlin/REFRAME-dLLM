"""FastV-style physical support pruning for the original LLaDA transformer."""

from dataclasses import dataclass
import math
import time

import torch
from torch import nn

from .llada_common import MASK_ID


def choose_support(relevance, targets, keep_ratio, candidates=None, protected=None):
    targets = sorted(set(int(x) for x in targets))
    target_set = set(targets)
    if candidates is None:
        candidates = [i for i in range(relevance.numel()) if i not in target_set]
    else:
        candidates = sorted(set(int(x) for x in candidates) - target_set)
    if protected is None:
        protected = []
    protected = sorted(set(int(x) for x in protected) - target_set - set(candidates))
    support = candidates
    count = min(len(support), max(0, math.ceil(len(support) * float(keep_ratio))))
    if count:
        candidates = torch.tensor(support, device=relevance.device)
        selected = candidates[torch.topk(relevance.index_select(0, candidates), count, sorted=False).indices].tolist()
    else:
        selected = []
    return sorted(targets + protected + selected)


class PositionedRotary(nn.Module):
    """Apply the original RoPE phases after non-contiguous physical pruning."""
    def __init__(self, base, positions):
        super().__init__()
        object.__setattr__(self, "base", base)
        self.length = max(positions) + 1
        self.register_buffer("positions", torch.tensor(positions, dtype=torch.long), persistent=False)

    def forward(self, q, k, block_end_index=None):
        base = object.__getattribute__(self, "base")
        qf, kf = (q.float(), k.float()) if base.config.rope_full_precision else (q, k)
        with torch.autocast(q.device.type, enabled=False):
            sin, cos = base.get_rotary_embedding(self.length, q.device)
            positions = self.positions.to(q.device)
            sin = sin.index_select(2, positions).type_as(qf)
            cos = cos.index_select(2, positions).type_as(qf)
            qf = base.apply_rotary_pos_emb(sin, cos, qf)
            kf = base.apply_rotary_pos_emb(sin, cos, kf)
        return qf.type_as(q), kf.type_as(k)


@dataclass(frozen=True)
class Config:
    prune_after_layer: int = 4
    support_keep_ratio: float = 0.5

    def validate(self, layers):
        if not 1 <= self.prune_after_layer < layers:
            raise ValueError("Prune point must leave at least one deep layer")
        if not 0 <= self.support_keep_ratio <= 1:
            raise ValueError("support_keep_ratio must be in [0,1]")


class LLaDABlockForward:
    def __init__(self, model, config):
        self.model, self.config = model, config
        config.validate(model.model.config.n_layers)
        self.records = []

    @staticmethod
    def _captured_block(block, hidden):
        capture = {}
        original = block._scaled_dot_product_attention

        def wrapped(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
            capture["q"], capture["k"] = q, k
            return original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

        block._scaled_dot_product_attention = wrapped
        try:
            output, _ = block(hidden, attention_bias=None, layer_past=None, use_cache=False)
        finally:
            block._scaled_dot_product_attention = original
        if set(capture) != {"q", "k"}:
            raise RuntimeError("Could not observe LLaDA attention projections")
        return output, capture

    @staticmethod
    def _relevance(capture, targets):
        q, k = capture["q"], capture["k"]
        target = torch.tensor(targets, device=q.device)
        q = q.index_select(-2, target)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        return scores.float().softmax(-1).mean(dim=(0, 1, 2))

    def __call__(self, input_ids, target_positions, prune=True):
        if input_ids.shape[0] != 1 or not target_positions:
            raise ValueError("LLaDA FastV pilot requires batch=1 and active target positions")
        target_set = set(target_positions)
        future_masks = [i for i, token in enumerate(input_ids[0].tolist())
                        if token == MASK_ID and i not in target_set]
        # No candidate can be removed in the final block (or when all support
        # is requested).  Use the exact upstream forward and avoid paying the
        # attention-capture/ranking overhead.
        if prune and (not future_masks or self.config.support_keep_ratio == 1):
            target = torch.tensor(target_positions, device=input_ids.device)
            return self.model(input_ids).logits.index_select(1, target)
        core = self.model.model
        hidden = core.transformer.wte(input_ids)
        if core.config.input_emb_norm:
            hidden = hidden * (core.config.d_model ** 0.5)
        hidden = core.transformer.emb_drop(hidden)
        positions = list(range(input_ids.shape[1]))
        layer_record = None
        for number, block in enumerate(core.transformer.blocks, start=1):
            if number == self.config.prune_after_layer:
                hidden, capture = self._captured_block(block, hidden)
                measure_score = not prune
                if measure_score and hidden.is_cuda:
                    torch.cuda.synchronize(hidden.device)
                started = time.perf_counter()
                relevance = self._relevance(capture, target_positions)
                # FastV protects every text token and prunes only its redundant
                # modality.  For LLaDA the corresponding redundant class is the
                # untouched future MASK canvas.  Prompt and already revealed
                # language tokens must never compete with those masks.
                protected = [i for i in positions if i not in target_set and i not in set(future_masks)]
                candidate_keep = choose_support(
                    relevance, target_positions, self.config.support_keep_ratio,
                    candidates=future_masks, protected=protected,
                )
                keep = candidate_keep if prune else positions
                if measure_score and hidden.is_cuda:
                    torch.cuda.synchronize(hidden.device)
                support = future_masks
                kept_support = [i for i in candidate_keep if i in set(future_masks)]
                denominator = float(relevance[support].sum().item()) if support else 0.0
                layer_record = dict(
                    layer=number, original_tokens=len(positions), targets=len(set(target_positions)),
                    support=len(support), protected=len(protected), kept_support=len(kept_support),
                    deep_tokens=len(candidate_keep),
                    retained_support_mass=(float(relevance[kept_support].sum().item()) / denominator if denominator else 1.0),
                    score_seconds=(time.perf_counter() - started if measure_score else 0.0),
                )
                if prune:
                    index = torch.tensor(keep, device=hidden.device)
                    hidden = hidden.index_select(1, index)
                    positions = keep
            else:
                rotary = block.rotary_emb
                if len(positions) != input_ids.shape[1]:
                    block.rotary_emb = PositionedRotary(rotary, positions).to(hidden.device)
                try:
                    hidden, _ = block(hidden, attention_bias=None, layer_past=None, use_cache=False)
                finally:
                    block.rotary_emb = rotary
        hidden = core.transformer.ln_f(hidden)
        compact = {position: index for index, position in enumerate(positions)}
        gather = torch.tensor([compact[int(position)] for position in target_positions], device=hidden.device)
        if core.config.weight_tying:
            logits = torch.nn.functional.linear(hidden, core.transformer.wte.weight)
        else:
            logits = core.transformer.ff_out(hidden)
        if core.config.scale_logits:
            logits.mul_(1 / math.sqrt(core.config.d_model))
        # Compute the head for every physically retained state. This avoids
        # crediting a separate target-only LM-head optimization to FastV.
        logits = logits.index_select(1, gather)
        self.records.append(layer_record)
        return logits
