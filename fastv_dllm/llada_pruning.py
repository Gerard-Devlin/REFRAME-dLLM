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


def compress_context(hidden, relevance, context, dominant_ratio, contextual_ratio,
                     merge_weight, assignment="cosine"):
    """VisionZip-style dominant retention plus contextual aggregation.

    ``context`` contains real language positions only. Dominant positions are
    selected by target-to-context attention. The rest are assigned to evenly
    distributed contextual anchors by hidden-state cosine similarity. Each
    anchor receives a scale-preserving interpolation with its cluster mean.
    """
    context = sorted(set(int(x) for x in context))
    if not context:
        return [], {}, 0, 0
    dominant_count = min(len(context), math.ceil(len(context) * dominant_ratio))
    context_tensor = torch.tensor(context, device=hidden.device)
    if dominant_count:
        dominant = context_tensor[
            torch.topk(relevance.index_select(0, context_tensor), dominant_count,
                       sorted=False).indices
        ].tolist()
    else:
        dominant = []
    dominant_set = set(dominant)
    remaining = [position for position in context if position not in dominant_set]
    contextual_count = min(len(remaining), math.ceil(len(context) * contextual_ratio))
    if not contextual_count:
        return sorted(dominant), {}, len(dominant), 0

    if assignment not in {"cosine", "spatial"}:
        raise ValueError("assignment must be cosine or spatial")
    # Spatially distributed anchors preserve coverage of the ordered language
    # sequence. Real text can use semantic assignment; homogeneous future MASK
    # support uses much cheaper contiguous pooling.
    states = hidden[0].index_select(
        0, torch.tensor(remaining, device=hidden.device)
    ).float()
    if assignment == "spatial":
        groups = torch.tensor_split(torch.arange(len(remaining), device=hidden.device),
                                    contextual_count)
        anchors = [remaining[int(group[len(group) // 2].item())] for group in groups]
        merged = {}
        for anchor, group in zip(anchors, groups):
            anchor_state = hidden[0, anchor].float()
            mean = states.index_select(0, group).mean(dim=0)
            merged[anchor] = (
                (1.0 - merge_weight) * anchor_state + merge_weight * mean
            ).to(hidden.dtype)
        return sorted(dominant + anchors), merged, len(dominant), len(anchors)

    if contextual_count == 1:
        anchor_offsets = [len(remaining) // 2]
    else:
        anchor_offsets = torch.linspace(
            0, len(remaining) - 1, contextual_count, device=hidden.device
        ).round().long().tolist()
    anchors = [remaining[offset] for offset in anchor_offsets]
    anchor_tensor = torch.tensor(anchors, device=hidden.device)
    anchor_states = hidden[0].index_select(0, anchor_tensor).float()
    similarity = torch.nn.functional.normalize(states, dim=-1) @ torch.nn.functional.normalize(
        anchor_states, dim=-1
    ).transpose(0, 1)
    assignment = similarity.argmax(dim=-1)
    merged = {}
    for number, anchor in enumerate(anchors):
        members = states[assignment == number]
        mean = members.mean(dim=0) if members.numel() else anchor_states[number]
        value = (1.0 - merge_weight) * anchor_states[number] + merge_weight * mean
        merged[anchor] = value.to(hidden.dtype)
    return sorted(dominant + anchors), merged, len(dominant), len(anchors)


class PositionedRotary(nn.Module):
    """Apply the original RoPE phases after non-contiguous physical pruning."""
    def __init__(self, base, positions, original_length, past_length=0):
        super().__init__()
        object.__setattr__(self, "base", base)
        self.length = past_length + original_length
        # The same compact-to-original mapping is reused by every remaining
        # layer.  Keep it on the GPU instead of rebuilding and transferring a
        # new tensor for each layer and denoising call.
        current = torch.as_tensor(positions, dtype=torch.long) + past_length
        keys = current
        if past_length:
            prefix = torch.arange(past_length, device=current.device, dtype=torch.long)
            keys = torch.cat((prefix, current))
        self.register_buffer("query_positions", current, persistent=False)
        self.register_buffer("key_positions", keys, persistent=False)

    def forward(self, q, k, block_end_index=None):
        base = object.__getattribute__(self, "base")
        qf, kf = (q.float(), k.float()) if base.config.rope_full_precision else (q, k)
        with torch.autocast(q.device.type, enabled=False):
            sin, cos = base.get_rotary_embedding(self.length, q.device)
            qsin = sin.index_select(2, self.query_positions).type_as(qf)
            qcos = cos.index_select(2, self.query_positions).type_as(qf)
            ksin = sin.index_select(2, self.key_positions).type_as(kf)
            kcos = cos.index_select(2, self.key_positions).type_as(kf)
            qf = base.apply_rotary_pos_emb(qsin, qcos, qf)
            kf = base.apply_rotary_pos_emb(ksin, kcos, kf)
        return qf.type_as(q), kf.type_as(k)


@dataclass(frozen=True)
class Config:
    prune_after_layer: int = 4
    support_keep_ratio: float = 0.5
    context_dominant_ratio: float = 1.0
    contextual_ratio: float = 0.0
    support_contextual_ratio: float = 0.0
    context_merge_weight: float = 0.5
    secondary_prune_after_layer: int = 0
    secondary_support_ratio: float = 1.0
    target_only_head: bool = False

    def validate(self, layers):
        if not 1 <= self.prune_after_layer < layers:
            raise ValueError("Prune point must leave at least one deep layer")
        if not 0 <= self.support_keep_ratio <= 1:
            raise ValueError("support_keep_ratio must be in [0,1]")
        if not 0 <= self.context_dominant_ratio <= 1:
            raise ValueError("context_dominant_ratio must be in [0,1]")
        if not 0 <= self.contextual_ratio <= 1:
            raise ValueError("contextual_ratio must be in [0,1]")
        if not 0 <= self.support_contextual_ratio <= 1:
            raise ValueError("support_contextual_ratio must be in [0,1]")
        if not 0 <= self.context_merge_weight <= 1:
            raise ValueError("context_merge_weight must be in [0,1]")
        if self.secondary_prune_after_layer:
            if not self.prune_after_layer < self.secondary_prune_after_layer < layers:
                raise ValueError("secondary prune point must follow the primary point")
        if not 0 <= self.secondary_support_ratio <= 1:
            raise ValueError("secondary_support_ratio must be in [0,1]")


class LLaDABlockForward:
    def __init__(self, model, config):
        self.model, self.config = model, config
        config.validate(model.model.config.n_layers)
        self.records = []

    @staticmethod
    def _captured_block(block, hidden, layer_past=None, use_cache=False):
        capture = {}
        original = block._scaled_dot_product_attention

        def wrapped(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
            capture["q"], capture["k"] = q, k
            return original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

        block._scaled_dot_product_attention = wrapped
        try:
            output, present = block(
                hidden, attention_bias=None, layer_past=layer_past, use_cache=use_cache
            )
        finally:
            block._scaled_dot_product_attention = original
        if set(capture) != {"q", "k"}:
            raise RuntimeError("Could not observe LLaDA attention projections")
        return output, capture, present

    @staticmethod
    def _relevance(capture, targets):
        q, k = capture["q"], capture["k"]
        target = torch.tensor(targets, device=q.device)
        q = q.index_select(-2, target)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        return scores.float().softmax(-1).mean(dim=(0, 1, 2))

    def __call__(self, input_ids, target_positions, prune=True,
                 past_key_values=None, use_cache=False, replace_position=None,
                 protected_prefix_length=0):
        if input_ids.shape[0] != 1 or not target_positions:
            raise ValueError("LLaDA FastV pilot requires batch=1 and active target positions")
        target_set = set(target_positions)
        future_masks = [i for i, token in enumerate(input_ids[0].tolist())
                        if token == MASK_ID and i not in target_set]
        mode = "mask" if prune is True else prune
        exact_zip = (mode == "zip" and self.config.context_dominant_ratio == 1
                     and self.config.contextual_ratio == 0
                     and self.config.support_keep_ratio == 1
                     and (not self.config.secondary_prune_after_layer
                          or self.config.secondary_support_ratio == 1))
        # No candidate can be removed in the final block (or when all support
        # is requested).  Use the exact upstream forward and avoid paying the
        # attention-capture/ranking overhead.
        if mode == "mask" and (not future_masks or self.config.support_keep_ratio == 1):
            target = torch.tensor(target_positions, device=input_ids.device)
            return self.model(
                input_ids, past_key_values=past_key_values, use_cache=use_cache,
                replace_position=replace_position,
            ).logits.index_select(1, target)
        if mode == "zip" and (not future_masks or exact_zip):
            target = torch.tensor(target_positions, device=input_ids.device)
            return self.model(
                input_ids, past_key_values=past_key_values, use_cache=use_cache,
                replace_position=replace_position,
            ).logits.index_select(1, target)
        core = self.model.model
        hidden = core.transformer.wte(input_ids)
        if core.config.input_emb_norm:
            hidden = hidden * (core.config.d_model ** 0.5)
        hidden = core.transformer.emb_drop(hidden)
        positions = list(range(input_ids.shape[1]))
        compact_positions = None
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        layer_record = None
        for number, block in enumerate(core.transformer.blocks, start=1):
            layer_past = None if past_key_values is None else past_key_values[number - 1]
            if number == self.config.prune_after_layer:
                needs_relevance = not (
                    mode == "zip"
                    and self.config.support_keep_ratio == 0
                    and self.config.context_dominant_ratio == 1
                )
                if needs_relevance:
                    hidden, capture, _ = self._captured_block(
                        block, hidden, layer_past=layer_past, use_cache=use_cache
                    )
                    relevance = self._relevance(capture, target_positions)[past_length:]
                else:
                    hidden, _ = block(
                        hidden, attention_bias=None, layer_past=layer_past, use_cache=use_cache
                    )
                    relevance = torch.zeros(input_ids.shape[1], device=hidden.device)
                measure_score = not mode
                if measure_score and hidden.is_cuda:
                    torch.cuda.synchronize(hidden.device)
                started = time.perf_counter()
                # FastV protects every text token and prunes only its redundant
                # modality. For LLaDA the corresponding redundant class is the
                # untouched future MASK canvas. Prompt and already revealed
                # language tokens must never compete with those masks.
                future_set = set(future_masks)
                context = [i for i in positions if i not in target_set and i not in future_set]
                merged_states = {}
                dominant_context = len(context)
                contextual_context = 0
                if mode == "zip":
                    prompt_context = [i for i in context if i < protected_prefix_length]
                    generated_context = [i for i in context if i >= protected_prefix_length]
                    kept_generated, merged_states, dominant_context, contextual_context = compress_context(
                        hidden, relevance, generated_context,
                        self.config.context_dominant_ratio,
                        self.config.contextual_ratio,
                        self.config.context_merge_weight,
                    )
                    kept_context = sorted(prompt_context + kept_generated)
                    dominant_context += len(prompt_context)
                else:
                    kept_context = context
                dominant_support = 0
                contextual_support = 0
                if mode == "zip":
                    kept_future, support_states, dominant_support, contextual_support = compress_context(
                        hidden, relevance, future_masks,
                        self.config.support_keep_ratio,
                        self.config.support_contextual_ratio,
                        self.config.context_merge_weight,
                        assignment="spatial",
                    )
                    merged_states.update(support_states)
                    candidate_keep = sorted(set(target_positions + kept_context + kept_future))
                else:
                    candidate_keep = choose_support(
                        relevance, target_positions, self.config.support_keep_ratio,
                        candidates=future_masks, protected=kept_context,
                    )
                keep = candidate_keep if mode else positions
                if measure_score and hidden.is_cuda:
                    torch.cuda.synchronize(hidden.device)
                support = future_masks
                kept_support = [i for i in candidate_keep if i in future_set]
                denominator = float(relevance[support].sum().item()) if support else 0.0
                layer_record = dict(
                    layer=number, original_tokens=len(positions), targets=len(set(target_positions)),
                    support=len(support), protected=len(context), kept_support=len(kept_support),
                    kept_context=len(kept_context), dominant_context=dominant_context,
                    contextual_context=contextual_context,
                    dominant_support=dominant_support,
                    contextual_support=contextual_support,
                    deep_tokens=len(candidate_keep),
                    final_deep_tokens=len(candidate_keep),
                    retained_support_mass=(float(relevance[kept_support].sum().item()) / denominator if denominator else 1.0),
                    score_seconds=(time.perf_counter() - started if measure_score else 0.0),
                )
                if mode:
                    index = torch.tensor(keep, device=hidden.device)
                    hidden = hidden.index_select(1, index)
                    if merged_states:
                        compact = {position: offset for offset, position in enumerate(keep)}
                        for position, state in merged_states.items():
                            hidden[0, compact[position]] = state
                    positions = keep
                    compact_positions = index
            else:
                rotary = block.rotary_emb
                if len(positions) != input_ids.shape[1]:
                    block.rotary_emb = PositionedRotary(
                        rotary, compact_positions, input_ids.shape[1], past_length=past_length
                    )
                try:
                    hidden, _ = block(
                        hidden, attention_bias=None, layer_past=layer_past, use_cache=use_cache
                    )
                finally:
                    block.rotary_emb = rotary
                if (mode == "zip" and self.config.secondary_prune_after_layer
                        and number == self.config.secondary_prune_after_layer):
                    target_compact = [i for i, position in enumerate(positions)
                                      if position in target_set]
                    future_compact = [i for i, position in enumerate(positions)
                                      if position in future_set]
                    context_compact = [i for i, position in enumerate(positions)
                                       if position not in target_set and position not in future_set]
                    kept_future, secondary_states, _, _ = compress_context(
                        hidden, torch.zeros(len(positions), device=hidden.device),
                        future_compact, 0.0, self.config.secondary_support_ratio,
                        self.config.context_merge_weight, assignment="spatial",
                    )
                    keep_compact = sorted(set(target_compact + context_compact + kept_future))
                    compact_index = torch.tensor(keep_compact, device=hidden.device)
                    hidden = hidden.index_select(1, compact_index)
                    if secondary_states:
                        new_compact = {position: offset for offset, position in enumerate(keep_compact)}
                        for position, state in secondary_states.items():
                            hidden[0, new_compact[position]] = state
                    positions = [positions[index] for index in keep_compact]
                    compact_positions = torch.tensor(positions, device=hidden.device)
                    layer_record["secondary_layer"] = number
                    layer_record["secondary_original_tokens"] = len(keep_compact) + (
                        len(future_compact) - len(kept_future)
                    )
                    layer_record["final_deep_tokens"] = len(keep_compact)
        hidden = core.transformer.ln_f(hidden)
        compact = {position: index for index, position in enumerate(positions)}
        gather = torch.tensor([compact[int(position)] for position in target_positions], device=hidden.device)
        head_hidden = hidden.index_select(1, gather) if self.config.target_only_head else hidden
        if core.config.weight_tying:
            logits = torch.nn.functional.linear(head_hidden, core.transformer.wte.weight)
        else:
            logits = core.transformer.ff_out(head_hidden)
        if core.config.scale_logits:
            logits.mul_(1 / math.sqrt(core.config.d_model))
        # Keep the all-retained head as the attribution control.  The optional
        # active-token head projects only positions consumed by the decoder.
        if not self.config.target_only_head:
            logits = logits.index_select(1, gather)
        self.records.append(layer_record)
        return logits
