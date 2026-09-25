"""FastV-style physical support-token pruning inside one v2 diffusion block."""

from dataclasses import dataclass
import sys
import time

import torch


def predictor_positions(start, small_block_size):
    """Raw logit/hidden positions used after v2's one-position logit shift."""
    if start < 0 or small_block_size < 1:
        raise ValueError("Invalid sub-block")
    return [0 if position == 0 else position - 1 for position in range(start, start + small_block_size)]


def choose_keep(relevance, required, support_keep):
    """Return sorted physical positions; required predictors can never be pruned."""
    if relevance.ndim != 1:
        raise ValueError("Expected one relevance score per current-block token")
    length = relevance.numel()
    required = sorted(set(int(i) for i in required))
    if any(i < 0 or i >= length for i in required):
        raise ValueError("Required position outside sequence")
    support = [i for i in range(length) if i not in required]
    count = min(max(0, int(support_keep)), len(support))
    if count:
        candidates = torch.tensor(support, device=relevance.device)
        chosen = candidates[torch.topk(relevance.index_select(0, candidates), count, sorted=False).indices].tolist()
    else:
        chosen = []
    return sorted(required + chosen)


def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left or right else 1.0


@dataclass(frozen=True)
class FastVConfig:
    prune_after_layer: int = 4
    support_keep: int = 8
    block_size: int = 32
    small_block_size: int = 8

    def validate(self, layers):
        if not 1 <= self.prune_after_layer < layers:
            raise ValueError(f"prune_after_layer must be in [1,{layers - 1}]")
        if self.block_size % self.small_block_size:
            raise ValueError("small_block_size must divide block_size")
        max_support = self.block_size - len(set(predictor_positions(0, self.small_block_size)))
        if not 0 <= self.support_keep <= self.block_size:
            raise ValueError(f"support_keep must be nonnegative (typical max {max_support})")


class BlockForward:
    """Run one ordinary denoise forward, optionally pruning after an early layer.

    Prefix keys/values stay exact.  Only current-block hidden states are
    physically shortened.  The returned logits are already aligned to the
    requested target sub-block when pruning is enabled.
    """

    def __init__(self, model, config, observe_layers=(), observe_keeps=(4, 8, 12, 16)):
        self.model = model
        self.config = config
        self.observe_layers = tuple(sorted(set(int(x) for x in observe_layers)))
        self.observe_keeps = tuple(sorted(set(int(x) for x in observe_keeps)))
        config.validate(len(model.model.layers))
        if any(not 1 <= layer <= len(model.model.layers) for layer in self.observe_layers):
            raise ValueError("Observe layer outside model")
        self.records = []

    @staticmethod
    def _capture(layer, hidden, attention_mask, position_ids, cache, cache_position, position_embeddings):
        captured = {}
        hooks = [
            layer.self_attn.q_proj.register_forward_hook(lambda _m, _a, out: captured.__setitem__("q", out)),
            layer.self_attn.k_proj.register_forward_hook(lambda _m, _a, out: captured.__setitem__("k", out)),
        ]
        try:
            output = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                update_past_key_values=False,
                use_block_cache=False,
            )
        finally:
            for hook in hooks:
                hook.remove()
        if set(captured) != {"q", "k"}:
            raise RuntimeError("Failed to capture q/k projections")
        return output, captured

    @staticmethod
    def _relevance(layer, captured, position_embeddings, cache, predictors):
        attention = layer.self_attn
        batch, length, _ = captured["q"].shape
        shape = (batch, length, -1, attention.head_dim)
        query = captured["q"].view(shape).transpose(1, 2)
        key = captured["k"].view(shape).transpose(1, 2)
        module = sys.modules[attention.__class__.__module__]
        query, key = module.apply_rotary_pos_emb(query, key, *position_embeddings)
        key = key.repeat_interleave(attention.num_key_value_groups, dim=1)
        prefix = None
        if cache is not None and len(cache) > attention.layer_idx:
            prefix = cache[attention.layer_idx][0].repeat_interleave(attention.num_key_value_groups, dim=1)
        all_key = torch.cat((prefix, key), dim=-2) if prefix is not None else key
        selected = torch.tensor(sorted(set(predictors)), device=query.device)
        query = query.index_select(-2, selected)
        weights = torch.matmul(query, all_key.transpose(-2, -1)) * attention.scaling
        weights = weights.float().softmax(-1)
        prefix_length = 0 if prefix is None else prefix.shape[-2]
        return weights[..., prefix_length:].mean(dim=(0, 1, 2))

    def __call__(self, input_ids, past_key_values, target_start, prune=True):
        if self.model.training or torch.is_grad_enabled():
            raise RuntimeError("FastV path is inference-only")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] != self.config.block_size:
            raise ValueError("First implementation requires batch=1 and one full current block")
        decoder = self.model.model
        prefix_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if prefix_length % self.config.block_size:
            raise ValueError("Pruned denoise requires a block-aligned exact prefix cache")
        hidden = decoder.embed_tokens(input_ids)
        cache_position = torch.arange(prefix_length, prefix_length + input_ids.shape[1], device=input_ids.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = decoder.rotary_emb(hidden, position_ids)
        predictors = predictor_positions(target_start, self.config.small_block_size)
        score_layers = set(self.observe_layers)
        if prune:
            score_layers.add(self.config.prune_after_layer)
        current_positions = list(range(self.config.block_size))
        forward_records = []

        for index, layer in enumerate(decoder.layers[: decoder.config.num_hidden_layers], start=1):
            if index in score_layers:
                hidden, captured = self._capture(
                    layer, hidden, None, position_ids, past_key_values, cache_position, position_embeddings
                )
                if hidden.is_cuda:
                    torch.cuda.synchronize(hidden.device)
                started = time.perf_counter()
                relevance = self._relevance(layer, captured, position_embeddings, past_key_values, predictors)
                if hidden.is_cuda:
                    torch.cuda.synchronize(hidden.device)
                score_seconds = time.perf_counter() - started
                supports = [i for i in range(relevance.numel()) if i not in set(predictors)]
                support_mass = float(relevance[supports].sum().item()) if supports else 0.0
                tops = {}
                masses = {}
                for keep_count in self.observe_keeps:
                    keep = choose_keep(relevance, predictors, keep_count)
                    selected_support = [i for i in keep if i not in set(predictors)]
                    tops[str(keep_count)] = selected_support
                    masses[str(keep_count)] = (
                        float(relevance[selected_support].sum().item()) / support_mass if support_mass else 1.0
                    )
                forward_records.append(
                    dict(layer=index, support_mass=support_mass, top_support=tops,
                         retained_support_mass=masses, scoring_seconds=score_seconds)
                )
                if prune and index == self.config.prune_after_layer:
                    keep = choose_keep(relevance, predictors, self.config.support_keep)
                    keep_tensor = torch.tensor(keep, device=hidden.device)
                    hidden = hidden.index_select(1, keep_tensor)
                    position_ids = position_ids.index_select(1, keep_tensor)
                    cache_position = cache_position.index_select(0, keep_tensor)
                    position_embeddings = tuple(value.index_select(1, keep_tensor) for value in position_embeddings)
                    current_positions = keep
            else:
                hidden = layer(
                    hidden,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    update_past_key_values=False,
                    use_block_cache=False,
                )

        hidden = decoder.norm(hidden)
        logits = self.model.lm_head(hidden)
        if prune:
            compact = {position: index for index, position in enumerate(current_positions)}
            gather = torch.tensor([compact[position] for position in predictors], device=logits.device)
            logits = logits.index_select(1, gather)
        self.records.append(
            dict(target_start=target_start, predictors=predictors, kept_positions=current_positions,
                 original_tokens=self.config.block_size, deep_tokens=len(current_positions), layers=forward_records)
        )
        return logits


def add_stability(records, keeps):
    """Annotate adjacent observed layers with top-support Jaccard overlap."""
    if isinstance(keeps, int):
        keeps = (keeps,)
    for record in records:
        for keep in keeps:
            key = str(keep)
            previous = None
            for layer in record["layers"]:
                current = layer["top_support"].get(key, [])
                layer[f"jaccard_prev_top{keep}"] = None if previous is None else jaccard(previous, current)
                previous = current
    return records
