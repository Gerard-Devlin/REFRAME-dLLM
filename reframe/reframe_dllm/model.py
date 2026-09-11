"""Request-local adapter for LLaDA's unmodified Transformer blocks.

Only selected token rows traverse the model. Their true absolute positions
are used for RoPE. Reference keys are post-RoPE; native Fast-dLLM caches are
pre-RoPE and MUST NOT be passed to this adapter.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
from types import MethodType

import torch

from .transport import fit_transport, grouped_attention, relative_error


@dataclass
class ReframeConfig:
    kind: str = "pair"
    pilots: int = 16                 # per side, includes validation pilots
    refresh_blocks: int = 2          # hard maximum reference age
    ridge: float = 1e-3
    max_pilot_error: float = 0.25    # heuristic, NOT an error certificate
    min_scale: float = 0.25
    max_scale: float = 4.0
    backend: str = "torch"
    materialize: bool = False

    def __post_init__(self):
        if self.kind not in {"stale", "shift", "scale", "pair"}:
            raise ValueError("kind must be stale, shift, scale, or pair")
        if self.pilots < 0 or (self.kind != "stale" and self.pilots < 4):
            raise ValueError("Transport needs at least 4 pilots per side")
        if self.refresh_blocks < 1 or self.ridge < 0 or self.max_pilot_error <= 0:
            raise ValueError("Invalid refresh, ridge or pilot error setting")
        if not 0 < self.min_scale < 1 < self.max_scale:
            raise ValueError("Scale bounds must contain identity and exclude zero")
        if self.backend not in {"torch", "flash"}:
            raise ValueError("backend must be torch or flash")


class RefreshRequired(RuntimeError):
    pass


def spaced_indices(indices, count):
    if count == 0 or indices.numel() == 0:
        return indices[:0]
    at = torch.linspace(0, indices.numel() - 1, min(count, indices.numel()),
                        device=indices.device).round().long()
    return indices[at]


def split_pilots(indices):
    """Every fourth pilot is held out, including for tiny groups."""
    at = torch.arange(indices.numel(), device=indices.device)
    return indices[at % 4 != 3], indices[at % 4 == 3]


def prepare_qkv(block, q, k, v, positions, sequence_length):
    batch, tokens, channels = q.shape
    dtype = k.dtype
    if block.q_norm is not None and block.k_norm is not None:
        q, k = block.q_norm(q).to(dtype), block.k_norm(k).to(dtype)
    dim = channels // block.config.n_heads
    q = q.view(batch, tokens, block.config.n_heads, dim).transpose(1, 2)
    k = k.view(batch, tokens, block.config.effective_n_kv_heads, dim).transpose(1, 2)
    v = v.view(batch, tokens, block.config.effective_n_kv_heads, dim).transpose(1, 2)
    # LLaDA rotates split halves, and pilots can be non-contiguous.
    sin, cos = block.rotary_emb.get_rotary_embedding(sequence_length, q.device)
    qwork, kwork = (q.float(), k.float()) if block.config.rope_full_precision else (q, k)
    sin, cos = sin.index_select(2, positions).to(qwork.dtype), cos.index_select(2, positions).to(qwork.dtype)
    q = block.rotary_emb.apply_rotary_pos_emb(sin, cos, qwork).to(dtype)
    k = block.rotary_emb.apply_rotary_pos_emb(sin, cos, kwork).to(dtype)
    return q, k, v


class ReframeSession:
    """Single request, batch=1, no padding, no model/attention monkeypatch leaks.

    A short-lived override reuses the existing layer norms, QKV projections,
    FFNs, output head and residuals. It never modifies their parameters.
    Calls are not thread-safe on a shared model; run requests sequentially.
    """
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or ReframeConfig()
        mc = model.model.config
        if model.training or not mc.rope or mc.alibi or mc.block_group_size != 1:
            raise ValueError("Requires eval-mode LLaDA with RoPE and ungrouped blocks")
        if str(mc.block_type) != "llama":
            raise ValueError("First prototype supports LLaDA's llama block only")
        self.blocks = list(model.model.transformer.blocks)
        self.reference = {}
        self.stats = dict(full_forwards=0, partial_forwards=0, partial_layer_calls=0,
                          fallback_refreshes=0, pilot_token_rows=0, commit_token_rows=0,
                          processed_token_rows=0, fit_count=0,
                          max_observed_pilot_error=0.0, fallback_reasons=[])

    @contextmanager
    def _override(self, fn):
        originals = []
        try:
            for block in self.blocks:
                if "attention" in block.__dict__:
                    raise RuntimeError("Model already has an attention override")
                block.attention = MethodType(fn, block)
                originals.append(block)
            yield
        finally:
            for block in originals:
                del block.attention

    def _check_input(self, x):
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError("REFRAME prototype requires unpadded batch size 1")

    @torch.no_grad()
    def full(self, x, refresh=True, capture_layers=()):
        """Full forward; optional read-only oracle capture never changes cache."""
        self._check_input(x)
        pos = torch.arange(x.shape[1], device=x.device)
        cache, captured = {}, {}
        session = self

        def attention(block, q, k, v, *args, **kwargs):
            q, k, v = prepare_qkv(block, q, k, v, pos, x.shape[1])
            if refresh:
                cache[block.layer_id] = (k.detach(), v.detach())
            if block.layer_id in capture_layers:
                captured[block.layer_id] = (q.detach(), k.detach(), v.detach())
            # Match the native attention backend on full forwards.
            out = block._scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
            out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
            return block.attn_out(out), None

        with self._override(attention):
            logits = self.model(x, use_cache=False).logits
        if refresh:
            session.reference = cache
            session.stats["full_forwards"] += 1
            session.stats["processed_token_rows"] += x.shape[1]
        return logits, captured

    @torch.no_grad()
    def partial(self, x, start, end, pending=None):
        """Run active block, pilots and (once) the last completed block.

        pending must contain committed token IDs in x. Those rows are freshly
        evaluated and inverse-written into the PREFIX reference frame only
        after ALL layers succeed. Partial failure cannot corrupt old caches.
        """
        self._check_input(x)
        if not self.reference:
            raise ValueError("Initialize reference with full() first")
        cfg = self.config
        total = x.shape[1]
        if not 0 <= start < end <= total:
            raise ValueError("Invalid block bounds")
        active = torch.arange(start, end, device=x.device)
        pending = active[:0] if pending is None else pending
        if pending.numel() and not bool(((pending >= 0) & (pending < start)).all().item()):
            raise ValueError("Pending completed block must precede active block")
        excluded = torch.zeros(total, device=x.device, dtype=torch.bool)
        excluded[active] = True
        excluded[pending] = True
        prefix = torch.arange(0, start, device=x.device)
        suffix = torch.arange(end, total, device=x.device)
        regions = [prefix[~excluded[prefix]], suffix]
        count = 0 if cfg.kind == "stale" else cfg.pilots
        pilots = [spaced_indices(ids, count) for ids in regions]
        positions = torch.cat([active, pending, *pilots]).unique(sorted=True)
        fresh_mask = torch.zeros(total, dtype=torch.bool, device=x.device)
        fresh_mask[positions] = True
        cached = [ids[~fresh_mask[ids]] for ids in regions]
        pilot_local = [torch.searchsorted(positions, ids) for ids in pilots]
        pending_local = torch.searchsorted(positions, pending)
        staged = {}
        errors = []
        self.stats["partial_forwards"] += 1
        self.stats["pilot_token_rows"] += sum(p.numel() for p in pilots)
        self.stats["commit_token_rows"] += pending.numel()
        self.stats["processed_token_rows"] += positions.numel()
        session = self

        def attention(block, q, k, v, *args, **kwargs):
            session.stats["partial_layer_calls"] += 1
            q, k, v = prepare_qkv(block, q, k, v, positions, total)
            rk, rv = session.reference[block.layer_id]
            groups = []
            maps = []
            for side in range(2):
                ids, loc = pilots[side], pilot_local[side]
                train, valid = split_pilots(torch.arange(ids.numel(), device=x.device))
                t = fit_transport(rk[:, :, ids[train]], rv[:, :, ids[train]],
                                  k[:, :, loc[train]], v[:, :, loc[train]], cfg.kind, cfg.ridge)
                session.stats["fit_count"] += int(train.numel() > 0 and cfg.kind != "stale")
                if not t.is_safe(cfg.min_scale, cfg.max_scale):
                    raise RefreshRequired(f"unsafe_transform_layer_{block.layer_id}_side_{side}")
                if valid.numel():
                    # Held-out pilots protect against simple overfitting; no
                    # guarantee is made about the rest of the cache.
                    pilot_errors = (relative_error(t.key(rk[:, :, ids[valid]]), k[:, :, loc[valid]]),
                                    relative_error(t.value(rv[:, :, ids[valid]]), v[:, :, loc[valid]]))
                    if not all(math.isfinite(e) for e in pilot_errors):
                        raise RefreshRequired(f"nonfinite_pilot_layer_{block.layer_id}_side_{side}")
                    err = max(pilot_errors)
                    errors.append(err)
                    session.stats["max_observed_pilot_error"] = max(session.stats["max_observed_pilot_error"], err)
                    if err > cfg.max_pilot_error:
                        raise RefreshRequired(f"pilot_residual_layer_{block.layer_id}_side_{side}")
                maps.append(t)
                ids_cached = cached[side]
                groups.append((rk.index_select(2, ids_cached), rv.index_select(2, ids_cached), t))
            # All current rows occur exactly once; never include pilot copies
            # in both reference attention and this fresh group.
            groups.append((k, v, None))
            out = grouped_attention(q, groups, cfg.backend, cfg.materialize)
            if pending.numel():
                kw, vw = maps[0].inverse(k[:, :, pending_local], v[:, :, pending_local])
                kw, vw = kw.to(rk.dtype), vw.to(rv.dtype)
                if not bool((torch.isfinite(kw).all() & torch.isfinite(vw).all()).item()):
                    raise RefreshRequired("nonfinite_inverse_write")
                staged[block.layer_id] = (kw, vw)
            out = out.transpose(1, 2).contiguous().view(1, positions.numel(), -1)
            return block.attn_out(out), None

        with self._override(attention):
            logits = self.model(x.index_select(1, positions), use_cache=False).logits
        for layer, (kw, vw) in staged.items():
            rk, rv = self.reference[layer]
            rk.index_copy_(2, pending, kw)
            rv.index_copy_(2, pending, vw)
        self.last_positions = positions
        self.last_pilot_error = max(errors, default=0.0)
        return logits.index_select(1, torch.searchsorted(positions, active))

    @torch.no_grad()
    def step(self, x, start, end, pending=None, force_refresh=False):
        if force_refresh or not self.reference:
            logits, _ = self.full(x)
            return logits[:, start:end], True
        try:
            return self.partial(x, start, end, pending), False
        except RefreshRequired as exc:
            self.stats["fallback_refreshes"] += 1
            self.stats["fallback_reasons"].append(str(exc))
            logits, _ = self.full(x)
            return logits[:, start:end], True
