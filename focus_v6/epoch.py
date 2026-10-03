"""Atomic provisional epochs and a falsifiable cost model.

The state machine guarantees rollback semantics, not neural-network accuracy.
An audit decides whether an epoch is allowed to commit; it is not an exact
certificate for the original bidirectional LLaDA distribution.
"""
from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class Epoch:
    base_version: int
    positions: tuple[int, ...]
    tokens: tuple[int, ...]
    provisional_canvas: tuple[int, ...]


class EpochLedger:
    """CPU reference ledger over immutable cache-version handles.

    Actual K/V tensors must live in owned shadow storage in a future runtime.
    This reference cannot protect mutable tensors supplied by a caller.
    """
    @staticmethod
    def _handles(cache):
        def immutable(value):
            return isinstance(value, (str, bytes, int, float, type(None))) or (
                isinstance(value, tuple) and all(immutable(v) for v in value))
        result = dict(cache)
        if any(not immutable(v) for v in result.values()):
            raise ValueError("reference ledger requires immutable cache-version handles, not tensor storage")
        return result

    def __init__(self, canvas, cache: Mapping[Any, Any], *, mask_id=126336, special_ids=()):
        self.canvas = tuple(canvas)
        self.cache = self._handles(cache)
        self.mask_id = int(mask_id)
        self.special_ids = set(special_ids) | {self.mask_id}
        self.version = 0
        self.pending = None
        self.staged_cache = None

    def begin(self, positions, tokens):
        if self.pending is not None:
            raise RuntimeError("one pending epoch at a time")
        positions, tokens = tuple(positions), tuple(tokens)
        if not positions or len(positions) != len(tokens) or len(set(positions)) != len(positions):
            raise ValueError("nonempty unique positions and aligned tokens required")
        if any(p < 0 or p >= len(self.canvas) or self.canvas[p] != self.mask_id for p in positions):
            raise ValueError("provisional writes must target current MASK slots")
        if any(t in self.special_ids for t in tokens):
            raise ValueError("EOS and other special tokens use the ordinary drain path")
        tentative = list(self.canvas)
        for p, t in zip(positions, tokens):
            tentative[p] = t
        self.pending = Epoch(self.version, positions, tokens, tuple(tentative))
        return self.pending

    def stage(self, epoch, cache):
        if epoch is not self.pending or epoch.base_version != self.version:
            raise ValueError("stale or unrelated epoch")
        self.staged_cache = self._handles(cache)

    def finish(self, epoch, passed):
        if epoch is not self.pending or epoch.base_version != self.version:
            raise ValueError("stale or unrelated epoch")
        if passed:
            if self.staged_cache is None:
                raise ValueError("accepted canvas requires its matching staged cache")
            self.canvas = epoch.provisional_canvas
            self.cache = self.staged_cache
            self.version += 1
        self.pending = None
        self.staged_cache = None
        return bool(passed)


def atomic_pass(probabilities, top1, drafts, *, gamma=.8):
    if not 0 < gamma <= 1 or not len(probabilities) == len(top1) == len(drafts) or not drafts:
        raise ValueError("valid gamma and nonempty aligned candidate arrays required")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("invalid probability")
    # Log-space prevents accidental underflow. This is a policy, not a claim
    # that these conditional marginals equal the true joint probability.
    budget = sum(math.log(p) if p else -math.inf for p in probabilities)
    return all(a == b for a, b in zip(top1, drafts)) and budget >= math.log(gamma)


def expected_rate_ratio(*, baseline_ms, joint_ms, recovery_ms,
                        pass_fraction, safe_progress, epoch_size, baseline_progress):
    values = (baseline_ms, joint_ms, recovery_ms, safe_progress, epoch_size, baseline_progress)
    if any(not math.isfinite(x) or x < 0 for x in values) or not 0 <= pass_fraction <= 1:
        raise ValueError("invalid cost/progress parameters")
    if baseline_ms <= 0 or joint_ms <= 0 or baseline_progress <= 0:
        raise ValueError("positive execution times and baseline progress required")
    cost = joint_ms + (1 - pass_fraction) * recovery_ms
    progress = safe_progress + pass_fraction * epoch_size
    return (progress / cost) / (baseline_progress / baseline_ms)
