"""Reference semantics for cache-carrying verification.

This module deliberately contains no model-specific monkey patches.  It defines
the information-flow contract that a Triton implementation must satisfy and a
small transactional cache primitive used by tests and probes.

Rows are laid out as ``[tracked | draft | verify]``.  A draft row contains the
proposed token identity.  A verify row contains MASK and predicts the draft at
the same rank.  ``True`` in a mask means that the query row may attend to the
key row, matching the pinned Flash-dLLM kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def _geometry(block: int, search: int) -> tuple[int, int]:
    if not isinstance(block, int) or not isinstance(search, int):
        raise TypeError("block and search must be integers")
    if block < 1 or search < 0 or search > block // 2:
        raise ValueError("invalid Flash verification geometry")
    return 2 * block - 2 * search, search


def flash_verify_mask(block: int, search: int) -> torch.Tensor:
    """Reproduce the pinned Flash-dLLM two-view mask exactly."""
    tracked, search = _geometry(block, search)
    mask = torch.ones((2 * block, 2 * block), dtype=torch.bool)
    ranks = torch.arange(search)
    earlier_equal = ranks[:, None] >= ranks[None, :]
    earlier = ranks[:, None] > ranks[None, :]
    mask[:tracked, tracked : tracked + search] = False
    mask[tracked : tracked + search, tracked : tracked + search] = earlier_equal
    mask[tracked : tracked + search, tracked + search :] = ~earlier_equal
    mask[tracked + search :, tracked : tracked + search] = earlier
    mask[tracked + search :, tracked + search :] = ~earlier
    return mask


def relay_verify_mask(block: int, search: int) -> torch.Tensor:
    """Build the rollback-safe FOCUS-v5 Relay mask.

    The public/tracked rows never ingest speculative identities.  Draft row i
    can depend on proposals 0..i and becomes the cache image for proposal i.
    Verify row i can depend only on proposals 0..i-1, so it cannot observe the
    token it is asked to verify.  Prefix MASK rows may interact in topological
    order; this preserves more context without opening a path from a future or
    rejected proposal.

    The mask changes Flash verification semantics and therefore is not claimed
    to be accuracy-equivalent.  Its value is the explicit all-layer rollback
    invariant, which is tested below and later measured on the real model.
    """
    tracked, search = _geometry(block, search)
    total = 2 * block
    mask = torch.zeros((total, total), dtype=torch.bool)
    if tracked:
        mask[:tracked, :tracked] = True

    for rank in range(search):
        draft = tracked + rank
        verify = tracked + search + rank
        # Stable public background.
        mask[draft, :tracked] = True
        mask[verify, :tracked] = True
        # A staged cache image may include itself and earlier versions.
        mask[draft, tracked : draft + 1] = True
        # It may also use only clean verification rows no later than itself.
        mask[draft, tracked + search : verify + 1] = True
        # A verifier never sees its own proposal or a later proposal.
        mask[verify, tracked:draft] = True
        mask[verify, tracked + search : verify + 1] = True
    return mask


@dataclass(frozen=True)
class JointRelayLayout:
    """Constant-width three-view layout used by the proposed runtime.

    ``tracked + clean + draft + verify`` is always ``3 * block`` rows because
    tracked follows Flash's ``2 * block - 2 * search`` budget.
    """

    block: int
    search: int

    def __post_init__(self) -> None:
        _geometry(self.block, self.search)

    @property
    def tracked(self) -> int:
        return 2 * self.block - 2 * self.search

    @property
    def clean(self) -> slice:
        return slice(self.tracked, self.tracked + self.block)

    @property
    def draft(self) -> slice:
        start = self.clean.stop
        return slice(start, start + self.search)

    @property
    def verify(self) -> slice:
        start = self.draft.stop
        return slice(start, start + self.search)

    @property
    def total(self) -> int:
        return self.verify.stop


def joint_relay_mask(block: int, search: int) -> torch.Tensor:
    """Build the three-view cache+verify mask.

    Stable rows comprise tracked cache refreshes and one clean MASK view of the
    complete active block.  They may mix bidirectionally but cannot read draft
    or verify rows.  Consequently their K/V and logits are valid regardless of
    how many current proposals pass.  Draft/verify rows may read the stable
    background and only a topological proposal prefix.

    The total query width is exactly ``3 * block`` for every search count.  The
    clean branch supplies next-cycle proposals while the speculative branch
    verifies current proposals, turning two serial Flash calls into one joint
    state transition after bootstrap.
    """
    layout = JointRelayLayout(block, search)
    mask = torch.zeros((layout.total, layout.total), dtype=torch.bool)
    stable_stop = layout.clean.stop
    mask[:stable_stop, :stable_stop] = True
    for rank in range(search):
        draft = layout.draft.start + rank
        verify = layout.verify.start + rank
        mask[draft, :stable_stop] = True
        mask[verify, :stable_stop] = True
        mask[draft, layout.draft.start : draft + 1] = True
        mask[draft, layout.verify.start : verify + 1] = True
        mask[verify, layout.draft.start:draft] = True
        mask[verify, layout.verify.start : verify + 1] = True
    return mask


def proposal_paths(mask: torch.Tensor, tracked: int, search: int, layers: int) -> torch.Tensor:
    """Return possible row-to-proposal information paths after ``layers``.

    Position-wise normalization/MLP/RoPE do not mix rows.  Residual connections
    are included.  Public cache is assumed to contain no current speculative
    token identities.
    """
    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("a square boolean mask is required")
    if tracked < 0 or search < 0 or tracked + 2 * search != mask.shape[0] or layers < 0:
        raise ValueError("invalid layout")
    paths = torch.zeros((mask.shape[0], search), dtype=torch.bool)
    paths[tracked : tracked + search] = torch.eye(search, dtype=torch.bool)
    adjacency = mask.to(torch.int64)
    for _ in range(layers):
        paths |= (adjacency @ paths.to(torch.int64)) > 0
    return paths


def validate_relay_noninterference(block: int, search: int, layers: int = 32) -> dict:
    """Fail if any row violates Relay's speculative dependency contract."""
    tracked, _ = _geometry(block, search)
    paths = proposal_paths(relay_verify_mask(block, search), tracked, search, layers)
    if bool(paths[:tracked].any()):
        raise AssertionError("tracked cache acquired a speculative dependency")
    draft_paths = paths[tracked : tracked + search]
    verify_paths = paths[tracked + search :]
    for rank in range(search):
        if bool(draft_paths[rank, rank + 1 :].any()):
            raise AssertionError(f"draft cache {rank} depends on a later proposal")
        if bool(verify_paths[rank, rank:].any()):
            raise AssertionError(f"verifier {rank} observes its own/later proposal")
    return {
        "tracked": tracked,
        "search": search,
        "layers": layers,
        "draft_paths": draft_paths,
        "verify_paths": verify_paths,
    }


def validate_joint_relay_noninterference(block: int, search: int, layers: int = 32) -> dict:
    """Validate clean-cache and prefix-verification isolation at all layers."""
    layout = JointRelayLayout(block, search)
    mask = joint_relay_mask(block, search)
    paths = torch.zeros((layout.total, search), dtype=torch.bool)
    paths[layout.draft] = torch.eye(search, dtype=torch.bool)
    adjacency = mask.to(torch.int64)
    for _ in range(layers):
        paths |= (adjacency @ paths.to(torch.int64)) > 0
    if bool(paths[: layout.clean.stop].any()):
        raise AssertionError("clean cache/proposal branch acquired speculative information")
    drafts = paths[layout.draft]
    verifies = paths[layout.verify]
    for rank in range(search):
        if bool(drafts[rank, rank + 1 :].any()):
            raise AssertionError(f"draft cache {rank} depends on a future proposal")
        if bool(verifies[rank, rank:].any()):
            raise AssertionError(f"verifier {rank} sees its own/future proposal")
    return {"layout": layout, "layers": layers, "paths": paths,
            "draft_paths": drafts, "verify_paths": verifies}


@dataclass
class RelayCacheTransaction:
    """Shadow K/V updates with atomic accepted-prefix commit.

    The clean branch supplies a valid non-speculative version for every clean
    position.  Draft versions cover a subset of those positions.  Commit first
    installs all clean versions, then atomically overlays exactly the accepted
    draft prefix.  Rejected positions therefore receive clean MASK K/V rather
    than stale or rejected-draft state.
    """

    public: Sequence[tuple[torch.Tensor, torch.Tensor]]
    clean: Sequence[tuple[torch.Tensor, torch.Tensor]]
    clean_positions: torch.Tensor
    draft: Sequence[tuple[torch.Tensor, torch.Tensor]]
    draft_positions: torch.Tensor
    _closed: bool = False

    def __post_init__(self) -> None:
        for name, positions in (("clean", self.clean_positions), ("draft", self.draft_positions)):
            if positions.dtype not in (torch.int32, torch.int64) or positions.ndim != 1:
                raise ValueError(f"{name} positions must be a one-dimensional integer tensor")
            if positions.numel() and int(positions.min()) < 0:
                raise ValueError("negative cache position")
        if not self.public or len(self.public) != len(self.clean) or len(self.public) != len(self.draft):
            raise ValueError("aligned non-empty layer caches are required")
        clean_count = self.clean_positions.numel()
        draft_count = self.draft_positions.numel()
        clean_set = set(map(int, self.clean_positions.tolist()))
        if any(int(position) not in clean_set for position in self.draft_positions.tolist()):
            raise ValueError("draft positions must have a clean fallback")
        for layer, (public, clean, draft) in enumerate(zip(self.public, self.clean, self.draft)):
            if len(public) != 2 or len(clean) != 2 or len(draft) != 2:
                raise ValueError("each layer must contain key and value")
            for old, clean_value, draft_value in zip(public, clean, draft):
                prefix = old.shape[:-2] + old.shape[-1:]
                if (old.ndim < 2 or clean_value.shape[-2] != clean_count or draft_value.shape[-2] != draft_count
                        or clean_value.shape[:-2] + clean_value.shape[-1:] != prefix
                        or draft_value.shape[:-2] + draft_value.shape[-1:] != prefix):
                    raise ValueError(f"incompatible cache shape at layer {layer}")
                for positions in (self.clean_positions, self.draft_positions):
                    if positions.numel() and int(positions.max()) >= old.shape[-2]:
                        raise ValueError("cache position outside public tensor")

    def commit(self, accepted: int) -> None:
        if self._closed:
            raise RuntimeError("transaction already closed")
        if not 0 <= accepted <= self.draft_positions.numel():
            raise ValueError("accepted count outside staged prefix")
        clean_pos = self.clean_positions.to(torch.long)
        draft_pos = self.draft_positions[:accepted].to(torch.long)
        for public, clean, draft in zip(self.public, self.clean, self.draft):
            for old, clean_value, draft_value in zip(public, clean, draft):
                if clean_pos.numel():
                    old.index_copy_(-2, clean_pos.to(old.device), clean_value.to(old.device))
                if draft_pos.numel():
                    old.index_copy_(-2, draft_pos.to(old.device), draft_value[..., :accepted, :].to(old.device))
        self._closed = True

    def rollback(self) -> None:
        if self._closed:
            raise RuntimeError("transaction already closed")
        self._closed = True
