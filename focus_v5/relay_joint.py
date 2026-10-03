"""Read-only construction helpers for a real FOCUS-v5 Relay joint pass.

The pinned Flash kernel requires a power-of-two verification tile.  Relay has
96 live rows for a 32-token block, so the implementation uses a 128-row kernel
tile and masks the final 32 rows.  This module builds the live query and the
external-cache key list from the exact Python state immediately before the
official Flash verification call.

No cache is committed here.  The first GPU experiment is deliberately a
shadow pass: it measures the hardware cost and the verification decisions
before the transactional state machine is allowed to mutate generation.
"""

from __future__ import annotations

import ast
import functools
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .relay_cache import JointRelayLayout, joint_relay_mask


@dataclass(frozen=True)
class JointCall:
    input_ids: torch.Tensor
    positions: list[Any]
    lengths: list[Any]
    layout: JointRelayLayout
    clean_positions: torch.Tensor
    draft_positions: torch.Tensor
    tracked_positions: torch.Tensor


def padded_joint_mask(block: int, search: int, kernel_width: int = 128,
                      *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return Relay's exact live mask padded for the Triton verify tile."""
    live = joint_relay_mask(block, search)
    if kernel_width < live.shape[0] or kernel_width & (kernel_width - 1):
        raise ValueError("kernel_width must be a power of two covering all live rows")
    result = torch.zeros((kernel_width, kernel_width), dtype=torch.bool, device=device)
    result[: live.shape[0], : live.shape[1]] = live.to(device=device)
    return result


def build_joint_call(state: dict[str, Any], *, kernel_width: int = 128) -> JointCall:
    """Build a batch-one 96-live-row joint call from pinned-generator locals.

    The current regular pass has already generated the speculative labels, but
    the official verifier has not run.  The joint call contains:

    ``[clean tracked | clean complete block | draft prefix | verifier prefix]``.

    Cache keys represented by a live query row are removed from the external
    key list, exactly as the official verify call removes its private rows.
    """
    required = {
        "model", "x", "x_draft", "full_pos", "num_decoded", "num_verify",
        "seqlen_k", "start_layer", "query_tracked_blocks", "active_batch",
        "num_active", "max_length", "block_m", "block_n", "elastic_cache",
        "rotary_emb_pos", "info", "attn_scores", "gamma",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise KeyError(f"missing pinned generator state: {missing}")
    block = int(state["block_m"])
    if block != 32:
        raise ValueError("the first Relay kernel experiment is fixed to block 32")
    active = list(state["active_batch"])
    if len(active) != 1 or int(active[0]) < 0:
        raise ValueError("the first Relay kernel experiment requires one active request")
    search = int(state["num_verify"])
    layout = JointRelayLayout(block, search)
    decoded = int(state["num_decoded"][0])
    if decoded < layout.tracked:
        raise ValueError("insufficient decoded prefix for the Relay tracked view")

    full_pos = state["full_pos"][0]
    tracked = full_pos[decoded - layout.tracked : decoded]
    clean = full_pos[decoded : decoded + block]
    if clean.numel() != block:
        raise ValueError("incomplete clean block")
    draft = clean[:search]
    if torch.unique(torch.cat((tracked, clean))).numel() != layout.tracked + block:
        raise AssertionError("joint query positions overlap")

    x = state["x"]
    x_draft = state["x_draft"]
    query_ids = torch.cat((x[tracked], x[clean], x_draft[draft], x[draft])).unsqueeze(0)
    query_pos = torch.cat((tracked, clean, draft, draft))
    if query_ids.shape[1] != layout.total or query_pos.numel() != layout.total:
        raise AssertionError("joint live-row geometry changed")

    key_end = int(state["seqlen_k"][0])
    key_pos = torch.cat((
        full_pos[: decoded - layout.tracked],
        full_pos[decoded + block : key_end],
    ))
    if torch.isin(key_pos, torch.cat((tracked, clean))).any():
        raise AssertionError("a joint private position leaked into the public key list")

    blocks = torch.tensor(
        [[0, key_pos.numel(), 0, layout.total]],
        device=query_ids.device,
        dtype=torch.int32,
    )
    mask = padded_joint_mask(block, search, kernel_width, device=query_ids.device)
    positions = [
        query_pos,
        key_pos,
        state["rotary_emb_pos"],
        state["info"],
        state["attn_scores"],
        mask,
    ]
    # The pinned kernel uses 2 * block_m as its verification tile.  A logical
    # block size of 64 therefore selects BLOCK_M=128 while q_end remains 96.
    lengths = [
        list(state["start_layer"]),
        blocks,
        None,
        state["query_tracked_blocks"],
        None,
        active,
        int(state["num_active"]),
        int(state["max_length"]),
        kernel_width // 2,
        int(state["block_n"]),
        state["elastic_cache"],
        True,
    ]
    return JointCall(query_ids, positions, lengths, layout, clean, draft, tracked)


def accepted_prefix(logits: torch.Tensor, call: JointCall, gamma: float) -> dict[str, Any]:
    """Apply the pinned cumulative-probability rule to Relay verifier rows."""
    search = call.layout.search
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != call.layout.total:
        raise ValueError("unexpected joint logits shape")
    if search == 0:
        return {"probabilities": [], "top1": [], "drafts": [], "accepted": 0}
    draft_rows = logits[:, call.layout.draft, :]
    verify_rows = logits[:, call.layout.verify, :]
    drafts = call.input_ids[0, call.layout.draft]
    probabilities = verify_rows[0].double().softmax(-1).gather(1, drafts[:, None]).squeeze(1)
    accepted = int((probabilities.cumprod(0) >= float(gamma)).sum().item())
    return {
        "probabilities": probabilities.detach().cpu().tolist(),
        "top1": verify_rows[0].argmax(-1).detach().cpu().tolist(),
        "drafts": drafts.detach().cpu().tolist(),
        "accepted": accepted,
        "draft_top1": draft_rows[0].argmax(-1).detach().cpu().tolist(),
    }


def instrument_before_verify(function: Callable, observer: Callable,
                             acceptance_observer: Callable | None = None) -> Callable:
    """Inject one callback immediately before the pinned verify model call.

    The original function and module globals remain unchanged.  The source
    shape is checked exactly and unfamiliar versions fail closed.
    """
    original = inspect.unwrap(function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    definition = tree.body[0]
    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("a standalone function is required")
    definition.decorator_list = []
    calls: list[ast.Assign] = []
    for node in ast.walk(definition):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "model":
            calls.append(node)
    calls.sort(key=lambda node: node.lineno)
    if len(calls) != 2:
        raise ValueError("expected exactly the pinned regular and verify model calls")
    target = calls[1]
    additions = [0]
    acceptance_additions = [0]

    class Inject(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign):
            if acceptance_observer is not None and ast.unparse(node) == "x0_p_verify = x0_p_verify.cumprod(dim=0)":
                acceptance_additions[0] += 1
                callback = ast.parse(
                    "_focus_official_accept(x0_p_verify, x_verify_j, "
                    "query_pos_flat[acc_seqlen_verify + T:acc_seqlen_verify + T + S], "
                    "p_verify.argmax(dim=-1), gamma)"
                ).body[0]
                return [ast.copy_location(callback, node), node]
            if node is not target:
                return self.generic_visit(node)
            additions[0] += 1
            callback = ast.parse("_focus_joint_shadow(model, locals())").body[0]
            return [ast.copy_location(callback, node), node]

    tree = Inject().visit(tree)
    if additions[0] != 1:
        raise ValueError("failed to locate the pinned verify call")
    if acceptance_observer is not None and acceptance_additions[0] != 1:
        raise ValueError("failed to locate the pinned cumulative acceptance rule")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    if "_focus_joint_shadow" in namespace:
        raise ValueError("observer namespace collision")
    namespace["_focus_joint_shadow"] = observer
    if acceptance_observer is not None:
        if "_focus_official_accept" in namespace:
            raise ValueError("acceptance observer namespace collision")
        namespace["_focus_official_accept"] = acceptance_observer
    exec(compile(tree, original.__code__.co_filename + ":relay_joint_shadow", "exec"), namespace)
    copied = torch.no_grad()(namespace[original.__name__])
    return functools.update_wrapper(copied, original)
