"""Query-row accounting for Relay diagnostics.

This is a screening model, not a latency claim.  It reports the amount of
regular-pass row work that can possibly be removed if accepted draft cache
images are promoted successfully.  Real end-to-end latency must be measured.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cycle:
    regular_rows: int
    verify_rows: int
    accepted: int
    next_regular_contains_accepted: int

    def __post_init__(self) -> None:
        values = (self.regular_rows, self.verify_rows, self.accepted, self.next_regular_contains_accepted)
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("cycle counts must be non-negative integers")
        if self.next_regular_contains_accepted > self.accepted:
            raise ValueError("cannot skip more accepted rows than exist")


def row_ceiling(cycles: list[Cycle]) -> dict:
    if not cycles:
        raise ValueError("at least one cycle is required")
    baseline = sum(c.regular_rows + c.verify_rows for c in cycles)
    removable = sum(c.next_regular_contains_accepted for c in cycles)
    relay = baseline - removable
    return {
        "cycles": len(cycles),
        "baseline_query_rows": baseline,
        "removable_identity_refresh_rows": removable,
        "relay_query_rows": relay,
        "optimistic_row_speedup": baseline / relay if relay else float("inf"),
        "scope": "Screening ceiling only; QKV, attention, MLP, launch and memory costs are not linear in rows.",
    }


def joint_row_ceiling(cycles: list[Cycle], block: int = 32) -> dict:
    """Compare two-call Flash rows with one constant-width Relay transition."""
    if not cycles or block < 1:
        raise ValueError("cycles and a positive block are required")
    baseline = sum(c.regular_rows + c.verify_rows for c in cycles)
    relay = len(cycles) * 3 * block
    return {
        "cycles": len(cycles),
        "baseline_query_rows": baseline,
        "relay_query_rows": relay,
        "optimistic_row_speedup": baseline / relay,
        "relay_rows_per_cycle": 3 * block,
        "scope": "Screening ceiling only. It assumes the clean branch can supply the next proposal; real acceptance, kernels and latency must be measured.",
    }
