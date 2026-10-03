"""Capture native private projection buffers without editing third-party code."""
from contextlib import contextmanager
import importlib

import torch


@contextmanager
def capture_private_kv():
    module = importlib.import_module("flash_cache_triton")
    original = module._flash_verify_qkv_proj_fwd
    captures = []

    class Capture:
        def __getitem__(self, grid):
            launch = original[grid]
            def run(*args, **kwargs):
                result = launch(*args, **kwargs)
                # Pinned kernel arguments: Xn,Q,K,V. Retain actual native
                # buffers, not recomputed/teacher-oracle KV. No public write.
                captures.append((args[2], args[3]))
                return result
            return run

    module._flash_verify_qkv_proj_fwd = Capture()
    try:
        yield captures
    finally:
        module._flash_verify_qkv_proj_fwd = original


@torch.no_grad()
def promote(private_bank, captured, rows, destinations):
    """Real tensor transaction into a PRIVATE shadow bank, including copy cost."""
    if len(private_bank) != len(captured) or not private_bank:
        raise ValueError("complete layer bank required")
    if rows.ndim != 1 or destinations.ndim != 1 or rows.dtype != torch.long or destinations.dtype != torch.long:
        raise ValueError("integer row vectors required")
    if len(rows) != len(destinations) or torch.unique(destinations).numel() != len(destinations):
        raise ValueError("one destination per physical position required")
    for (target_k, target_v), (source_k, source_v) in zip(private_bank, captured):
        if any(target.data_ptr() == source.data_ptr() for target in (target_k, target_v)
               for source in (source_k, source_v)):
            raise ValueError("private projection cannot alias committed bank")
        target_k.index_copy_(0, destinations, source_k.index_select(0, rows))
        target_v.index_copy_(0, destinations, source_v.index_select(0, rows))
