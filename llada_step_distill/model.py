from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import MODEL_ID, REVISION

TARGET_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"})
EXPECTED_RANK256_PARAMETERS = 671_088_640


def snapshot() -> Path:
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    except Exception:
        import os

        cache = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub"))
        path = cache.expanduser() / "models--GSAI-ML--LLaDA-8B-Instruct" / "snapshots" / REVISION
    if not (path / "config.json").is_file():
        raise RuntimeError(f"Pinned checkpoint is incomplete: {path}")
    return path


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot(), local_files_only=True)


def load_model(device: torch.device | str, *, training: bool = False, model_path: Path | None = None):
    from transformers import AutoConfig
    from v1.llada.model.modeling_llada import ActivationCheckpointingStrategy, LLaDAModelLM

    root = Path(model_path) if model_path is not None else snapshot()
    config = AutoConfig.from_pretrained(root, local_files_only=True)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(
        root, config=config, local_files_only=True, dtype=torch.bfloat16
    ).to(device)
    model.requires_grad_(False)
    if training:
        model.train()
        model.model.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    else:
        model.eval()
    return model


class LoRALinear(nn.Module):
    """A mergeable FP32 LoRA branch around one frozen LLaDA projection."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.enabled = True
        base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        with torch.autocast(device_type=x.device.type, enabled=x.device.type == "cuda", dtype=torch.bfloat16):
            update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return out + update.to(out.dtype) * self.scale

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        delta = self.lora_B.float().matmul(self.lora_A.float()).mul_(self.scale)
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        return self.base


def _parent_and_name(root: nn.Module, qualified: str) -> tuple[nn.Module, str]:
    pieces = qualified.split(".")
    parent = root
    for piece in pieces[:-1]:
        parent = getattr(parent, piece)
    return parent, pieces[-1]


def inject_lora(model: nn.Module, rank: int = 256, alpha: int = 512, dropout: float = 0.0) -> list[str]:
    targets = [
        (name, module) for name, module in model.named_modules()
        if (isinstance(module, nn.Linear)
            and name.rsplit(".", 1)[-1] in TARGET_MODULES
            and name != "model.transformer.ff_out")
    ]
    if len(targets) != 32 * 7:
        raise AssertionError(f"Expected 224 target projections, found {len(targets)}")
    names = []
    for name, module in targets:
        parent, child = _parent_and_name(model, name)
        setattr(parent, child, LoRALinear(module, rank, alpha, dropout))
        names.append(name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 256 and trainable != EXPECTED_RANK256_PARAMETERS:
        raise AssertionError(f"rank-256 trainable parameters {trainable:,} != {EXPECTED_RANK256_PARAMETERS:,}")
    return names


def lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


@contextlib.contextmanager
def lora_enabled(model: nn.Module, enabled: bool):
    modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
    prior = [module.enabled for module in modules]
    for module in modules:
        module.enabled = enabled
    try:
        yield
    finally:
        for module, value in zip(modules, prior):
            module.enabled = value


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.endswith("lora_A") or key.endswith("lora_B")
    }


def load_adapter(model: nn.Module, path_or_state, strict: bool = True) -> None:
    state = torch.load(path_or_state, map_location="cpu", weights_only=False) if isinstance(path_or_state, (str, Path)) else path_or_state
    if "adapter" in state:
        state = state["adapter"]
    result = model.load_state_dict(state, strict=False)
    missing_lora = [key for key in result.missing_keys if key.endswith(("lora_A", "lora_B"))]
    unexpected = [key for key in result.unexpected_keys if key.endswith(("lora_A", "lora_B"))]
    if strict and (missing_lora or unexpected):
        raise RuntimeError(f"Adapter mismatch: missing={missing_lora[:3]} unexpected={unexpected[:3]}")


def merge_lora(model: nn.Module) -> list[str]:
    merged = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent, child = _parent_and_name(model, name)
            setattr(parent, child, module.merge())
            merged.append(name)
    return merged


def detach_output_head(model) -> nn.Module:
    head = model.model.transformer.ff_out
    if not isinstance(head, nn.Linear):
        raise TypeError("Output head was already detached or wrapped")
    model.model.transformer.ff_out = nn.Identity()
    head.requires_grad_(False)
    return head


def hidden_states(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Requires the output head to have been replaced by Identity."""
    return model(input_ids=input_ids, use_cache=False).logits


def chunked_logsumexp(hidden: torch.Tensor, head: nn.Linear, chunk_size: int = 4096) -> torch.Tensor:
    parts = []
    for start in range(0, head.weight.shape[0], chunk_size):
        logits = F.linear(hidden, head.weight[start : start + chunk_size])
        parts.append(torch.logsumexp(logits.float(), dim=-1))
    return torch.logsumexp(torch.stack(parts, dim=-1), dim=-1)


def selected_log_probs(hidden: torch.Tensor, token_ids: torch.Tensor, head: nn.Linear, chunk_size: int = 4096) -> torch.Tensor:
    log_z = chunked_logsumexp(hidden, head, chunk_size)
    selected_weight = head.weight.index_select(0, token_ids.reshape(-1)).reshape(*token_ids.shape, -1)
    if token_ids.ndim == hidden.ndim - 1:
        selected_logits = (hidden.float() * selected_weight.float()).sum(-1)
        return selected_logits - log_z
    selected_logits = (hidden.unsqueeze(-2).float() * selected_weight.float()).sum(-1)
    return selected_logits - log_z.unsqueeze(-1)


def max_log_probs(hidden: torch.Tensor, head: nn.Linear, chunk_size: int = 4096) -> torch.Tensor:
    log_z = chunked_logsumexp(hidden, head, chunk_size)
    maxima = []
    for start in range(0, head.weight.shape[0], chunk_size):
        maxima.append(F.linear(hidden, head.weight[start : start + chunk_size]).float().amax(-1))
    return torch.stack(maxima).amax(0) - log_z


@torch.no_grad()
def topk_distribution(hidden: torch.Tensor, head: nn.Linear, k: int = 32, chunk_size: int = 4096):
    log_z = chunked_logsumexp(hidden, head, chunk_size)
    values = torch.full((*hidden.shape[:-1], k), -torch.inf, device=hidden.device)
    indices = torch.zeros((*hidden.shape[:-1], k), dtype=torch.long, device=hidden.device)
    for start in range(0, head.weight.shape[0], chunk_size):
        logits = F.linear(hidden, head.weight[start : start + chunk_size]).float()
        local_values, local_indices = logits.topk(min(k, logits.shape[-1]), dim=-1)
        candidate_values = torch.cat((values, local_values), dim=-1)
        candidate_indices = torch.cat((indices, local_indices + start), dim=-1)
        values, order = candidate_values.topk(k, dim=-1)
        indices = candidate_indices.gather(-1, order)
    log_probs = values - log_z.unsqueeze(-1)
    residual = torch.clamp(1.0 - log_probs.exp().sum(-1), min=1e-12).log()
    return indices, log_probs, residual
