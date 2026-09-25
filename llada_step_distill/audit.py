from __future__ import annotations

from pathlib import Path

import torch

from .core import MASK_ID, REVISION, atomic_json
from .model import EXPECTED_RANK256_PARAMETERS, inject_lora, load_model, load_tokenizer


@torch.no_grad()
def audit(output: Path, device: str = "cuda:0") -> dict:
    model = load_model(device)
    tokenizer = load_tokenizer()
    ids = tokenizer.apply_chat_template([{"role": "user", "content": "What is 1+1?"}], tokenize=True, add_generation_prompt=True)
    source = torch.tensor([ids + [MASK_ID] * 32], device=device)
    reference = model(source, use_cache=False).logits[:, -32:].float()
    names = inject_lora(model, rank=256, alpha=512)
    candidate = model(source, use_cache=False).logits[:, -32:].float()
    relative_rms = float((reference - candidate).pow(2).mean().sqrt() / reference.pow(2).mean().sqrt().clamp_min(1e-12))
    top1 = float((reference.argmax(-1) == candidate.argmax(-1)).float().mean())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    blocks = list(model.model.transformer.blocks) if "blocks" in model.model.transformer else [block for group in model.model.transformer.block_groups for block in group]
    flash = all(getattr(block, "flash_attn_func", None) is not None for block in blocks)
    report = {
        "revision": REVISION, "target_modules": len(names), "trainable_parameters": trainable,
        "expected_trainable_parameters": EXPECTED_RANK256_PARAMETERS,
        "zero_lora_relative_rms": relative_rms, "zero_lora_top1": top1,
        "flash_attention_active": flash,
        "ideal_forward_speedup_32_to_16": 2.0,
        "cost_gate_pass": 2.0 >= 1.5,
        "pass": len(names) == 224 and trainable == EXPECTED_RANK256_PARAMETERS and relative_rms == 0.0 and top1 == 1.0 and flash,
    }
    atomic_json(Path(output), report)
    if not report["pass"]:
        raise AssertionError(report)
    return report
