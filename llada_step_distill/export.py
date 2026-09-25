from __future__ import annotations

import gc
from pathlib import Path

import torch

from .core import MASK_ID, atomic_json
from .decode import generate_fixed_quota
from .model import inject_lora, load_adapter, load_model, load_tokenizer, merge_lora


@torch.no_grad()
def export(checkpoint: Path, output: Path, device: str = "cuda:0") -> dict:
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Export output must be new or empty")
    model = load_model(device)
    tokenizer = load_tokenizer()
    inject_lora(model)
    load_adapter(model, checkpoint / "adapter.pt" if checkpoint.is_dir() else checkpoint)
    prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": "What is 2+3?"}], tokenize=True, add_generation_prompt=True)
    probe = torch.tensor([prompt_ids + [MASK_ID] * 32], device=device)
    adapter_logits = model(probe).logits[:, -32:].float().cpu()
    adapter_generation = generate_fixed_quota(model, torch.tensor([prompt_ids], device=device), steps_per_block=16, gen_length=32).output.cpu()
    merged_names = merge_lora(model)
    merged_logits = model(probe).logits[:, -32:].float().cpu()
    merged_generation = generate_fixed_quota(model, torch.tensor([prompt_ids], device=device), steps_per_block=16, gen_length=32).output.cpu()
    relative_rms = float((adapter_logits - merged_logits).pow(2).mean().sqrt() / adapter_logits.pow(2).mean().sqrt().clamp_min(1e-12))
    top1 = float((adapter_logits.argmax(-1) == merged_logits.argmax(-1)).float().mean())
    generation_equal = bool(torch.equal(adapter_generation, merged_generation))
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output)
    del model
    gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoConfig
    from v1.llada.model.modeling_llada import LLaDAModelLM
    config = AutoConfig.from_pretrained(output, local_files_only=True)
    reloaded = LLaDAModelLM.from_pretrained(output, config=config, local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
    reload_logits = reloaded(probe).logits[:, -32:].float().cpu()
    reload_rms = float((merged_logits - reload_logits).pow(2).mean().sqrt() / merged_logits.pow(2).mean().sqrt().clamp_min(1e-12))
    reload_top1 = float((merged_logits.argmax(-1) == reload_logits.argmax(-1)).float().mean())
    report = {
        "merged_modules": len(merged_names), "adapter_merge_relative_rms": relative_rms,
        "adapter_merge_top1": top1, "short_generation_equal": generation_equal,
        "reload_relative_rms": reload_rms, "reload_top1": reload_top1,
        "pass": relative_rms < 1e-3 and top1 == 1.0 and generation_equal and reload_rms < 1e-6 and reload_top1 == 1.0,
    }
    atomic_json(output / "export_validation.json", report)
    if not report["pass"]:
        raise AssertionError(report)
    return report
