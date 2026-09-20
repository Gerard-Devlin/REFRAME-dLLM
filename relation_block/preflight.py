"""Server-only official parity and real-weight forward/backward smoke gate."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from .common import digest, manifest, snapshot, write_json
from .model import Model, add_lora, clean_mask
from .codec import Codec


def implementation_hashes():
    root = Path(__file__).parent
    return {n: digest(root / n) for n in ("common.py", "prepare.py", "model.py", "codec.py", "train.py", "evaluate.py", "preflight.py")}


def require_gate(data):
    path = Path(data) / "preflight.json"
    if not path.exists():
        raise ValueError("Run preflight first; no training was started")
    gate = json.loads(path.read_text())
    if not gate.get("pass") or gate["data_hash"] != digest(Path(data) / "manifest.json"):
        raise ValueError("Preflight missing/failed or data changed")
    if gate["implementation"] != implementation_hashes():
        raise ValueError("Implementation changed; rerun preflight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    args = p.parse_args()
    m = manifest(args.data)
    gate_path = args.data / "preflight.json"
    write_json(gate_path, {"pass": False, "status": "running"})
    root = snapshot()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
    prompt = tok.apply_chat_template([{"role": "user", "content": "What is two plus three? Explain each step carefully. " * 12}],
                                     tokenize=True, add_generation_prompt=True)
    prompt = prompt[:2 * m["block_size"]]
    x = torch.tensor([prompt], device="cuda")
    torch.manual_seed(1234)
    own = Model.load(root).eval()
    with torch.no_grad():
        positions = torch.arange(x.shape[1], device="cuda")[None]
        ours = own(x, positions, clean_mask(x.shape[1], m["block_size"], "cuda"))[0].float().cpu()
        # Exact caching only at completed block boundaries.
        cut = min(m["block_size"], x.shape[1] - 1)
        if cut != m["block_size"]:
            raise ValueError("Parity prompt too short")
        own_prefill, kv = own(x[:, :cut], positions[:, :cut],
                              clean_mask(cut, m["block_size"], "cuda"), cache=True)
        own_prefill = own_prefill.float().cpu()
        cached = own(x[:, cut:], positions[:, cut:],
                     clean_mask(x.shape[1] - cut, m["block_size"], "cuda", past=cut), past=kv)[0].float().cpu()
    del kv, own
    gc.collect()
    torch.cuda.empty_cache()
    print("Comparing original checkpoint against official pinned modeling code...", flush=True)
    official = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True,
                    local_files_only=True, dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        reference = official(x, use_cache=False, block_size=m["block_size"]).logits.float().cpu()
        official_prefill_output = official(x[:, :cut], use_cache=True,
            update_past_key_values=True, block_size=m["block_size"])
        official_prefill = official_prefill_output.logits.float().cpu()
        official_cached = official(x[:, cut:], use_cache=True,
            past_key_values=official_prefill_output.past_key_values,
            update_past_key_values=False, block_size=m["block_size"]).logits.float().cpu()
    def relative_rms(actual, expected):
        return ((actual - expected).square().mean().sqrt() /
                expected.square().mean().sqrt()).item()
    def top1_agreement(actual, expected):
        return (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()
    full_relative = relative_rms(ours, reference)
    prefill_relative = relative_rms(own_prefill, official_prefill)
    cached_relative = relative_rms(cached, official_cached)
    full_agreement = top1_agreement(ours, reference)
    cached_agreement = top1_agreement(cached, official_cached)
    # Full-sequence and cached BF16 attention use different CUDA query shapes.
    # Their drift is acceptable only when our result matches the official result
    # on each corresponding path; never use cross-path equality as a backend gate.
    official_cache_drift = relative_rms(official_cached, reference[:, cut:])
    official_cache_top1 = top1_agreement(official_cached, reference[:, cut:])
    print(f"Official path parity: full RMS={full_relative:.6g}, prefill RMS={prefill_relative:.6g}, "
          f"cached RMS={cached_relative:.6g}; full/cached top1={full_agreement:.4f}/{cached_agreement:.4f}", flush=True)
    print(f"Official BF16 cached-vs-full diagnostic: RMS={official_cache_drift:.6g}, "
          f"top1={official_cache_top1:.4f}", flush=True)
    if (full_relative > 1e-4 or prefill_relative > 1e-4 or cached_relative > 1e-4 or
            full_agreement < 1.0 or cached_agreement < 1.0):
        raise AssertionError("Official parity failed. Stop; do not train around this failure.")
    del official, official_prefill_output, official_prefill, official_cached
    del reference, ours, own_prefill, cached
    gc.collect()
    torch.cuda.empty_cache()
    from .train import batch, loss
    own = Model.load(root)
    add_lora(own, 16)
    own.gradient_checkpointing = True
    own.train()
    rows = json.loads((args.data / "train.json").read_text())
    raw, response, prefix = batch(rows, [0], m["length"], m["pad_id"], "cuda")
    spec = json.loads((args.data / "codec.json").read_text())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = {}
    for arm in ("token", "relation"):
        own.zero_grad(set_to_none=True)
        codec = Codec(spec, identity=arm == "token").cuda()
        value = loss(own, raw, response, prefix, codec, torch.rand_like(raw, dtype=torch.float),
                     torch.full((1, m["length"] // m["block_size"]), .5, device="cuda"),
                     m["mask_id"], m["block_size"])
        value.backward()
        gradients = [v.grad for v in own.parameters() if v.requires_grad and v.grad is not None]
        if not torch.isfinite(value) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError("Non-finite loss/gradient")
        if not any((g != 0).any() for g in gradients):
            raise AssertionError("All adapter gradients zero")
        losses[arm] = value.item()
    torch.cuda.synchronize()
    gate = dict(pass_=True, data_hash=digest(args.data / "manifest.json"),
                implementation=implementation_hashes(),
                official_full_relative_rms=full_relative,
                official_prefill_relative_rms=prefill_relative,
                official_cached_relative_rms=cached_relative,
                official_full_top1_agreement=full_agreement,
                official_cached_top1_agreement=cached_agreement,
                official_bf16_cache_vs_full_rms=official_cache_drift,
                official_bf16_cache_vs_full_top1=official_cache_top1,
                losses=losses, two_microbatch_backward_seconds=time.perf_counter() - started,
                peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                note="Engineering gate only. No optimizer step; no quality/speedup claim.")
    gate["pass"] = gate.pop("pass_")
    write_json(gate_path, gate)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
