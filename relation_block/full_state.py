"""Full-training checkpoints: shared FP32 model plus one optimizer shard per rank."""
import json
import random
import shutil
from pathlib import Path
import torch
import torch.distributed as dist
from .common import write_json


def barrier():
    if dist.is_initialized():
        dist.barrier()


def epoch_batches(size, batch_size, seed):
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return [order[i:i + batch_size] for i in range(0, size, batch_size)]


def lr_scale(step, total):
    import math
    warmup = max(1, math.ceil(.03 * total))
    if step < warmup:
        return (step + 1) / warmup
    return .5 * (1 + math.cos(math.pi * (step - warmup + 1) / max(1, total - warmup)))


def local_optimizer(optimizer):
    return getattr(optimizer, "optim", optimizer)


def save_checkpoint(output, model, optimizer, metadata, rank, world):
    output = Path(output)
    output = output.resolve()
    target = output / f"step_{metadata['completed_steps']:08d}"
    staging = output / (target.name + ".incomplete")
    if rank == 0:
        if staging.exists():
            if staging.resolve().parent != output:
                raise ValueError("Checkpoint staging path escapes output directory")
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
    barrier()
    state = dict(optimizer=local_optimizer(optimizer).state_dict(),
                 torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                 cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
    torch.save(state, staging / f"rank_{rank}.pt")
    if rank == 0:
        torch.save(model.state_dict(), staging / "model_fp32.pt")
        # Streaming one tensor at a time into the CPU export avoids a second GPU model.
        export = {k: v.detach().to(device="cpu", dtype=torch.bfloat16) for k, v in model.state_dict().items()}
        torch.save(export, staging / "model_bf16.pt")
        write_json(staging / "config.json", model.config)
        write_json(staging / "metadata.json", dict(metadata, world_size=world, format="full-v1"))
    barrier()
    if rank == 0:
        if target.exists():
            raise ValueError(f"Checkpoint already exists: {target}")
        staging.rename(target)
        write_json(output / "latest.json", {"checkpoint": target.name})
        write_json(output / "status.json", dict(metadata, world_size=world, format="full-v1"))
        checkpoints = sorted(output.glob("step_[0-9]*"))
        checkpoints = [p for p in checkpoints if (p / "metadata.json").exists() and not p.name.endswith(".incomplete")]
        for old in checkpoints[:-2]:
            if old.resolve().parent != output:
                raise ValueError("Checkpoint retention path escapes output directory")
            shutil.rmtree(old)
    barrier()
    return target


def resolve_checkpoint(path):
    path = Path(path)
    if (path / "latest.json").exists():
        path = path / json.loads((path / "latest.json").read_text())["checkpoint"]
    if not (path / "metadata.json").exists() or path.name.endswith(".incomplete"):
        raise ValueError("Expected a completed full-v1 checkpoint directory; LoRA checkpoints cannot resume full training")
    return path


def restore_checkpoint(path, model, optimizer, rank, world, expected):
    path = resolve_checkpoint(path)
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("format") != "full-v1" or meta["world_size"] != world:
        raise ValueError("Full checkpoint format or GPU count differs")
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Resume metadata differs: {key}")
    model.load_state_dict(torch.load(path / "model_fp32.pt", map_location="cpu", weights_only=True))
    state = torch.load(path / f"rank_{rank}.pt", map_location="cpu", weights_only=True)
    local_optimizer(optimizer).load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"])
    return meta


def load_full_model(path, device="cuda"):
    from .model import Model
    path = resolve_checkpoint(path)
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("format") != "full-v1":
        raise ValueError("Expected full-v1 weights")
    with torch.device("meta"):
        model = Model(json.loads((path / "config.json").read_text()))
    model.load_state_dict(torch.load(path / "model_bf16.pt", map_location="cpu", weights_only=True), assign=True)
    if model.config.get("tie_word_embeddings", False):
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.to(device).eval(), meta
