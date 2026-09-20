"""Optional real two-process CPU/Gloo test, no model or dataset downloads."""
import os
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from relation_block.codec import Codec
from relation_block.model import Model, add_lora
from relation_block.train import loss


def setup_model():
    torch.set_num_threads(1)
    torch.manual_seed(15)
    c = dict(vocab_size=41, hidden_size=32, intermediate_size=64, num_attention_heads=4,
             num_key_value_heads=2, num_hidden_layers=2, rms_norm_eps=1e-6,
             rope_theta=1000000., tie_word_embeddings=True)
    m = Model(c)
    m.gradient_checkpointing = True
    return m


def inputs():
    rng = torch.Generator().manual_seed(18)
    ids = torch.randint(0, 39, (4, 16), generator=rng)
    prefix = torch.tensor([3, 4, 5, 6])
    response = torch.arange(16)[None] >= prefix[:, None]
    noise = torch.rand(4, 16, generator=rng)
    prob = torch.full((4, 2), .6)
    codec = Codec(dict(block_size=8, vocab_size=41, protected=[39, 40], swaps=[[3, 4, 7]]))
    return ids, prefix, response, noise, prob, codec


def worker(rank, store, output):
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=2)
    m = setup_model()
    ddp = DistributedDataParallel(m, broadcast_buffers=False)
    from torch.distributed.optim import ZeroRedundancyOptimizer
    from relation_block.full_state import save_checkpoint, restore_checkpoint
    optimizer = ZeroRedundancyOptimizer(m.parameters(), optimizer_class=torch.optim.AdamW, lr=2e-5)
    ids, prefix, response, noise, prob, codec = inputs()
    for j in range(2):
        i = j*2 + rank
        from contextlib import nullcontext
        with ddp.no_sync() if j == 0 else nullcontext():
            value = loss(ddp, ids[i:i+1], response[i:i+1], prefix[i:i+1], codec,
                         noise[i:i+1], prob[i:i+1], 40, 8)
            (value * response[i:i+1].sum() * 2 / response.sum()).backward()
    if rank == 0:
        torch.save({n: p.grad for n, p in m.named_parameters() if p.requires_grad}, output)
    optimizer.step()
    saved = save_checkpoint(Path(output).parent / "full", m, optimizer, {"completed_steps": 1}, rank, 2)
    optimizer.zero_grad(set_to_none=True)
    # Restore into a fresh DDP + ZeRO instance and check the next Adam update.
    restored = setup_model()
    restored_ddp = DistributedDataParallel(restored, broadcast_buffers=False)
    opt2 = ZeroRedundancyOptimizer(restored.parameters(), optimizer_class=torch.optim.AdamW, lr=2e-5)
    restore_checkpoint(saved, restored, opt2, rank, 2, {})
    for net, opt in ((ddp, optimizer), (restored_ddp, opt2)):
        # Uneven tail: 3 real examples, rank 1's final slot is zero-weight padding.
        for j in range(2):
            slot = j * 2 + rank
            i = min(slot, 2)
            value = loss(net, ids[i:i+1], response[i:i+1], prefix[i:i+1], codec,
                         noise[i:i+1], prob[i:i+1], 40, 8)
            (value * response[i:i+1].sum() * 2 / response[:3].sum() * int(slot < 3)).backward()
        opt.step()
    for a, b in zip(m.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if rank == 0:
        torch.save(m.state_dict(), output + ".final")
    dist.destroy_process_group()


@pytest.mark.skipif(os.getenv("RUN_DDP_TESTS") != "1", reason="Set RUN_DDP_TESTS=1 for two-process Gloo")
def test_two_process_matches_global_batch(tmp_path):
    store = (tmp_path / "store").resolve().as_uri()
    output = str(tmp_path / "grads.pt")
    mp.spawn(worker, args=(store, output), nprocs=2, join=True)
    m = setup_model()
    ids, prefix, response, noise, prob, codec = inputs()
    loss(m, ids, response, prefix, codec, noise, prob, 40, 8).backward()
    grads = torch.load(output, weights_only=True)
    for n, p in m.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, grads[n], atol=3e-5, rtol=3e-5)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5)
    opt.step(); opt.zero_grad(set_to_none=True)
    loss(m, ids[:3], response[:3], prefix[:3], codec, noise[:3], prob[:3], 40, 8).backward()
    opt.step()
    distributed = torch.load(output + ".final", weights_only=True)
    for name, value in m.state_dict().items():
        torch.testing.assert_close(value, distributed[name], atol=2e-6, rtol=2e-6)
