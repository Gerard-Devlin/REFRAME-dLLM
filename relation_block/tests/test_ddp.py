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
    add_lora(m, 4)
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
