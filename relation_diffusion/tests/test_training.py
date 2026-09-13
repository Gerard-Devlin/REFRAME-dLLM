import itertools
import json
import os
import socket
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from relation_diffusion.codec import Codec, SparseCoupling
from relation_diffusion.compare import compare
from relation_diffusion.neural import Denoiser, ModelConfig, masked_loss, path_nll, sample, update_batch
from relation_diffusion.torch_codec import TorchCodec, codec_spec

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]


def run(module, *args, env=None):
    return subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT,
                          env=env, check=True, capture_output=True, text=True, timeout=120)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("relation_data")
    train, valid = root / "train.txt", root / "valid.txt"
    train.write_text("\n".join(f"Training document number {i}: cats walk through a quiet garden." for i in range(40)), encoding="utf-8")
    valid.write_text("\n".join(f"Heldout document {i}: fish swim under a blue river." for i in range(15)), encoding="utf-8")
    output = root / "data"
    run("relation_diffusion.prepare", "--train-text", train, "--validation-text", valid,
        "--output", output, "--length", 32, "--prefix", 8, "--fit-blocks", 32)
    return output


@pytest.mark.parametrize("kind", ["identity", "rename", "random", "relation"])
def test_torch_codec_inverse_and_protected_prefix(kind):
    rng = np.random.default_rng(19)
    fit = rng.integers(0, 257, (64, 12))
    fit[:, 7] = 256
    spec = codec_spec(kind, fit, 2, depth=2)
    codec = TorchCodec(spec)
    x = torch.from_numpy(fit)
    z = codec.encode(x)
    assert torch.equal(codec.decode(z), x)
    assert torch.equal(z[:, :2], x[:, :2])
    assert torch.equal(z == 256, x == 256)
    if kind in {"random", "relation"}:
        reference = Codec(257, tuple(SparseCoupling(257, layer["pairs"], layer["tables"]) for layer in spec["layers"]))
        np.testing.assert_array_equal(z.numpy(), reference.encode(fit, prefix_len=2))
    if torch.cuda.is_available():
        assert torch.equal(codec.cuda().encode(x.cuda()).cpu(), z)
        assert torch.equal(codec.decode(z.cuda()).cpu(), x)


def test_fixed_path_defines_normalized_distribution_and_no_target_leak():
    torch.manual_seed(3)
    model = Denoiser(ModelConfig(vocab_size=2, length=4, width=8, layers=1, heads=2)).eval()
    x = torch.tensor([[0, *seq] for seq in itertools.product(range(2), repeat=3)])
    for steps in (1, 2, 3):
        nll = path_nll(model, x, prefix=1, steps=steps)
        assert torch.allclose((-nll).exp().sum(), torch.tensor(1.), atol=2e-6)
    seen = []
    handle = model.register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
    path_nll(model, x, prefix=1, steps=3)
    handle.remove()
    for i, state in enumerate(seen):
        assert torch.equal(state[:, :1 + i], x[:, :1 + i])
        assert torch.all(state[:, 1 + i:] == model.cfg.vocab_size)
    generated = sample(model, x[:2, :1], length=4, steps=2)
    assert generated.max() < 2 and torch.equal(generated[:, :1], x[:2, :1])


def test_zero_masks_is_connected_and_global_noise_is_partitionable():
    logits = torch.randn(2, 4, 3, requires_grad=True)
    loss = masked_loss(logits, torch.zeros(2, 4, dtype=torch.long),
                       torch.zeros(2, 4, dtype=torch.bool), torch.ones(2, 1), 1)
    loss.backward()
    assert loss == 0 and torch.isfinite(logits.grad).all()
    ids, mask, probability = update_batch(100, 12, 16, 4, 1234, 8)
    ids2, mask2, p2 = update_batch(100, 12, 16, 4, 1234, 8)
    assert torch.equal(ids, ids2) and torch.equal(mask, mask2) and torch.equal(probability, p2)
    assert not mask[:, :4].any()
    _, allmask, _ = update_batch(100, 12, 16, 4, 1234, 8, "one-step")
    assert allmask[:, 4:].all() and not allmask[:, :4].any()


def training_args(data, output):
    return ["--data", data, "--output", output, "--device", "cpu", "--dtype", "float32",
            "--width", 16, "--heads", 2, "--layers", 1, "--steps", 4,
            "--global-batch", 4, "--micro-batch", 1, "--codec", "relation2", "--log-every", 1]


def test_train_resume_and_real_checkpoint_evaluation(prepared, tmp_path):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run("relation_diffusion.train", *training_args(prepared, full))
    run("relation_diffusion.train", *training_args(prepared, resumed), "--max-seconds", 0.000001)
    assert json.loads((resumed / "status.json").read_text())["completed_steps"] == 1
    run("relation_diffusion.train", *training_args(prepared, resumed), "--resume")
    a = torch.load(full / "checkpoint.pt", weights_only=False)
    b = torch.load(resumed / "checkpoint.pt", weights_only=False)
    assert a["step"] == b["step"] == 4
    for name, tensor in a["model"].items():
        assert torch.equal(tensor, b["model"][name]), name
    output = tmp_path / "evaluation.json"
    run("relation_diffusion.evaluate", "--checkpoint", full / "checkpoint.pt", "--data", prepared,
        "--output", output, "--device", "cpu", "--limit", 3, "--batch", 2,
        "--repeats", 1, "--steps", "1,2,4")
    result = json.loads(output.read_text())
    assert result["examples"] == 3 and len(result["scores"]) == 3
    assert all(np.isfinite(r["path_bits_per_symbol"]) for r in result["scores"])
    assert result["error_spread"]["max_changed_original_positions"] <= 4
    run("relation_diffusion.preflight", "--data", prepared, "--output", tmp_path / "preflight.json",
        "--device", "cpu", "--dtype", "float32", "--width", 16, "--heads", 2,
        "--layers", 1, "--repeats", 1)


def test_refuses_unmatched_comparisons():
    from relation_diffusion.compare import MATCH
    base = {k: 1 for k in MATCH}
    base.update(seed=1234, codec="identity", objective="diffusion", scores=[])
    other = dict(base, codec="relation2", trained_steps=2)
    with pytest.raises(ValueError, match="trained_steps"):
        compare([base, other])


@pytest.mark.skipif(os.environ.get("RUN_DDP_TESTS") != "1", reason="Opt-in two-process Gloo integration")
def test_ddp_matches_single_process(prepared, tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    run("relation_diffusion.train", *training_args(prepared, one))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
    if sys.platform == "win32":
        # Windows torchrun's static rendezvous can hardcode an unavailable
        # libuv store. Exercise the same DDP trainer with two env:// workers.
        processes = []
        for rank in range(2):
            env = dict(os.environ, USE_LIBUV="0", OMP_NUM_THREADS="1", MASTER_ADDR="127.0.0.1",
                       MASTER_PORT=port, WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
            processes.append(subprocess.Popen([sys.executable, "-m", "relation_diffusion.train",
                                               *map(str, training_args(prepared, two))],
                                              cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        try:
            for proc in processes:
                stdout, stderr = proc.communicate(timeout=60)
                assert proc.returncode == 0, stderr.decode(errors="replace")
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
    else:
        run("torch.distributed.run", "--nnodes=1", "--nproc_per_node=2", "--master-addr=127.0.0.1",
            "--master-port", port, "-m", "relation_diffusion.train", *training_args(prepared, two))
    a = torch.load(one / "checkpoint.pt", weights_only=False)
    b = torch.load(two / "checkpoint.pt", weights_only=False)
    for name, tensor in a["model"].items():
        # The attention key-bias is a softmax-invariant direction. FP32
        # reduction-order noise in its near-zero gradient is amplified by Adam.
        # Also compare actual logits below, not only parameter distances.
        torch.testing.assert_close(tensor, b["model"][name], atol=1e-5, rtol=2e-5)
    ma = Denoiser(ModelConfig(**a["metadata"]["model"])).eval()
    mb = Denoiser(ModelConfig(**b["metadata"]["model"])).eval()
    ma.load_state_dict(a["model"])
    mb.load_state_dict(b["model"])
    torch.manual_seed(9)
    tokens = torch.randint(258, (3, 32))
    torch.testing.assert_close(ma(tokens), mb(tokens), atol=2e-6, rtol=2e-5)
