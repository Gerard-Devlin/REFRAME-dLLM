import copy
import json
import torch
from relation_block.model import Model
from relation_block.codec import Codec
from relation_block.train import loss
from relation_block.full_state import epoch_batches, lr_scale, save_checkpoint, restore_checkpoint, load_full_model


def model():
    return Model(dict(vocab_size=41, hidden_size=16, intermediate_size=24,
                      num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=1,
                      rms_norm_eps=1e-6, rope_theta=1000000., tie_word_embeddings=True))


def objective(m):
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    response = torch.tensor([[False, False, True, True, True, True, True, True]])
    codec = Codec(dict(block_size=4, vocab_size=41, protected=[39, 40], swaps=[[3, 4, 7]]))
    return loss(m, ids, response, torch.tensor([2]), codec,
                torch.full((1, 8), .4), torch.full((1, 2), .5), 40, 4)


def test_full_gradient_and_shared_head():
    torch.manual_seed(27)
    m = model()
    m.gradient_checkpointing = True
    objective(m).backward()
    assert m.lm_head.weight is m.model.embed_tokens.weight
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    for name in ("model.embed_tokens.weight", "model.layers.0.mlp.up_proj.weight", "model.norm.weight"):
        assert dict(m.named_parameters())[name].grad.abs().sum() > 0


def test_full_checkpoint_resume_matches_next_update_and_export(tmp_path):
    torch.manual_seed(22)
    m = model()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5)
    objective(m).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    meta = dict(completed_steps=1, arm="token", status="running")
    saved = save_checkpoint(tmp_path, m, opt, meta, 0, 1)
    expected_rng = torch.rand(5)
    objective(m).backward(); opt.step()
    expected = copy.deepcopy(m.state_dict())
    restored = model()
    opt2 = torch.optim.AdamW(restored.parameters(), lr=2e-5)
    restore_checkpoint(saved, restored, opt2, 0, 1, {"arm": "token"})
    torch.testing.assert_close(torch.rand(5), expected_rng, rtol=0, atol=0)
    objective(restored).backward(); opt2.step()
    for name, x in restored.state_dict().items():
        torch.testing.assert_close(x, expected[name], rtol=0, atol=0)
    export, metadata = load_full_model(saved, "cpu")
    assert export.lm_head.weight is export.model.embed_tokens.weight
    assert all(p.dtype == torch.bfloat16 for p in export.parameters())
    original = torch.load(saved / "model_fp32.pt", weights_only=True)
    for name, x in export.state_dict().items():
        torch.testing.assert_close(x, original[name].bfloat16(), rtol=0, atol=0)


def test_epoch_tail_and_resume_order():
    batches = epoch_batches(29, 12, 1234)
    assert [len(b) for b in batches] == [12, 12, 5]
    assert sorted(sum(batches, [])) == list(range(29))
    assert batches[1:] == epoch_batches(29, 12, 1234)[1:]
    assert batches != epoch_batches(29, 12, 1235)
    assert lr_scale(0, 1000) == 1 / 30
    assert lr_scale(999, 1000) == 0


def test_checkpoint_retention_and_legacy_rejection(tmp_path):
    import pytest
    m = model(); opt = torch.optim.AdamW(m.parameters())
    for i in range(1, 4):
        save_checkpoint(tmp_path, m, opt, dict(completed_steps=i), 0, 1)
    assert len(list(tmp_path.glob("step_*"))) == 2
    assert json.loads((tmp_path / "latest.json").read_text())["checkpoint"] == "step_00000003"
    with pytest.raises(ValueError, match="LoRA"):
        restore_checkpoint(tmp_path / "checkpoint.pt", m, opt, 0, 1, {})


def test_fp32_master_bf16_autocast_optimizer():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    m = model().cuda()
    m.gradient_checkpointing = True
    # Exercise selected-head loss under the same precision policy as real training.
    ids = torch.randint(0, 39, (1, 8), device="cuda")
    from relation_block.model import clean_mask
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5)
    before = m.model.embed_tokens.weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v = m(ids, torch.arange(8, device="cuda")[None], clean_mask(8, 4, "cuda"),
              select=(torch.tensor([0, 0], device="cuda"), torch.tensor([3, 5], device="cuda")),
              targets=torch.tensor([7, 9], device="cuda"))
    v.backward(); opt.step()
    assert not torch.equal(before, m.model.embed_tokens.weight)
    assert all(p.dtype == torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert all(state["exp_avg"].dtype == torch.float32 for state in opt.state.values())


def test_real_train_loop_full_epoch_tail_resume(tmp_path, monkeypatch):
    import pytest
    import sys
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("tensorboard")
    from safetensors.torch import save_file
    from relation_block import train, preflight
    from relation_block.full_state import resolve_checkpoint
    source = tmp_path / "source"; source.mkdir()
    data = tmp_path / "data"; data.mkdir()
    base = model()
    (source / "config.json").write_text(json.dumps(base.config))
    save_file({n: p.detach().clone() for n, p in base.state_dict().items() if n != "lm_head.weight"}, str(source / "model.safetensors"))
    rows = [{"ids": [2, 3, 4, 5, 6, 7, 8, 9], "prefix": 2} for _ in range(5)]
    (data / "train.json").write_text(json.dumps(rows))
    (data / "gsm8k_dev_full.json").write_text("[]")
    (data / "codec.json").write_text(json.dumps(dict(block_size=4, vocab_size=41, protected=[39, 40], swaps=[[3, 4, 7]])))
    (data / "manifest.json").write_text(json.dumps(dict(files={}, length=8, pad_id=39, mask_id=40, block_size=4, revision="tiny")))
    monkeypatch.setattr(train, "snapshot", lambda: source)
    monkeypatch.setattr(preflight, "require_gate", lambda _: None)
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    common = ["train", "--data", str(data), "--arm", "relation", "--global-batch", "2", "--smoke", "--eval-every", "0"]
    continuous, interrupted = tmp_path / "continuous", tmp_path / "interrupted"
    def run(output, extra):
        monkeypatch.setattr(sys, "argv", common + ["--output", str(output)] + extra)
        train.main()
    run(continuous, [])
    run(interrupted, ["--stop-after", "1"])
    run(interrupted, ["--resume", str(interrupted)])
    a, b = resolve_checkpoint(continuous), resolve_checkpoint(interrupted)
    ma, mb = (torch.load(p / "model_fp32.pt", weights_only=True, map_location="cpu") for p in (a, b))
    for name in ma:
        torch.testing.assert_close(ma[name], mb[name], rtol=0, atol=0)
    status = json.loads((interrupted / "status.json").read_text())
    assert status["completed_steps"] == 3 and status["sampler_cursor"] == 5
    assert status["original_tokens"] == 40 and status["supervised_tokens"] == 30
    assert status["trainable_parameters"] == status["total_parameters"]
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(interrupted / "tensorboard")).Reload()
    assert [event.step for event in events.Scalars("train/loss")] == [1, 2, 3]
    assert "performance/eta_seconds" in events.Tags()["scalars"]
