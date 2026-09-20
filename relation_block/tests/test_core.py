import copy
import json
from pathlib import Path
import pytest
import torch
from relation_block.codec import Codec, fit
from relation_block.model import Model, LoRA, add_lora, clean_mask, training_mask, adapter_state, load_adapter
from relation_block.train import loss
from relation_block.evaluate import answer, generate


def cfg():
    return dict(vocab_size=41, hidden_size=32, intermediate_size=64,
                num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
                rms_norm_eps=1e-6, rope_theta=1000000., tie_word_embeddings=True,
                hidden_act="silu", max_position_embeddings=1024, attention_dropout=0.,
                use_sliding_window=False)


def spec():
    return dict(block_size=8, vocab_size=41, protected=[39, 40],
                swaps=[[3, 4, 7], [8, 9, 7]])


def test_codec_roundtrip_prefix_boundaries_specials():
    codec = Codec(spec())
    x = torch.randint(0, 41, (9, 32))
    x[:, 1:3] = torch.tensor([3, 4])
    prefix = torch.arange(9)
    z = codec(x, prefix)
    assert torch.equal(codec(z, prefix), x)
    assert z[0, 2] == 7
    assert z[2, 2] == 4  # Pair overlaps prompt => do not transform.
    assert torch.equal(z[:, ::8], x[:, ::8])
    special = (x == 39) | (x == 40)
    assert torch.equal(z[special], x[special])
    for i in range(4):
        assert torch.equal(codec(x[:, i*8:(i+1)*8], prefix, offset=i*8), z[:, i*8:(i+1)*8])


def test_single_error_is_local():
    codec = Codec(spec())
    x = torch.tensor([[1, 3, 4, 8, 9, 2, 1, 1]])
    z = codec(x, 0)
    for i in range(8):
        wrong = z.clone()
        wrong[0, i] = (wrong[0, i] + 1) % 39
        assert (codec(wrong, 0) != x).sum() <= 2


def test_sparse_fit_deterministic_and_train_only():
    rows = [{"ids": [1, 3, 4, 8, 7, 8, 7, 1], "prefix": 0}] * 16
    a = fit(rows, 8, 41, [39, 40], min_count=2)
    assert a == fit(rows[::-1], 8, 41, [39, 40], min_count=2)
    assert a["swaps"] == [[3, 4, 7]]
    assert Codec(a).left.numel() == 41


def test_no_clean_target_leakage_transitive():
    length, block = 24, 8
    mask = training_mask(length, block, "cpu")[0, 0]
    reach = mask.clone()
    for _ in range(4):
        reach = reach | ((reach.float() @ mask.float()) > 0)
    for i in range(length):
        # No noisy query sees clean current/future blocks at ANY depth.
        first_forbidden = length + (i // block) * block
        assert not reach[i, first_forbidden:].any()
    # Clean head at end of block cannot see first token of next block.
    assert not reach[length + 7, length + 8:].any()


def test_cached_equals_uncached():
    torch.manual_seed(5)
    m = Model(cfg()).eval()
    x = torch.randint(0, 39, (1, 24))
    pos = torch.arange(24)[None]
    whole = m(x, pos, clean_mask(24, 8, "cpu"))[0]
    _, kv = m(x[:, :16], pos[:, :16], clean_mask(16, 8, "cpu"), cache=True)
    tail = m(x[:, 16:], pos[:, 16:], clean_mask(8, 8, "cpu", past=16), past=kv)[0]
    torch.testing.assert_close(tail, whole[:, 16:], atol=2e-5, rtol=2e-5)


def test_qwen_reference_parity():
    # Independent implementation reference. No downloads, random tiny weights.
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(7)
    c = Qwen2Config(**cfg())
    c._attn_implementation = "sdpa"
    reference = Qwen2ForCausalLM(c).eval()
    m = Model(cfg()).eval()
    m.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.randint(0, 39, (2, 24))
    allowed = clean_mask(24, 8, "cpu")
    additive = torch.where(allowed, 0., torch.finfo(torch.float32).min)
    with torch.no_grad():
        y = reference(ids, attention_mask=additive, use_cache=False).logits
        ours = m(ids, torch.arange(24)[None].expand(2, -1), allowed)[0]
    torch.testing.assert_close(ours, y, atol=2e-5, rtol=2e-5)


def test_strict_meta_safetensors_load(tmp_path):
    from safetensors.torch import save_file
    m = Model(cfg())
    weights = {k: v.detach().clone() for k, v in m.state_dict().items() if k != "lm_head.weight"}
    (tmp_path / "config.json").write_text(json.dumps(cfg()))
    save_file(weights, str(tmp_path / "model.safetensors"))
    loaded = Model.load(tmp_path, "cpu", torch.float32)
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    for k, v in m.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v)
    weights["unexpected.weight"] = torch.zeros(1)
    save_file(weights, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        Model.load(tmp_path, "cpu", torch.float32)


def test_lora_loss_gradients_checkpoint_and_roundtrip(tmp_path):
    torch.manual_seed(9)
    m = Model(cfg())
    base = copy.deepcopy(m.state_dict())
    add_lora(m, 4)
    m.gradient_checkpointing = True
    ids = torch.randint(0, 38, (2, 16))
    response = torch.arange(16)[None, :] >= torch.tensor([4, 5])[:, None]
    prefix = torch.tensor([4, 5])
    v = loss(m, ids, response, prefix, Codec(spec()), torch.rand(2, 16), torch.full((2, 2), .5), 40, 8)
    v.backward()
    assert torch.isfinite(v)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters() if p.requires_grad)
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.001).step()
    path = tmp_path / "checkpoint.pt"
    torch.save({"meta": {"rank": 4}, "adapter": adapter_state(m)}, path)
    restored = Model(cfg())
    restored.load_state_dict(base)
    load_adapter(restored, path)
    m.eval(); restored.eval()
    pos, mask = torch.arange(16)[None], clean_mask(16, 8, "cpu")
    torch.testing.assert_close(m(ids, pos, mask)[0], restored(ids, pos, mask)[0])


def test_lora_merged_adapter_matches_unmerged(tmp_path):
    torch.manual_seed(19)
    original = Model(cfg()).eval()
    base = copy.deepcopy(original)
    add_lora(base, 4)
    with torch.no_grad():
        for module in base.modules():
            if isinstance(module, LoRA):
                module.b.normal_(std=.01)
    path = tmp_path / "checkpoint.pt"
    torch.save({"meta": {"rank": 4}, "adapter": adapter_state(base)}, path)
    merged = copy.deepcopy(original)
    load_adapter(merged, path, merge=True)
    assert not any(isinstance(x, LoRA) for x in merged.modules())
    ids = torch.randint(0, 39, (2, 16))
    pos, mask = torch.arange(16)[None].expand(2, -1), clean_mask(16, 8, "cpu")
    torch.testing.assert_close(base(ids, pos, mask)[0], merged(ids, pos, mask)[0], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("prefix_length", [5, 8, 11, 16])
def test_free_generation_no_masks_cache_accounting(prefix_length):
    torch.manual_seed(13)
    m = Model(cfg()).eval()
    out, calls = generate(m, [2] * prefix_length, Codec(spec()), 40, 39, 4, 17)
    assert 40 not in out
    assert len(out) <= 17
    assert calls["denoise"] > 0
    assert calls["prefill"] == int(prefix_length >= 8)


def test_answers():
    assert answer("work 99 then #### 1,200") == answer("#### 1200", gold=True)
    assert answer("\\boxed{42}") == "42"
    assert answer("no answer") is None


def test_global_microbatch_gradients_match():
    torch.manual_seed(3)
    a = Model(cfg()); add_lora(a, 4)
    b = copy.deepcopy(a)
    ids = torch.randint(0, 39, (2, 16))
    prefix = torch.tensor([3, 7])
    response = torch.arange(16)[None] >= prefix[:, None]
    r, probs = torch.rand(2, 16), torch.full((2, 2), .6)
    codec = Codec(spec())
    loss(a, ids, response, prefix, codec, r, probs, 40, 8).backward()
    for i in range(2):
        v = loss(b, ids[i:i+1], response[i:i+1], prefix[i:i+1], codec, r[i:i+1], probs[i:i+1], 40, 8)
        (v * response[i:i+1].sum() / response.sum()).backward()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.requires_grad:
            torch.testing.assert_close(pa.grad, pb.grad, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_lora_backward_and_generation():
    # Random tiny weights only; never downloads or opens a real checkpoint.
    torch.manual_seed(14)
    m = Model(cfg()).to(device="cuda", dtype=torch.bfloat16)
    add_lora(m, 4)
    m.gradient_checkpointing = True
    codec = Codec(spec()).cuda()
    ids = torch.randint(0, 39, (1, 16), device="cuda")
    prefix = torch.tensor([4], device="cuda")
    response = torch.arange(16, device="cuda")[None] >= prefix[:, None]
    value = loss(m, ids, response, prefix, codec, torch.rand(1, 16, device="cuda"),
                 torch.full((1, 2), .5, device="cuda"), 40, 8)
    value.backward()
    assert torch.isfinite(value)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    m.eval()
    generated, calls = generate(m, [2] * 9, codec, 40, 39, 2, 9)
    assert 40 not in generated and calls["denoise"] > 0
