import json
import pytest
import torch
from relation_block.diagnose import repetition, summarize_pairs, compare_weight_files, reconstruction
from relation_block.model import Model
from relation_block.codec import Codec


def test_paired_length_report():
    a = [dict(id=1, correct=False, length_capped=True, prediction="ha " * 12),
         dict(id=2, correct=True, length_capped=False, prediction="answer 2")]
    b = [dict(id=1, correct=True, length_capped=False, prediction="answer 1"),
         dict(id=2, correct=False, length_capped=True, prediction="no answer")]
    result = summarize_pairs(a, b)
    assert result["wrong_to_correct"] == result["correct_to_wrong"] == 1
    assert result["capped_to_correct"] == 1
    assert result["short_repeated_4gram"] > result["long_repeated_4gram"]
    with pytest.raises(ValueError, match="same questions"):
        summarize_pairs(a, b[::-1])
    assert repetition("short") == 0


def test_all_tensor_roundtrip_detects_changes(tmp_path):
    from safetensors.torch import save_file
    root = tmp_path / "original"; root.mkdir()
    out = tmp_path / "checkpoint"; out.mkdir()
    x = torch.randn(8, 4).bfloat16()
    save_file({"model.embed_tokens.weight": x}, str(root / "model.safetensors"))
    (root / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    original = {"model.embed_tokens.weight": x, "lm_head.weight": x}
    torch.save(original, out / "model_bf16.pt")
    fp32 = {k: v.float() for k, v in original.items()}
    torch.save(fp32, out / "model_fp32.pt")
    assert compare_weight_files(out, root)["pass_"]
    fp32["model.embed_tokens.weight"][0, 0] += .01
    torch.save(fp32, out / "model_fp32.pt")
    assert not compare_weight_files(out, root)["pass_"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_reconstruction_counts_every_target_once():
    cfg = dict(vocab_size=41, hidden_size=16, intermediate_size=24,
               num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=1,
               rms_norm_eps=1e-6, rope_theta=1000000., tie_word_embeddings=True)
    m = Model(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    codec = Codec(dict(block_size=4, vocab_size=41, protected=[39, 40], swaps=[])).cuda()
    rows = [dict(ids=[1, 2, 3, 4, 5, 6, 39, 7], prefix=2)]
    result = reconstruction(m, codec, rows, dict(length=8, block_size=4, pad_id=0, mask_id=40, eos_id=39))
    for entry in result.values():
        counts = entry["categories"]
        assert counts["all"]["total"] == 6
        assert counts["eos"]["total"] == 1
        assert counts["boundary"]["total"] == 1
        assert counts["ordinary"]["total"] == 4
    assert not m._forward_pre_hooks and not m.lm_head._forward_hooks
