from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v1" / "llada"))
sys.path.insert(0, str(ROOT / "reframe"))
torch.set_num_threads(1)


@pytest.fixture
def tiny_model():
    from model.configuration_llada import LLaDAConfig
    from model.modeling_llada import LLaDAModelLM
    torch.manual_seed(7)
    cfg = LLaDAConfig(d_model=32, n_heads=4, n_layers=2, mlp_hidden_size=64,
                     vocab_size=128, embedding_size=128, block_type="llama",
                     activation_type="silu", rope=True, weight_tying=False,
                     max_sequence_length=128, attention_dropout=0.0,
                     residual_dropout=0.0, embedding_dropout=0.0,
                     eos_token_id=126, pad_token_id=126)
    return LLaDAModelLM(cfg, init_params=True).eval()
