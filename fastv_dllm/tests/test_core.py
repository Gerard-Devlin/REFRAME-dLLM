import torch

from fastv_dllm.pruning import FastVConfig, add_stability, choose_keep, predictor_positions
from fastv_dllm.report import summarize


def test_predictor_positions_respect_v2_shift():
    assert predictor_positions(0, 8) == [0, 0, 1, 2, 3, 4, 5, 6]
    assert predictor_positions(8, 8) == list(range(7, 15))
    assert predictor_positions(24, 8) == list(range(23, 31))


def test_required_predictors_are_never_pruned():
    relevance = torch.arange(10, dtype=torch.float32)
    keep = choose_keep(relevance, required=[1, 1, 7], support_keep=2)
    assert keep == [1, 7, 8, 9]


def test_all_support_is_identity_order():
    relevance = torch.randn(32)
    predictors = predictor_positions(8, 8)
    assert choose_keep(relevance, predictors, 32) == list(range(32))


def test_config_rejects_last_layer_pruning():
    try:
        FastVConfig(prune_after_layer=28).validate(28)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid no-compute-saving prune point")


def test_stability_uses_support_rankings():
    records = [{"layers": [
        {"top_support": {"2": [1, 2]}},
        {"top_support": {"2": [2, 3]}},
    ]}]
    add_stability(records, (2,))
    assert records[0]["layers"][0]["jaccard_prev_top2"] is None
    assert records[0]["layers"][1]["jaccard_prev_top2"] == 1 / 3


def _row(seconds, correct, flash=0, deep=None):
    return dict(seconds=seconds, correct=correct, logical_forwards=10, ordinary_denoise=8,
                tokens=32, length_capped=False, backend=dict(flash_calls=flash, fallback_calls=0),
                deep_tokens=[] if deep is None else deep)


def test_attribution_separates_flash_and_method_speedups():
    records = [{
        "sdpa_native": _row(4, True),
        "flash_native": _row(2, True, flash=10),
        "sdpa_fastv": _row(2, True, deep=[16]),
        "flash_fastv": _row(1, True, flash=10, deep=[16]),
        "sdpa_cache": _row(3, True),
        "flash_cache": _row(1.5, True, flash=10),
    }]
    result = summarize(records, tuple(records[0]))
    assert result["attribution"]["flash_engineering_speedup"] == 2
    assert result["attribution"]["fastv_method_speedup_same_flash"] == 2
    assert result["attribution"]["official_cache_speedup_same_flash"] == 4 / 3
