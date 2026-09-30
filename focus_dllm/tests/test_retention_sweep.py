import pytest

from focus_dllm.tuning.retention_sweep import CONFIGS, sample_slice


def test_screening_and_future_validation_use_disjoint_prompt_ids():
    samples = [{'id': i} for i in range(164)]
    screen = sample_slice(samples, 0, 32)
    existing_screen = sample_slice(samples, 0, 64)
    validate = sample_slice(samples, 64, 64)
    assert screen == existing_screen[:32]
    assert not {s['id'] for s in existing_screen} & {s['id'] for s in validate}
    assert samples == [{'id': i} for i in range(164)]
    assert validate == sample_slice(samples, 64, 64)
    with pytest.raises(ValueError):
        sample_slice(samples, 128, 64)


def test_retention_ablation_keeps_native_threshold_and_fixed_parameters():
    assert all(c['threshold'] == .90 for c in CONFIGS.values())
    assert CONFIGS['support16_r1_l4']['refresh'] == 1
    assert CONFIGS['support32_r1_l4']['support'] == 32
    assert CONFIGS['support16_r1_l8']['layer'] == 8
    assert all(c['keep'] == 0. for c in CONFIGS.values())
