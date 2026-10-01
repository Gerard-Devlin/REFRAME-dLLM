import pytest
import torch

from focus_dllm.tuning.kv_delta_diagnostic import aggregate_geometry, delta_geometry


def positions(values):
    return torch.tensor(values, dtype=torch.long)


def test_shared_shift_and_teacher_only_scalar():
    old = torch.zeros(1, 1, 4, 2)
    fresh = torch.tensor([[[[1., 2.], [1., 2.], [3., 6.], [3., 6.]]]])
    row = delta_geometry(old, fresh, positions([0, 1]), positions([2, 3]))
    assert row['delta_energy'] == 90.
    assert row['source_mean_error'] == 40.
    assert row['offline_fitted_error'] == pytest.approx(0.)
    identical = delta_geometry(old, fresh, positions([0]), positions([1]))
    assert identical['source_mean_error'] == 0.
    assert torch.count_nonzero(old) == 0


def test_opposing_changes_are_not_reported_as_improvement():
    old = torch.zeros(1, 1, 2, 1)
    fresh = torch.tensor([[[[1.], [-1.]]]])
    row = delta_geometry(old, fresh, positions([0]), positions([1]))
    assert row['source_mean_error'] / row['delta_energy'] == 4.


def test_aggregation_uses_energy_not_mean_of_ratios():
    old = torch.zeros(1, 1, 2, 1)
    a = delta_geometry(old, torch.tensor([[[[1.], [1.]]]]), positions([0]), positions([1]))
    b = delta_geometry(old, torch.tensor([[[[-10.], [10.]]]]), positions([0]), positions([1]))
    result = aggregate_geometry([a, b])
    assert result['source_mean_relative_error'] == pytest.approx(400 / 101)
    zero = delta_geometry(old, old, positions([0]), positions([1]))
    assert aggregate_geometry([zero])['source_mean_relative_error'] is None
    assert delta_geometry(old, old, positions([]), positions([1])) is None


def test_rejects_leakage_duplicate_and_misaligned_positions():
    old = torch.zeros(1, 1, 4, 2)
    with pytest.raises(ValueError, match='own prediction'):
        delta_geometry(old, old, positions([1, 2]), positions([2, 3]))
    with pytest.raises(ValueError, match='Repeated'):
        delta_geometry(old, old, positions([1, 1]), positions([2, 3]))
    with pytest.raises(ValueError, match='outside'):
        delta_geometry(old, old, positions([0]), positions([4]))
    with pytest.raises(ValueError, match='matching'):
        delta_geometry(old, old[:, :, :3], positions([0]), positions([1]))
