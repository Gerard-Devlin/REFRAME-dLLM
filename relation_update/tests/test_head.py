import copy

import pytest
import torch

from relation_update.head import HeadConfig, ResidualHead, distillation_loss


def batch(seed=7, b=2, length=4, hidden=8, features=6, k=3):
    g = torch.Generator().manual_seed(seed)
    return {
        "hidden": torch.randn(b, length, hidden, generator=g),
        "candidate_features": torch.randn(b, length, k, features, generator=g),
        "candidates": torch.arange(k).expand(b, length, k),
        "base_log_probs": torch.randn(b, length, k + 1, generator=g).log_softmax(-1),
        "commit_features": torch.randn(b, length, features, generator=g),
        "committed": torch.tensor([[True, False, False, False]]).expand(b, length).clone(),
        "eligible": torch.tensor([[False, True, True, True]]).expand(b, length).clone(),
    }


@pytest.mark.parametrize("mode", ["relation", "blind", "calibration"])
def test_zero_initialization_keeps_original_distribution_and_tail(mode):
    data = batch()
    head = ResidualHead(HeadConfig(8, 6, width=12, mode=mode))
    probabilities = head(data).exp()
    torch.testing.assert_close(probabilities, data["base_log_probs"].exp())
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 4))
    assert torch.all(probabilities[..., :-1].sum(-1) < 1)
    assert sum(p.numel() for p in head.parameters()) < 10000


def activate_outputs(head):
    with torch.no_grad():
        for name, parameter in head.named_parameters():
            if "output.1" in name:
                parameter.normal_(std=.3)


def test_blind_is_exactly_matched_and_invariant_to_commit_tokens_and_locations():
    relation = ResidualHead(HeadConfig(8, 6, 12, "relation"))
    blind = ResidualHead(HeadConfig(8, 6, 12, "blind"))
    assert list(relation.state_dict()) == list(blind.state_dict())
    assert sum(p.numel() for p in relation.parameters()) == sum(p.numel() for p in blind.parameters())
    activate_outputs(blind)
    data = batch()
    changed = copy.deepcopy(data)
    changed["commit_features"] = 10 * torch.randn_like(data["commit_features"])
    changed["committed"] = ~data["committed"]
    torch.testing.assert_close(blind(data), blind(changed), rtol=0, atol=0)
    blind(data).square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in blind.parameters())


def test_relation_can_use_commit_identity_and_position():
    torch.manual_seed(123)
    head = ResidualHead(HeadConfig(8, 6, 12))
    activate_outputs(head)
    data = batch()
    changed_token = copy.deepcopy(data)
    changed_token["commit_features"][:, 0] *= -1
    assert not torch.allclose(head(data), head(changed_token))
    changed_position = copy.deepcopy(data)
    changed_position["committed"] = data["committed"].roll(1, dims=1)
    changed_position["commit_features"] = data["commit_features"].roll(1, dims=1)
    assert not torch.allclose(head(data), head(changed_position))


def test_uncommitted_token_features_do_not_leak_into_message():
    head = ResidualHead(HeadConfig(8, 6, 12))
    activate_outputs(head)
    data = batch()
    changed = copy.deepcopy(data)
    changed["commit_features"][~changed["committed"]] *= -100
    torch.testing.assert_close(head(data), head(changed), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["relation", "blind", "calibration"])
def test_no_teacher_or_backbone_gradient_leakage(mode):
    head = ResidualHead(HeadConfig(8, 6, 12, mode))
    data = batch()
    for key in ("hidden", "candidate_features", "base_log_probs", "commit_features"):
        data[key].requires_grad_()
    data["teacher_probs"] = torch.randn_like(data["base_log_probs"]).softmax(-1).requires_grad_()
    first = head(data)
    altered = dict(data, teacher_probs=torch.ones_like(data["teacher_probs"]))
    torch.testing.assert_close(first, head(altered), rtol=0, atol=0)
    loss, count = distillation_loss(first, data["teacher_probs"], data["eligible"])
    (loss / count).backward()
    for key in ("hidden", "candidate_features", "base_log_probs", "commit_features", "teacher_probs"):
        assert data[key].grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_tail_probability_is_learned_not_discarded():
    head = ResidualHead(HeadConfig(8, 6, 12))
    data = batch()
    teacher = torch.zeros_like(data["base_log_probs"])
    teacher[..., -1] = 1
    before = head(data).exp()[..., -1].mean().item()
    optimizer = torch.optim.Adam(head.parameters(), lr=.03)
    for _ in range(25):
        optimizer.zero_grad()
        loss, count = distillation_loss(head(data), teacher, data["eligible"])
        (loss / count).backward()
        optimizer.step()
    assert head(data).exp()[data["eligible"]][:, -1].mean().item() > .95
    assert before < .95


def test_relation_learns_changed_target_from_newly_committed_identity():
    # Identical current distributions and hidden states; only the intervention
    # tells the two examples apart. Neither sharpening nor a blind head can fit.
    torch.manual_seed(42)
    data = batch(b=2)
    for key in ("hidden", "candidate_features", "base_log_probs"):
        data[key][1] = data[key][0]
    data["base_log_probs"] = torch.tensor([.70, .10, .10, .10]).log().expand(2, 4, 4).clone()
    data["commit_features"][1, 0] = -data["commit_features"][0, 0]
    teacher = torch.zeros_like(data["base_log_probs"])
    teacher[0, :, 1] = 1
    teacher[1, :, 2] = 1
    head = ResidualHead(HeadConfig(8, 6, 16))
    optimizer = torch.optim.Adam(head.parameters(), lr=.03)
    before, count = distillation_loss(head(data), teacher, data["eligible"])
    for _ in range(90):
        optimizer.zero_grad()
        loss, count = distillation_loss(head(data), teacher, data["eligible"])
        (loss / count).backward()
        optimizer.step()
    after, _ = distillation_loss(head(data), teacher, data["eligible"])
    assert after.item() < before.item() * .02
    assert torch.equal(head(data).argmax(-1)[data["eligible"]], teacher.argmax(-1)[data["eligible"]])


def test_kl_is_token_sum_including_tail_and_handles_empty_targets():
    predicted = torch.tensor([[[.3, .7], [.8, .2]]]).log().requires_grad_()
    teacher = torch.tensor([[[0., 1.], [.5, .5]]], requires_grad=True)
    total, count = distillation_loss(predicted, teacher, torch.tensor([[True, False]]))
    torch.testing.assert_close(total, -predicted[0, 0, 1])
    assert count.item() == 1
    empty, count = distillation_loss(predicted, teacher, torch.tensor([[False, False]]))
    empty.backward()
    assert empty.item() == 0 and count.item() == 0
    assert torch.isfinite(predicted.grad).all()
    assert teacher.grad is None


def test_no_commits_and_single_position_are_finite():
    data = batch()
    data = {k: v[:, :1].clone() for k, v in data.items()}
    data["committed"].zero_()
    head = ResidualHead(HeadConfig(8, 6, 12))
    activate_outputs(head)
    assert torch.isfinite(head(data)).all()


def test_malformed_shapes_are_rejected():
    head = ResidualHead(HeadConfig(8, 6, 12))
    data = batch()
    data["base_log_probs"] = data["base_log_probs"][..., :-1]
    with pytest.raises(ValueError, match="OTHER"):
        head(data)
    with pytest.raises(ValueError, match="Unknown head mode"):
        HeadConfig(8, 6, mode="teacher")
