import json

import torch

from relation_update.data import build_inputs, coarsen, load_prompts
from relation_update.train import epoch_indices, local_records


def test_prompt_split_ignores_answers_and_deduplicates_full_heldout(tmp_path):
    train = [dict(ids=[1, 2, 3], prefix=2), dict(ids=[1, 2, 8], prefix=2),
             dict(ids=[4, 5, 6], prefix=2), dict(ids=[7, 7, 1], prefix=2)]
    heldout = [dict(ids=[1, 2, 9], prefix=2), dict(ids=[8, 8, 9], prefix=2)]
    (tmp_path / "train.json").write_text(json.dumps(train))
    (tmp_path / "heldout.json").write_text(json.dumps(heldout))
    result = load_prompts(tmp_path, 2, 1, 3)
    assert {tuple(r["ids"]) for r in result["train"]} == {(4, 5), (7, 7)}
    assert all(len(r["ids"]) == 2 for split in result.values() for r in split)


def test_other_mass_and_teacher_is_not_in_input_whitelist():
    p = torch.tensor([[[0.4, 0.3, 0.2, 0.1]]])
    candidates = torch.tensor([[[0, 1]]])
    bins = coarsen(p.log(), candidates)
    assert torch.allclose(bins, torch.tensor([[[0.4, 0.3, 0.3]]]))
    row = dict(hidden=torch.ones(1, 4), candidates=candidates[0], base_log_probs=bins[0].log(),
               commit_ids=torch.tensor([0]), committed=torch.tensor([False]), eligible=torch.tensor([True]),
               teacher_probs=bins[0], teacher_top1=torch.tensor([2]))
    batch = build_inputs([row], torch.ones(4, 3), "cpu")
    assert "teacher_probs" not in batch and "teacher_top1" not in batch
    assert not batch["commit_features"].any()


def test_arbitrary_world_size_last_batch_neither_drops_nor_duplicates_supervision():
    dataset = [dict(eligible=torch.ones(2, dtype=torch.bool), index=i) for i in range(19)]
    for world in (1, 4, 5, 6):
        schedule = epoch_indices(len(dataset), 2, world, 123, 0)
        observed = []
        for indices in schedule:
            for rank in range(world):
                records = local_records(dataset, indices, rank, 2)
                observed.extend(r["index"] for r in records if r["eligible"].any())
        assert sorted(observed) == list(range(19))
        assert schedule == epoch_indices(len(dataset), 2, world, 123, 0)
        assert schedule != epoch_indices(len(dataset), 2, world, 123, 1)
    assert all(r["eligible"].all() for r in dataset)
