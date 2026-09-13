import itertools

import numpy as np
import pytest

from relation_diffusion.codec import (
    ChainDifference, Codec, PointwiseRename, SparseCoupling,
    complete_sparse_permutation, fit_coupling,
)
from relation_diffusion.probe import binary_states, build_codecs, corruption_probe, distribution, exact_path, main


def test_sparse_completion_exhaustive():
    for sources in itertools.permutations(range(4), 3):
        for codes in itertools.permutations(range(4), 3):
            table = complete_sparse_permutation(sources, codes)
            assert [table.get(s, s) for s in sources] == list(codes)
            assert sorted(table.get(i, i) for i in range(4)) == list(range(4))


def test_full_nonbinary_bijection_and_prefix():
    response = np.array(list(itertools.product(range(5), repeat=4)))
    x = np.concatenate([np.tile([4, 3], (len(response), 1)), response], axis=1)
    first = fit_coupling(x, 5, ((0, 1), (2, 3)), prefix_len=2, top_k=3)
    second = SparseCoupling(5, ((1, 2),), {0: {0: 2, 2: 4, 4: 0}, 2: {1: 3, 3: 1}})
    codec = Codec(5, (first, second))
    z = codec.encode(x, 2)
    assert np.array_equal(z[:, :2], x[:, :2])
    assert np.array_equal(codec.decode(z, 2), x)
    assert len(np.unique(z, axis=0)) == len(x)


def test_protected_ids_and_unseen_anchors():
    x = np.array([[1, 3], [1, 3], [1, 2], [1, 2], [0, 3], [0, 3]])
    codec = fit_coupling(x, 5, ((0, 1),), protected_ids=(0, 4), top_k=2)
    all_pairs = np.array(list(itertools.product(range(5), repeat=2)))
    z = codec.encode(all_pairs)
    keep = np.isin(all_pairs[:, 1], [0, 4]) | (all_pairs[:, 0] != 1)
    assert np.array_equal(z[keep], all_pairs[keep])
    assert np.array_equal(codec.decode(z), all_pairs)


def test_invalid_mapping_partitions_and_masks():
    with pytest.raises(ValueError):
        SparseCoupling(3, ((0, 1),), {0: {0: 1}})
    with pytest.raises(ValueError):
        SparseCoupling(3, ((0, 1), (1, 2)), {})
    with pytest.raises(ValueError):
        Codec(3).encode(np.array([[0, 3]]))


def test_pointwise_rename_preserves_every_fixed_path():
    x = binary_states(6)
    p = distribution(x, "copy_chain")
    z = PointwiseRename(2).encode(x)
    for groups in ([[0, 1, 2, 3, 4, 5]], [[0, 2, 4], [1, 3, 5]]):
        q_x, info_x = exact_path(x, p, groups)
        q_z, info_z = exact_path(z, p, groups)
        np.testing.assert_allclose(q_x, q_z, atol=1e-14)
        assert info_x["decomposition_error"] < 1e-12
        assert info_z["decomposition_error"] < 1e-12
        assert abs(info_z["q_mass"] - 1) < 1e-12


def test_chain_success_and_independent_counterexample():
    x = binary_states(8)
    codec = ChainDifference(2)
    z = codec.encode(x)
    groups = [list(range(8))]
    p = distribution(x, "copy_chain")
    _, before = exact_path(x, p, groups)
    _, after = exact_path(z, p, groups)
    assert before["kl_nats"] > 3
    assert abs(after["kl_nats"]) < 1e-12
    p_iid = distribution(x, "independent_biased")
    _, before_iid = exact_path(x, p_iid, groups)
    _, after_iid = exact_path(z, p_iid, groups)
    assert abs(before_iid["kl_nats"]) < 1e-12
    assert after_iid["kl_nats"] > 0.5
    assert corruption_probe(codec, x, p)["max_changed_original_tokens_per_code_flip"] == 8


def test_fixed_path_joint_is_normalized_and_serial_is_exact():
    x = binary_states(6)
    p = distribution(x, "copy_chain")
    q, info = exact_path(x, p, [[i] for i in range(6)])
    np.testing.assert_allclose(p, q, atol=1e-14)
    assert info["decomposition_error"] < 1e-12
    for groups in ([[0, 1], [2, 3], [4, 5]], [[0, 2, 4], [1, 3, 5]]):
        _, info = exact_path(x, p, groups)
        assert abs(info["q_mass"] - 1) < 1e-12
        assert info["decomposition_error"] < 1e-12
    with pytest.raises(ValueError):
        exact_path(x, p, [[0, 1], [1, 2]])


def test_random_control_really_mixes_positions():
    x = binary_states(4)
    p = distribution(x, "independent_biased")
    for seed in (0, 1, 1234):
        codec = build_codecs(x, seed)["random_one_layer"]
        _, info = exact_path(codec.encode(x), p, [list(range(4))])
        assert info["kl_nats"] > 0.1
        assert corruption_probe(codec, x, p)["max_changed_original_tokens_per_code_flip"] == 2


def test_cli_smallest_supported_length(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["probe", "--length", "2", "--train-samples", "16"])
    main()
    assert "KL@2=" in capsys.readouterr().out
