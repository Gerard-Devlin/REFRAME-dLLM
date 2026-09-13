"""Exact binary diagnostics. No trained denoiser, text corpus, or GPU speed claim."""

import argparse
import json
from pathlib import Path

import numpy as np

from .codec import ChainDifference, Codec, PointwiseRename, SparseCoupling, fit_coupling


def binary_states(length):
    if not 2 <= length <= 16:
        raise ValueError("Exact enumeration is limited to lengths 2..16")
    return ((np.arange(2 ** length)[:, None] >> np.arange(length)) & 1).astype(np.int64)


def distribution(states, kind):
    if kind == "copy_chain":
        flips = np.sum(states[:, 1:] != states[:, :-1], axis=1)
        p = 0.5 * 0.03 ** flips * 0.97 ** (states.shape[1] - 1 - flips)
    elif kind == "independent_biased":
        ones = states.sum(axis=1)
        p = 0.2 ** ones * 0.8 ** (states.shape[1] - ones)
    else:
        raise ValueError(kind)
    return p / p.sum()


def marginal(states, p, positions):
    positions = list(positions)
    codes = states[:, positions] @ (2 ** np.arange(len(positions)))
    mass = np.bincount(codes, weights=p, minlength=2 ** len(positions))
    return mass, codes


def entropy(states, p, positions):
    mass, _ = marginal(states, p, positions)
    positive = mass > 0
    return float(-np.sum(mass[positive] * np.log(mass[positive])))


def exact_path(states, p, groups):
    """Optimal factorized conditionals for a FIXED reveal partition.

    These are population oracle conditionals, not neural-model predictions.
    log_q is indexed by states, permitting exact comparison after a bijection.
    The calculation requires a full-support binary population and an exhaustive
    unique state table. All supplied synthetic distributions have full support.
    """
    positions = [int(i) for group in groups for i in group]
    if sorted(positions) != list(range(states.shape[1])):
        raise ValueError("Each position must be revealed exactly once")
    if len(states) != 2 ** states.shape[1] or len(np.unique(states, axis=0)) != len(states):
        raise ValueError("Expected a complete unique binary state table")
    if np.any((states != 0) & (states != 1)) or np.any(p <= 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Expected binary states and a normalized full-support population")
    seen, log_q, conditional_tc = [], np.zeros(len(p)), 0.0
    for group in groups:
        group = [int(i) for i in group]
        observed_mass, observed_codes = marginal(states, p, seen)
        h_seen = entropy(states, p, seen)
        sum_h = 0.0
        for position in group:
            joint_mass, joint_codes = marginal(states, p, seen + [position])
            log_q += np.log(joint_mass[joint_codes]) - np.log(observed_mass[observed_codes])
            sum_h += entropy(states, p, seen + [position]) - h_seen
        conditional_tc += sum_h - (entropy(states, p, seen + group) - h_seen)
        seen += group
    q = np.exp(log_q)
    kl = float(np.sum(p * (np.log(p) - log_q)))
    return q, {
        "kl_nats": kl,
        "tv": float(0.5 * np.sum(np.abs(p - q))),
        "oracle_path_nll_nats": float(-p @ log_q),
        "conditional_tc_nats": float(conditional_tc),
        "decomposition_error": abs(kl - conditional_tc),
        "q_mass": float(q.sum()),
    }


def corruption_probe(codec, states, p):
    z = codec.encode(states)
    if not np.array_equal(codec.decode(z), states):
        raise AssertionError("Codec round trip failed")
    average, maximum = 0.0, 0
    for position in range(states.shape[1]):
        changed = z.copy()
        changed[:, position] ^= 1
        errors = np.sum(codec.decode(changed) != states, axis=1)
        average += float(p @ errors) / states.shape[1]
        maximum = max(maximum, int(errors.max()))
    return {"mean_changed_original_tokens_per_code_flip": average,
            "max_changed_original_tokens_per_code_flip": maximum}


def build_codecs(train, seed):
    length = train.shape[1]
    first_pairs = tuple((i, i + 1) for i in range(0, length - 1, 2))
    second_pairs = tuple((i, i + 1) for i in range(1, length - 1, 2))
    first = fit_coupling(train, 2, first_pairs)
    second = fit_coupling(first.encode(train), 2, second_pairs)
    rng = np.random.default_rng(seed)
    # Require actual cross-token dependence. In a binary vocabulary this control
    # is necessarily XOR (or its complement), a limitation recorded in the report.
    first_flag = int(rng.integers(2))
    random_tables = {a: ({0: 1, 1: 0} if (first_flag ^ a) else {}) for a in range(2)}
    return {
        "identity": Codec(2),
        "pointwise_rename": PointwiseRename(2),
        "random_one_layer": Codec(2, (SparseCoupling(2, first_pairs, random_tables),)),
        "statistical_one_layer": Codec(2, (first,)),
        "statistical_two_layers": Codec(2, (first, second)),
        "chain_difference_reference": ChainDifference(2),
    }


def run(length=12, train_samples=8192, seed=1234):
    if train_samples < 1:
        raise ValueError("train_samples must be positive")
    states = binary_states(length)
    rng = np.random.default_rng(seed)
    report = {
        "scope": "Exact synthetic population oracle; no neural fitting error or GPU timing measured",
        "length": length, "train_samples": train_samples, "seed": seed,
        "fit_protocol": "Frequency codecs fitted on a separate sampled synthetic training set",
        "schedule": "Fixed contiguous groups; one-step is also a flow + factorized-base baseline",
        "random_control_limit": "Nontrivial binary coupling is XOR up to relabeling; its tie with a fitted codec is not evidence for learned encoding",
        "populations": {},
    }
    for kind in ("copy_chain", "independent_biased"):
        p = distribution(states, kind)
        train = states[rng.choice(len(states), size=train_samples, p=p)]
        cases = {}
        for name, codec in build_codecs(train, seed).items():
            encoded = codec.encode(states)
            paths = {}
            for steps in sorted({1, 2, 4, length}):
                if steps > length:
                    continue
                groups = [g.tolist() for g in np.array_split(np.arange(length), steps)]
                q, metrics = exact_path(encoded, p, groups)
                metrics["generated_all_equal_probability"] = float(q[np.all(states == states[:, :1], axis=1)].sum())
                paths[str(steps)] = metrics
            cases[name] = {"paths": paths, "corruption": corruption_probe(codec, states, p)}
        report["populations"][kind] = {
            "entropy_nats": entropy(states, p, range(length)),
            "true_all_equal_probability": float(p[np.all(states == states[:, :1], axis=1)].sum()),
            "codecs": cases,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--train-samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.length, args.train_samples, args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
    for kind, population in report["populations"].items():
        print(kind)
        for name, case in population["codecs"].items():
            p = case["paths"]
            scores = " ".join(f"KL@{steps}={metrics['kl_nats']:.6f}" for steps, metrics in p.items())
            print(f"  {name:28} {scores} "
                  f"flip_max={case['corruption']['max_changed_original_tokens_per_code_flip']}")


if __name__ == "__main__":
    main()
