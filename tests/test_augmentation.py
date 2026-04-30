"""Tests for the augmentation family (E0 / G0 / G1 / G2 / A0)."""

from __future__ import annotations

import numpy as np
import pytest

import tensoreeg as te


# -------------------------------------------------------------------------
# Shape and SPD invariants (apply to every augmentation)


@pytest.mark.parametrize("name,fn,kwargs", [
    ("g0", te.augment_g0, dict(n_aug=3, sigma=0.15, seed=1)),
    ("e0", te.augment_e0, dict(n_aug=3, g0_sigma=0.15, seed=1)),
])
def test_unlabeled_aug_shape_and_spd(name, fn, kwargs, synthetic_anchors):
    anchors, labels = synthetic_anchors
    res = fn(anchors, labels=labels, **kwargs)
    n_anchor, p, _ = anchors.shape
    n_aug = kwargs["n_aug"]
    assert res["cov"].shape == (n_anchor * n_aug, p, p), name
    # Symmetric and SPD.
    for C in res["cov"]:
        assert np.allclose(C, C.T, atol=1e-10), name
        assert np.all(np.linalg.eigvalsh(C) > 0), name


@pytest.mark.parametrize("name,fn,kwargs", [
    ("g1", te.augment_g1, dict(n_aug=3, sigma=0.15, seed=1)),
    ("g2", te.augment_g2, dict(n_aug=3, beta_alpha=1.0, seed=1)),
])
def test_labeled_aug_shape_and_spd(name, fn, kwargs, synthetic_anchors):
    anchors, labels = synthetic_anchors
    res = fn(anchors, labels, **kwargs)
    n_anchor, p, _ = anchors.shape
    n_aug = kwargs["n_aug"]
    assert res["cov"].shape == (n_anchor * n_aug, p, p), name
    for C in res["cov"]:
        assert np.allclose(C, C.T, atol=1e-10), name
        assert np.all(np.linalg.eigvalsh(C) > 0), name


# -------------------------------------------------------------------------
# Reproducibility within Python


def test_g0_reproducible_with_seed(synthetic_anchors):
    anchors, labels = synthetic_anchors
    a1 = te.augment_g0(anchors, n_aug=3, sigma=0.1, labels=labels, seed=42)
    a2 = te.augment_g0(anchors, n_aug=3, sigma=0.1, labels=labels, seed=42)
    assert np.allclose(a1["cov"], a2["cov"], atol=0)
    assert np.array_equal(a1["anchor"], a2["anchor"])


def test_g2_reproducible_with_seed(synthetic_anchors):
    anchors, labels = synthetic_anchors
    a1 = te.augment_g2(anchors, labels, n_aug=4, beta_alpha=1.0, seed=99)
    a2 = te.augment_g2(anchors, labels, n_aug=4, beta_alpha=1.0, seed=99)
    assert np.allclose(a1["cov"], a2["cov"], atol=0)
    assert np.array_equal(a1["trace"]["partner"], a2["trace"]["partner"])
    assert np.allclose(a1["trace"]["alpha"], a2["trace"]["alpha"], atol=0)


def test_different_seeds_produce_different_output(synthetic_anchors):
    anchors, labels = synthetic_anchors
    a = te.augment_g0(anchors, n_aug=3, sigma=0.1, labels=labels, seed=1)
    b = te.augment_g0(anchors, n_aug=3, sigma=0.1, labels=labels, seed=2)
    assert not np.allclose(a["cov"], b["cov"])


# -------------------------------------------------------------------------
# G2 partner constraints (same-class only, never self when alternatives exist)


def test_g2_partners_are_same_class(synthetic_anchors):
    anchors, labels = synthetic_anchors
    res = te.augment_g2(anchors, labels, n_aug=5, seed=0)
    for k, partner in enumerate(res["trace"]["partner"]):
        i = res["anchor"][k]
        assert labels[partner] == labels[i], (
            f"G2 partner {partner} (label {labels[partner]}) for anchor {i} "
            f"(label {labels[i]})"
        )
    # Check that partner != anchor when alternatives exist (n_per_class >= 2).
    same_self = sum(int(p == i) for p, i in zip(res["trace"]["partner"], res["anchor"]))
    assert same_self == 0


def test_g2_alpha_in_unit_interval(synthetic_anchors):
    anchors, labels = synthetic_anchors
    res = te.augment_g2(anchors, labels, n_aug=5, seed=0)
    assert np.all((res["trace"]["alpha"] >= 0) & (res["trace"]["alpha"] <= 1))


# -------------------------------------------------------------------------
# E0 amplitude-match diagnostic


def test_e0_amplitude_match_rho_close_to_one(synthetic_anchors):
    anchors, labels = synthetic_anchors
    res = te.augment_e0(anchors, n_aug=5, g0_sigma=0.1, labels=labels, seed=7)
    rho = res["diagnostic"]["rho"]
    assert 0.5 < rho < 2.0, f"rho out of range: {rho}"


# -------------------------------------------------------------------------
# A0 transductive alignment


def test_a0_aligns_source_mean_toward_target_mean(synthetic_anchors):
    anchors, labels = synthetic_anchors
    rng = np.random.default_rng(0)
    # Build a synthetic 'target' stack that is the source rescaled by 2x.
    target = anchors * 2.0
    res = te.augment_a0(anchors, target, labels=labels)
    # Aligned source mean should be closer to target mean than original.
    src_mean = anchors.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    aligned_mean = res["cov"].mean(axis=0)
    d_before = np.linalg.norm(src_mean - tgt_mean)
    d_after = np.linalg.norm(aligned_mean - tgt_mean)
    assert d_after < d_before, "A0 alignment did not reduce mean distance"
    # SPD.
    for C in res["cov"]:
        assert np.all(np.linalg.eigvalsh(0.5 * (C + C.T)) > 0)


# -------------------------------------------------------------------------
# Input validation


def test_g0_rejects_bad_sigma(synthetic_anchors):
    anchors, labels = synthetic_anchors
    with pytest.raises(ValueError, match="sigma"):
        te.augment_g0(anchors, n_aug=2, sigma=-0.1, labels=labels, seed=1)


def test_g1_requires_at_least_two_anchors():
    one = np.eye(4)[None, :, :]  # shape (1, 4, 4)
    with pytest.raises(ValueError, match="at least two"):
        te.augment_g1(one, labels=[0], n_aug=2, sigma=0.1)


def test_label_length_mismatch_raises(synthetic_anchors):
    anchors, _ = synthetic_anchors
    with pytest.raises(ValueError, match="length"):
        te.augment_g2(anchors, labels=[0, 1, 0], n_aug=2)
