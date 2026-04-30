"""Unit tests for tensoreeg.spd: matrix logarithm/exponential, projections, distances, vech."""

from __future__ import annotations

import numpy as np
import pytest

from tensoreeg import spd


def random_spd(p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    return A @ A.T + p * np.eye(p)


# -------------------------------------------------------------------------
# spd_logm / spd_expm round-trip


def test_logm_expm_roundtrip_recovers_input():
    for p in (3, 6, 12):
        for s in (1, 7, 42):
            C = random_spd(p, s)
            recovered = spd.spd_expm(spd.spd_logm(C))
            assert np.allclose(recovered, C, atol=1e-10), \
                f"round trip failed for p={p}, seed={s}"


def test_logm_rejects_non_spd():
    A = np.eye(4)
    A[0, 0] = -1.0  # negative eigenvalue
    with pytest.raises(ValueError, match="positive definite"):
        spd.spd_logm(A)


def test_expm_returns_symmetric_for_symmetric_input():
    for p in (3, 8):
        for s in (2, 9):
            rng = np.random.default_rng(s)
            G = rng.standard_normal((p, p))
            Z = 0.5 * (G + G.T)
            C = spd.spd_expm(Z)
            assert np.allclose(C, C.T, atol=1e-12)
            # SPD?
            assert np.all(np.linalg.eigvalsh(C) > 0)


# -------------------------------------------------------------------------
# spd_project


def test_spd_project_clips_eigenvalues():
    C = random_spd(5, 3)
    M = C - 5 * np.eye(5)  # likely indefinite
    proj = spd.spd_project(M, floor=1e-3)
    assert np.all(np.linalg.eigvalsh(proj) >= 1e-3 - 1e-12)
    # Symmetric.
    assert np.allclose(proj, proj.T, atol=1e-12)


# -------------------------------------------------------------------------
# Distances


def test_log_euclidean_distance_zero_for_equal():
    C = random_spd(7, 11)
    assert spd.log_euclidean_distance(C, C) < 1e-12


def test_log_euclidean_distance_positive_for_different():
    C1 = random_spd(7, 11)
    C2 = random_spd(7, 13)
    assert spd.log_euclidean_distance(C1, C2) > 1e-3


def test_affine_invariant_distance_zero_for_equal():
    C = random_spd(7, 11)
    assert spd.affine_invariant_distance(C, C) < 1e-10


def test_affine_invariant_invariant_under_congruence():
    """d_AI(W C1 W^T, W C2 W^T) == d_AI(C1, C2) for invertible W."""
    C1 = random_spd(5, 1)
    C2 = random_spd(5, 2)
    rng = np.random.default_rng(99)
    W = rng.standard_normal((5, 5))
    while abs(np.linalg.det(W)) < 1e-3:
        W = rng.standard_normal((5, 5))
    d1 = spd.affine_invariant_distance(C1, C2)
    d2 = spd.affine_invariant_distance(W @ C1 @ W.T, W @ C2 @ W.T)
    assert abs(d1 - d2) < 1e-6


def test_distance_shape_mismatch_raises():
    C1 = random_spd(4, 1)
    C2 = random_spd(5, 2)
    with pytest.raises(ValueError, match="same shape"):
        spd.log_euclidean_distance(C1, C2)


# -------------------------------------------------------------------------
# vech round trip


def test_vech_log_unvech_roundtrip():
    for p in (3, 6, 8):
        C = random_spd(p, p * 7 + 3)
        z = spd.vech_log(C)
        # Unvech the vech of log(C) should reconstruct log(C).
        Z_back = spd.unvech(z, p)
        Z_orig = spd.spd_logm(C)
        assert np.allclose(Z_back, Z_orig, atol=1e-10)


def test_vech_log_offdiagonal_sqrt2_weight():
    """Frobenius norm of log(C) equals Euclidean norm of vech_log(C)."""
    C = random_spd(6, 17)
    z = spd.vech_log(C)
    Z = spd.spd_logm(C)
    assert abs(float(np.linalg.norm(z)) - float(np.linalg.norm(Z))) < 1e-10
