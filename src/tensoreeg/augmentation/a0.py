"""A0: transductive Riemannian alignment toward an unlabeled target manifold."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.linalg import eigh

from .. import spd
from ._common import CovStack, coerce_cov_stack, coerce_labels

__all__ = ["augment_cov_alignment_riemannian"]


def _riemann_mean(
    cov_stack: np.ndarray, max_iter: int = 30, tol: float = 1e-6
) -> np.ndarray:
    """Affine-invariant Riemannian (Frechet) mean of an SPD stack.

    Iterative Karcher-flow on the SPD manifold; the initial guess is the
    arithmetic mean and we iterate the tangent-space mean update until
    the tangent-space residual norm falls below ``tol`` or ``max_iter``
    is reached.
    """
    if cov_stack.ndim != 3:
        raise ValueError("cov_stack must be a (n, p, p) array.")
    n, p, _ = cov_stack.shape
    M = np.mean(cov_stack, axis=0)
    M = 0.5 * (M + M.T)
    for _ in range(max_iter):
        vals, vecs = eigh(0.5 * (M + M.T))
        vals = np.maximum(vals, 1e-12)
        inv_sqrt = (vecs * (vals ** -0.5)) @ vecs.T
        sqrt_M = (vecs * (vals ** 0.5)) @ vecs.T
        tangent = np.zeros((p, p))
        for k in range(n):
            W = inv_sqrt @ cov_stack[k] @ inv_sqrt
            tangent += spd.spd_logm(W)
        tangent /= n
        if np.linalg.norm(tangent) < tol:
            break
        update = sqrt_M @ spd.spd_expm(tangent) @ sqrt_M
        M = 0.5 * (update + update.T)
    return M


def augment_cov_alignment_riemannian(
    source_cov_list: CovStack,
    target_cov_list: CovStack,
    labels: Optional[Sequence] = None,
) -> dict:
    r"""Riemannian alignment of source covariances toward a target manifold (A0).

    Whitens the source-session anchor covariances toward the affine-
    invariant Riemannian mean of an **unlabeled** target-session
    covariance stack:

    .. math::

        W = M_T^{1/2} M_S^{-1/2},
        \\qquad
        \\tilde C_i = W C_i W^{\\top}.

    A0 consults unlabeled target covariances and is therefore
    *transductive*. In Shen & Degras (2026) it is reported as a
    transfer-learning upper bound rather than a non-inferiority
    candidate.

    Parameters
    ----------
    source_cov_list : list of (p, p) ndarray, or (n_src, p, p) ndarray
        SPD source-session anchor covariances.
    target_cov_list : list of (p, p) ndarray, or (n_tgt, p, p) ndarray
        SPD target-session covariances (unlabeled). Only their geometry
        is used.
    labels : sequence, optional
        Anchor labels passed through to the result for downstream code.
    """
    source = coerce_cov_stack(source_cov_list, "source_cov_list")
    target = coerce_cov_stack(target_cov_list, "target_cov_list")
    n_src, p, _ = source.shape
    if target.shape[1] != p:
        raise ValueError(
            "source and target covariances must have the same dimension."
        )
    labels_arr = coerce_labels(labels, n_src)

    M_S = _riemann_mean(source)
    M_T = _riemann_mean(target)

    vals_s, vecs_s = eigh(M_S)
    vals_s = np.maximum(vals_s, 1e-12)
    src_inv_sqrt = (vecs_s * (vals_s ** -0.5)) @ vecs_s.T

    vals_t, vecs_t = eigh(M_T)
    vals_t = np.maximum(vals_t, 1e-12)
    tgt_sqrt = (vecs_t * (vals_t ** 0.5)) @ vecs_t.T

    W = tgt_sqrt @ src_inv_sqrt

    aligned = np.empty_like(source)
    for i in range(n_src):
        A = W @ source[i] @ W.T
        aligned[i] = 0.5 * (A + A.T)

    return {
        "cov": aligned,
        "labels": labels_arr,
        "params": {"uses_target": "unlabeled_covariance_only"},
    }
