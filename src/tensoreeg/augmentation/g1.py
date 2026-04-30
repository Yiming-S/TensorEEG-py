"""G1: empirical tangent Gaussian augmentation."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.linalg import cholesky, eigh

from .. import spd
from ._common import CovStack, coerce_cov_stack, coerce_labels, make_rng

__all__ = ["augment_cov_empirical_tangent"]


def augment_cov_empirical_tangent(
    cov_list: CovStack,
    labels: Sequence,
    n_aug: int = 5,
    sigma: float = 0.10,
    shrinkage: float = 0.10,
    seed: Optional[int] = None,
) -> dict:
    r"""Class-aware Gaussian perturbation in the log-Euclidean tangent space (G1).

    For each class :math:`c`, the empirical tangent covariance
    :math:`\\hat\\Sigma_c` is estimated from the half-vectorised
    log-covariances of the same-class anchors with linear shrinkage
    toward the diagonal,

    .. math::

        \\hat\\Sigma_c = (1-\\lambda)\\,S_c + \\lambda\\,\\mathrm{diag}(S_c)
                       + \\epsilon I,

    where :math:`S_c` is the sample covariance of the same-class
    half-vectorised log-anchors and :math:`\\lambda` is ``shrinkage``.
    Synthetic samples are drawn from
    :math:`\\mathcal{N}(\\mathbf{0}, \\hat\\Sigma_c)` and added to the
    anchor's tangent vector at scale ``sigma``.

    Parameters
    ----------
    cov_list, labels
        SPD anchors (must contain at least 2) and their integer/string
        labels.
    n_aug : int, default 5
    sigma : float, default 0.10
        Tangent perturbation scale.
    shrinkage : float in [0, 1], default 0.10
        Linear shrinkage toward the diagonal of the sample covariance.
    seed : int, optional
    """
    anchors = coerce_cov_stack(cov_list, "cov_list")
    n_anchor, p, _ = anchors.shape
    if n_anchor < 2:
        raise ValueError("cov_list must contain at least two anchors.")
    if n_aug < 1:
        raise ValueError("n_aug must be a positive integer.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if not (0 <= shrinkage <= 1):
        raise ValueError("shrinkage must lie in [0, 1].")
    n_aug = int(n_aug)
    labels_arr = coerce_labels(labels, n_anchor, required=True)
    rng = make_rng(seed)

    rows, cols = np.triu_indices(p)
    weight = np.where(rows == cols, 1.0, np.sqrt(2.0))
    d_dim = rows.size

    def _vech_log(C: np.ndarray) -> np.ndarray:
        Z = spd.spd_logm(C)
        return Z[rows, cols] * weight

    def _unvech(z: np.ndarray) -> np.ndarray:
        M = np.zeros((p, p))
        M[rows, cols] = z / weight
        M = M + M.T - np.diag(np.diag(M))
        return 0.5 * (M + M.T)

    z_anchors = np.stack([_vech_log(anchors[i]) for i in range(n_anchor)])

    classes = np.unique(labels_arr)
    cov_per_class = {}
    for cls in classes:
        mask = labels_arr == cls
        z_cls = z_anchors[mask]
        if z_cls.shape[0] < 2:
            cov_per_class[cls] = np.eye(d_dim)
            continue
        sample_cov = np.cov(z_cls, rowvar=False, ddof=1)
        target = np.diag(np.diag(sample_cov))
        cov_per_class[cls] = (
            (1.0 - shrinkage) * sample_cov
            + shrinkage * target
            + 1e-10 * np.eye(d_dim)
        )

    total = n_anchor * n_aug
    out_cov = np.empty((total, p, p), dtype=float)
    anchor_idx = np.empty(total, dtype=np.int64)
    rep_idx = np.empty(total, dtype=np.int64)
    out_labels = np.empty(total, dtype=labels_arr.dtype)

    pos = 0
    for i in range(n_anchor):
        cls = labels_arr[i]
        Sigma_c = cov_per_class[cls]
        try:
            chol_c = cholesky(Sigma_c, lower=False)
            use_chol = True
        except np.linalg.LinAlgError:
            chol_c = None
            ev_vals, ev_vecs = eigh(Sigma_c)
            ev_vals = np.maximum(ev_vals, 0.0)
            use_chol = False
        z_i = z_anchors[i]
        for r in range(n_aug):
            if use_chol:
                eps = rng.standard_normal(d_dim) @ chol_c
            else:
                eps = (rng.standard_normal(d_dim) * np.sqrt(ev_vals)) @ ev_vecs.T
            z_aug = z_i + sigma * eps
            out_cov[pos] = spd.spd_expm(_unvech(z_aug))
            anchor_idx[pos] = i
            rep_idx[pos] = r
            out_labels[pos] = cls
            pos += 1

    return {
        "cov": out_cov,
        "anchor": anchor_idx,
        "replicate": rep_idx,
        "labels": out_labels,
        "params": {
            "sigma": float(sigma),
            "shrinkage": float(shrinkage),
            "n_aug": n_aug,
            "seed": seed,
        },
    }
