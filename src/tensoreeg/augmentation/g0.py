"""G0: log-Euclidean isotropic tangent jitter."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .. import spd
from .._utils import warn_synthetic_real_ratio
from ._common import CovStack, coerce_cov_stack, coerce_labels, make_rng

__all__ = ["augment_cov_riemannian"]


def augment_cov_riemannian(
    cov_list: CovStack,
    n_aug: int = 5,
    sigma: float = 0.10,
    drift: Optional[np.ndarray] = None,
    labels: Optional[Sequence] = None,
    seed: Optional[int] = None,
) -> dict:
    r"""Riemannian (log-Euclidean) augmentation of trial covariance matrices.

    For each anchor :math:`C_i` and each of ``n_aug`` replicates the
    procedure is

    1. :math:`Z_i \\leftarrow \\log(C_i)` (symmetric matrix logarithm),
    2. sample symmetric Gaussian perturbation
       :math:`E_i = (G_i + G_i^{\\top})/2` with
       :math:`G_i \\sim \\mathcal{N}(0, \\sigma^2 I)`,
    3. optionally add a session-drift term :math:`D` (also symmetric),
    4. :math:`\\tilde Z_i = Z_i + E_i + D`,
    5. :math:`\\tilde C_i = \\exp(\\tilde Z_i)`.

    Parameters
    ----------
    cov_list : list of (p, p) ndarray, or (n, p, p) ndarray
        SPD anchor covariances.
    n_aug : int, default 5
        Number of synthetic replicates per anchor.
    sigma : float, default 0.10
        Standard deviation of the symmetric tangent-space perturbation.
    drift : (p, p) ndarray, optional
        Session-drift term added (after symmetrisation) to every replicate.
    labels : sequence, optional
        Length ``n_anchor``. Each replicate inherits the anchor's label.
    seed : int, optional
        RNG seed. Reproducible within Python.

    Returns
    -------
    result : dict with keys
        ``cov`` (n_anchor*n_aug, p, p) ndarray,
        ``anchor`` (n_anchor*n_aug,) int ndarray,
        ``replicate`` (n_anchor*n_aug,) int ndarray,
        ``labels`` ndarray or None,
        ``params`` dict.
    """
    anchors = coerce_cov_stack(cov_list, "cov_list")
    n_anchor, p, _ = anchors.shape
    if n_aug < 1:
        raise ValueError("n_aug must be a positive integer.")
    n_aug = int(n_aug)
    if not (np.isfinite(sigma) and sigma >= 0):
        raise ValueError("sigma must be a single non-negative finite number.")
    warn_synthetic_real_ratio(n_aug, n_anchor)

    drift_sym: Optional[np.ndarray] = None
    if drift is not None:
        drift = np.asarray(drift, dtype=float)
        if drift.shape != (p, p):
            raise ValueError(f"drift must be a {p} x {p} matrix.")
        drift_sym = 0.5 * (drift + drift.T)

    labels_arr = coerce_labels(labels, n_anchor)
    rng = make_rng(seed)

    total = n_anchor * n_aug
    out_cov = np.empty((total, p, p), dtype=float)
    anchor_idx = np.empty(total, dtype=np.int64)
    rep_idx = np.empty(total, dtype=np.int64)
    out_labels = None if labels_arr is None else np.empty(total, dtype=labels_arr.dtype)

    pos = 0
    for i in range(n_anchor):
        Z_i = spd.spd_logm(anchors[i])
        for r in range(n_aug):
            G = rng.normal(scale=sigma, size=(p, p))
            E = 0.5 * (G + G.T)
            Z_aug = Z_i + E
            if drift_sym is not None:
                Z_aug = Z_aug + drift_sym
            out_cov[pos] = spd.spd_expm(Z_aug)
            anchor_idx[pos] = i
            rep_idx[pos] = r
            if out_labels is not None:
                out_labels[pos] = labels_arr[i]
            pos += 1

    return {
        "cov": out_cov,
        "anchor": anchor_idx,
        "replicate": rep_idx,
        "labels": out_labels,
        "params": {"sigma": float(sigma), "n_aug": n_aug, "seed": seed},
    }
