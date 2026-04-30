"""G2: log-Euclidean geodesic mixup (the most fidelity-preserving variant)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .. import spd
from ._common import CovStack, coerce_cov_stack, coerce_labels, make_rng

__all__ = ["augment_cov_geodesic_mixup"]


def augment_cov_geodesic_mixup(
    cov_list: CovStack,
    labels: Sequence,
    n_aug: int = 5,
    beta_alpha: float = 1.0,
    seed: Optional[int] = None,
) -> dict:
    r"""Class-aware log-Euclidean geodesic mixup (G2).

    For each anchor :math:`C_i` and each replicate, a same-class partner
    :math:`C_j` and a Beta-distributed weight
    :math:`\\alpha \\sim \\mathrm{Beta}(a, a)` are sampled, and the
    synthetic covariance is

    .. math::

        \\tilde C =
            \\exp\\!\\big( (1-\\alpha) \\log C_i + \\alpha \\log C_j \\big).

    G2 was the most stable geometry-preserving variant in the audit of
    Shen & Degras (2026), with pooled paired-difference SD
    :math:`\\hat\\tau = 0.08` pp across four datasets at
    :math:`n_\\mathrm{cal} = 20`.

    Returns
    -------
    result : dict with the standard augmentation keys plus a ``trace``
        sub-dict containing the partner index and Beta sample for each
        synthetic.
    """
    anchors = coerce_cov_stack(cov_list, "cov_list")
    n_anchor, p, _ = anchors.shape
    if n_anchor < 2:
        raise ValueError("cov_list must contain at least two anchors.")
    if n_aug < 1:
        raise ValueError("n_aug must be a positive integer.")
    if beta_alpha <= 0:
        raise ValueError("beta_alpha must be positive.")
    n_aug = int(n_aug)
    labels_arr = coerce_labels(labels, n_anchor, required=True)
    rng = make_rng(seed)

    anchors_log = np.stack([spd.spd_logm(anchors[i]) for i in range(n_anchor)])

    classes = np.unique(labels_arr)
    same_class = {cls: np.where(labels_arr == cls)[0] for cls in classes}

    total = n_anchor * n_aug
    out_cov = np.empty((total, p, p), dtype=float)
    anchor_idx = np.empty(total, dtype=np.int64)
    rep_idx = np.empty(total, dtype=np.int64)
    partner_idx = np.empty(total, dtype=np.int64)
    alpha_vals = np.empty(total, dtype=float)
    out_labels = np.empty(total, dtype=labels_arr.dtype)

    pos = 0
    for i in range(n_anchor):
        cls = labels_arr[i]
        candidates = same_class[cls][same_class[cls] != i]
        if candidates.size == 0:
            candidates = same_class[cls]
        Z_i = anchors_log[i]
        for r in range(n_aug):
            partner = (
                int(candidates[0])
                if candidates.size == 1
                else int(rng.choice(candidates))
            )
            alpha = float(rng.beta(beta_alpha, beta_alpha))
            Z_aug = (1.0 - alpha) * Z_i + alpha * anchors_log[partner]
            out_cov[pos] = spd.spd_expm(Z_aug)
            anchor_idx[pos] = i
            rep_idx[pos] = r
            partner_idx[pos] = partner
            alpha_vals[pos] = alpha
            out_labels[pos] = cls
            pos += 1

    return {
        "cov": out_cov,
        "anchor": anchor_idx,
        "replicate": rep_idx,
        "labels": out_labels,
        "trace": {"partner": partner_idx, "alpha": alpha_vals},
        "params": {
            "beta_alpha": float(beta_alpha),
            "n_aug": n_aug,
            "seed": seed,
        },
    }
