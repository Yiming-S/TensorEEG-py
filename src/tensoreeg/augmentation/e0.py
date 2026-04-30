"""E0: amplitude-matched Euclidean covariance perturbation (off-manifold control)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .. import spd
from ._common import CovStack, coerce_cov_stack, coerce_labels, make_rng

__all__ = ["augment_cov_amplitude_matched_euclidean"]


def _anchor_log_distances(
    anchors_log: np.ndarray, augmented: np.ndarray, anchor_idx: np.ndarray
) -> np.ndarray:
    """Per-synthetic log-Euclidean distance back to its anchor."""
    out = np.empty(len(augmented), dtype=float)
    for k, C in enumerate(augmented):
        diff = spd.spd_logm(C) - anchors_log[anchor_idx[k]]
        out[k] = float(np.linalg.norm(diff))
    return out


def augment_cov_amplitude_matched_euclidean(
    cov_list: CovStack,
    n_aug: int = 5,
    g0_sigma: float = 0.10,
    labels: Optional[Sequence] = None,
    seed: Optional[int] = None,
    eigenvalue_floor: float = 1e-6,
    tolerance_relative: float = 0.10,
) -> dict:
    r"""Amplitude-matched Euclidean covariance perturbation (E0).

    Off-manifold negative-control augmentation. Calibrates a single
    Euclidean noise scale :math:`\\sigma_E` so that the median anchor-to-
    augmented log-Euclidean distance matches that of G0 at ``g0_sigma``.

    Procedure:

    1. Generate the would-be G0 perturbations and record their median
       Frobenius norm in tangent space, ``target_distance``.
    2. Search a 41-point log-spaced grid in :math:`[10^{-3}, 10^{2}]`
       for the :math:`\\sigma_E` whose realised median distance matches
       ``target_distance``.
    3. Run E0 at the chosen :math:`\\sigma_E`: for each anchor and
       replicate, sample symmetric Gaussian noise in the **ambient**
       Euclidean space, add it to :math:`C_i`, and project back to the
       SPD cone by clipping eigenvalues at ``eigenvalue_floor``.

    Returns
    -------
    result : dict
        Same shape as :func:`augment_cov_riemannian` plus a
        ``diagnostic`` dict containing ``rho`` (achieved /
        ``target_distance``) and a ``success`` flag.
    """
    anchors = coerce_cov_stack(cov_list, "cov_list")
    n_anchor, p, _ = anchors.shape
    if n_aug < 1:
        raise ValueError("n_aug must be a positive integer.")
    n_aug = int(n_aug)
    if not (np.isfinite(g0_sigma) and g0_sigma > 0):
        raise ValueError("g0_sigma must be a single positive number.")
    if not (np.isfinite(eigenvalue_floor) and eigenvalue_floor > 0):
        raise ValueError("eigenvalue_floor must be positive.")
    labels_arr = coerce_labels(labels, n_anchor)
    rng = make_rng(seed)

    total = n_anchor * n_aug
    anchors_log = np.stack([spd.spd_logm(anchors[i]) for i in range(n_anchor)])

    # 1. G0 target distances (deterministic given the same RNG sequence).
    g0_log_distances = np.empty(total, dtype=float)
    pos = 0
    for i in range(n_anchor):
        for _ in range(n_aug):
            G = rng.normal(scale=g0_sigma, size=(p, p))
            E = 0.5 * (G + G.T)
            g0_log_distances[pos] = float(np.linalg.norm(E))
            pos += 1
    target_distance = float(np.median(g0_log_distances))

    # 2. Sigma search grid.
    sigma_grid = np.logspace(-3, 2, num=41)

    def distance_at_sigma(sigma_e: float):
        out = np.empty((total, p, p), dtype=float)
        idx = np.empty(total, dtype=np.int64)
        pos = 0
        for i in range(n_anchor):
            Ci = anchors[i]
            for _ in range(n_aug):
                N = rng.normal(scale=sigma_e, size=(p, p))
                H = 0.5 * (N + N.T)
                out[pos] = spd.spd_project(Ci + H, floor=eigenvalue_floor)
                idx[pos] = i
                pos += 1
        med = float(np.median(_anchor_log_distances(anchors_log, out, idx)))
        return out, idx, med

    achieved = np.array(
        [distance_at_sigma(s)[2] for s in sigma_grid], dtype=float
    )
    best_idx = int(np.argmin(np.abs(achieved - target_distance)))
    sigma_e = float(sigma_grid[best_idx])

    out_cov, anchor_idx, achieved_med = distance_at_sigma(sigma_e)
    rho = achieved_med / target_distance if target_distance > 0 else float("nan")

    rep_idx = np.tile(np.arange(n_aug, dtype=np.int64), n_anchor)
    out_labels = None if labels_arr is None else labels_arr[anchor_idx]

    return {
        "cov": out_cov,
        "anchor": anchor_idx,
        "replicate": rep_idx,
        "labels": out_labels,
        "params": {
            "g0_sigma": float(g0_sigma),
            "sigma_e": sigma_e,
            "n_aug": n_aug,
            "seed": seed,
            "eigenvalue_floor": float(eigenvalue_floor),
        },
        "diagnostic": {
            "target_distance_g0": target_distance,
            "achieved_distance_e0": achieved_med,
            "rho": rho,
            "tolerance": float(tolerance_relative),
            "success": bool(abs(rho - 1.0) <= tolerance_relative)
            if np.isfinite(rho) else False,
        },
    }
