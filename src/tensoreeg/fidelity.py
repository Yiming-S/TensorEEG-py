"""Six-metric covariance fidelity audit.

The metrics implemented here are the ones reported in Table 5 of Shen
& Degras (2026):

- log-Euclidean distance to a class-mean reference,
- affine-invariant Riemannian distance to the same reference,
- Pearson correlation of eigenvalue spectra,
- trace ratio,
- condition-number ratio,
- mean anchor-to-synthetic perturbation distance.

Plus, in the wrapper :func:`audit_covariance_fidelity`, the dimension-
normalised versions of every distance metric (divided by
:math:`\\sqrt{p(p+1)/2}`) so that pooled audits across datasets with
different channel counts are not dominated by the trivial dimension
scaling.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from scipy.linalg import eigh

from . import spd
from .augmentation._common import CovStack, coerce_cov_stack

__all__ = [
    "cov_logeuclidean_distance",
    "cov_affine_invariant_distance",
    "cov_eigenvalue_correlation",
    "cov_trace_ratio",
    "cov_condition_ratio",
    "cov_anchor_perturbation_distance",
    "audit_covariance_fidelity",
]


def cov_logeuclidean_distance(C1: np.ndarray, C2: np.ndarray) -> float:
    """Log-Euclidean Frobenius distance.

    Convenience wrapper around :func:`tensoreeg.spd.log_euclidean_distance`
    so the fidelity module can be imported without reaching into ``spd``.
    """
    return spd.log_euclidean_distance(C1, C2)


def cov_affine_invariant_distance(
    C1: np.ndarray, C2: np.ndarray, eigenvalue_floor: float = 1e-12
) -> float:
    """Affine-invariant Riemannian distance (Pennec et al., 2006)."""
    return spd.affine_invariant_distance(C1, C2, eigenvalue_floor=eigenvalue_floor)


def cov_eigenvalue_correlation(C1: np.ndarray, C2: np.ndarray) -> float:
    """Pearson correlation of the descending eigenvalue spectra.

    Returns ``nan`` if either spectrum has zero variance.
    """
    A1 = spd.ensure_symmetric(C1)
    A2 = spd.ensure_symmetric(C2)
    if A1.shape != A2.shape:
        raise ValueError("C1 and C2 must have the same shape.")
    v1 = np.sort(eigh(A1, eigvals_only=True))[::-1]
    v2 = np.sort(eigh(A2, eigvals_only=True))[::-1]
    if v1.std() == 0 or v2.std() == 0:
        return float("nan")
    return float(np.corrcoef(v1, v2)[0, 1])


def cov_trace_ratio(C_aug: np.ndarray, C_real: np.ndarray) -> float:
    r"""Trace ratio :math:`\\mathrm{tr}(C_{\\mathrm{aug}}) / \\mathrm{tr}(C_{\\mathrm{real}})`."""
    A_aug = np.asarray(C_aug, dtype=float)
    A_real = np.asarray(C_real, dtype=float)
    if A_aug.shape != A_real.shape:
        raise ValueError("C_aug and C_real must have the same shape.")
    return float(np.trace(A_aug) / max(np.trace(A_real), 1e-12))


def cov_condition_ratio(C_aug: np.ndarray, C_real: np.ndarray) -> float:
    r"""Condition-number ratio :math:`\\kappa(C_{\\mathrm{aug}}) / \\kappa(C_{\\mathrm{real}})`."""
    A_aug = spd.ensure_symmetric(C_aug)
    A_real = spd.ensure_symmetric(C_real)
    if A_aug.shape != A_real.shape:
        raise ValueError("C_aug and C_real must have the same shape.")
    e_aug = eigh(A_aug, eigvals_only=True)
    e_real = eigh(A_real, eigvals_only=True)
    k_aug = float(np.max(e_aug) / max(np.min(e_aug), 1e-12))
    k_real = float(np.max(e_real) / max(np.min(e_real), 1e-12))
    return k_aug / max(k_real, 1e-12)


def cov_anchor_perturbation_distance(
    anchors: CovStack, synthetic: CovStack, anchor_idx: Sequence[int]
) -> float:
    """Mean log-Euclidean distance from each synthetic to its anchor."""
    anchors_arr = coerce_cov_stack(anchors, "anchors")
    synth_arr = coerce_cov_stack(synthetic, "synthetic")
    anchor_idx_arr = np.asarray(anchor_idx, dtype=np.int64)
    if synth_arr.shape[0] != anchor_idx_arr.size:
        raise ValueError("anchor_idx must have length equal to len(synthetic).")
    if synth_arr.shape[0] == 0:
        return float("nan")
    anchors_log = np.stack(
        [spd.spd_logm(anchors_arr[i]) for i in range(anchors_arr.shape[0])]
    )
    d = np.empty(synth_arr.shape[0], dtype=float)
    for k in range(synth_arr.shape[0]):
        diff = spd.spd_logm(synth_arr[k]) - anchors_log[anchor_idx_arr[k]]
        d[k] = float(np.linalg.norm(diff))
    return float(np.mean(d))


def audit_covariance_fidelity(
    anchors: CovStack,
    synthetic: CovStack,
    anchor_idx: Sequence[int],
    reference: Optional[np.ndarray] = None,
) -> dict:
    r"""Six-metric covariance fidelity audit.

    Returns the same fields as the R reference, plus a ``normalized``
    sub-dict containing the distance metrics divided by
    :math:`\\sqrt{p(p+1)/2}` for cross-channel-count pooling.

    Parameters
    ----------
    anchors : list of (p, p) ndarray, or (n_anchor, p, p) ndarray
        SPD anchor covariances.
    synthetic : list of (p, p) ndarray, or (n_syn, p, p) ndarray
        SPD synthetic covariances.
    anchor_idx : sequence of int
        Length ``n_syn``. Maps each synthetic to its anchor.
    reference : (p, p) ndarray, optional
        Per-class reference for the LE / AI / spectrum / trace / condition
        metrics. Defaults to the symmetrised arithmetic mean of
        ``anchors``.
    """
    anchors_arr = coerce_cov_stack(anchors, "anchors")
    synth_arr = coerce_cov_stack(synthetic, "synthetic")
    n_syn = synth_arr.shape[0]
    if n_syn == 0:
        return {
            "n_synthetic": 0,
            "log_euclidean_to_reference": float("nan"),
            "affine_invariant_to_reference": float("nan"),
            "eigenvalue_correlation": float("nan"),
            "trace_ratio": float("nan"),
            "condition_number_ratio": float("nan"),
            "anchor_perturbation_distance": float("nan"),
            "normalized": {},
        }

    if reference is None:
        ref = np.mean(anchors_arr, axis=0)
        reference = 0.5 * (ref + ref.T)
    else:
        reference = spd.ensure_symmetric(reference)

    syn_mean = np.mean(synth_arr, axis=0)
    syn_mean = 0.5 * (syn_mean + syn_mean.T)

    le_to_ref = float(np.mean([
        cov_logeuclidean_distance(synth_arr[k], reference) for k in range(n_syn)
    ]))
    ai_to_ref = float(np.mean([
        cov_affine_invariant_distance(synth_arr[k], reference) for k in range(n_syn)
    ]))
    eig_corr = cov_eigenvalue_correlation(syn_mean, reference)
    tr_ratio = cov_trace_ratio(syn_mean, reference)
    cond_rat = cov_condition_ratio(syn_mean, reference)
    anc_dist = cov_anchor_perturbation_distance(anchors_arr, synth_arr, anchor_idx)

    p = reference.shape[0]
    feature_dim = p * (p + 1) / 2
    d_norm = np.sqrt(feature_dim)

    return {
        "n_synthetic": n_syn,
        "log_euclidean_to_reference": le_to_ref,
        "affine_invariant_to_reference": ai_to_ref,
        "eigenvalue_correlation": eig_corr,
        "trace_ratio": tr_ratio,
        "condition_number_ratio": cond_rat,
        "anchor_perturbation_distance": anc_dist,
        "normalized": {
            "log_euclidean": le_to_ref / d_norm,
            "affine_invariant": ai_to_ref / d_norm,
            "anchor_perturbation": anc_dist / d_norm,
            "feature_dim": feature_dim,
        },
    }
