"""SPD (symmetric positive-definite) manifold utilities.

Functions in this module are the numerical foundation for the whole
package: every covariance augmentation routine, fidelity metric and
Riemannian alignment step is built on top of ``spd_logm``, ``spd_expm``,
``spd_project`` and the SPD distances defined here.

We use eigendecomposition (``scipy.linalg.eigh``) rather than a Schur
decomposition to compute matrix logs/exponentials, mirroring the R
reference implementation: it is faster, more numerically stable on SPD
inputs, and avoids the complex-eigenvalue warnings ``scipy.linalg.logm``
can emit on rounding-perturbed inputs.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import eigh

__all__ = [
    "ensure_symmetric",
    "spd_logm",
    "spd_expm",
    "spd_project",
    "log_euclidean_distance",
    "affine_invariant_distance",
    "vech_indices",
    "vech_log",
    "unvech",
]


def _as_2d(C: np.ndarray, name: str = "C") -> np.ndarray:
    arr = np.asarray(C, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D array, got shape {arr.shape}")
    return arr


def ensure_symmetric(C: np.ndarray) -> np.ndarray:
    """Return ``0.5 * (C + C.T)`` so the input is exactly symmetric.

    Float arithmetic in eigendecompositions can otherwise produce a tiny
    asymmetric residual that compounds across calls.
    """
    A = _as_2d(C)
    return 0.5 * (A + A.T)


def spd_logm(C: np.ndarray) -> np.ndarray:
    """Symmetric matrix logarithm of an SPD matrix via eigendecomposition.

    Parameters
    ----------
    C : (p, p) ndarray
        A symmetric positive-definite matrix.

    Returns
    -------
    (p, p) ndarray
        ``V diag(log(d)) V.T`` where ``V, d`` are the eigenvectors and
        eigenvalues of the symmetrised input.

    Raises
    ------
    ValueError
        If any eigenvalue is non-positive (use ``ridge`` in
        :func:`tensoreeg.tensor_to_cov` to keep the cone strictly open).
    """
    A = ensure_symmetric(C)
    vals, vecs = eigh(A)
    if np.any(vals <= 0):
        raise ValueError(
            "Covariance is not strictly positive definite "
            f"(min eigenvalue {float(np.min(vals)):.3e}). "
            "Increase 'ridge' in tensor_to_cov()."
        )
    return (vecs * np.log(vals)) @ vecs.T


def spd_expm(Z: np.ndarray) -> np.ndarray:
    """Symmetric matrix exponential of a symmetric matrix.

    Returns an SPD matrix when ``Z`` is symmetric.
    """
    A = ensure_symmetric(Z)
    vals, vecs = eigh(A)
    return (vecs * np.exp(vals)) @ vecs.T


def spd_project(M: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix onto the SPD cone by clipping eigenvalues.

    Used by :func:`tensoreeg.augmentation.augment_cov_amplitude_matched_euclidean`
    (E0) when the additive Euclidean perturbation pushes the result outside
    the cone.
    """
    A = ensure_symmetric(M)
    vals, vecs = eigh(A)
    vals = np.maximum(vals, floor)
    return (vecs * vals) @ vecs.T


def log_euclidean_distance(C1: np.ndarray, C2: np.ndarray) -> float:
    r"""Log-Euclidean distance ``\|log(C1) - log(C2)\|_F``.

    Equivalent to the geodesic distance under the affine-invariant metric
    in a neighbourhood of the identity (Arsigny et al., 2007).
    """
    A1 = _as_2d(C1, "C1")
    A2 = _as_2d(C2, "C2")
    if A1.shape != A2.shape:
        raise ValueError("C1 and C2 must have the same shape.")
    Z1 = spd_logm(A1)
    Z2 = spd_logm(A2)
    return float(np.linalg.norm(Z1 - Z2))


def affine_invariant_distance(
    C1: np.ndarray, C2: np.ndarray, eigenvalue_floor: float = 1e-12
) -> float:
    r"""Affine-invariant Riemannian distance.

    .. math::

        d_{\\mathrm{AI}}(C_1, C_2) =
            \\big\\| \\log(C_1^{-1/2} C_2 C_1^{-1/2}) \\big\\|_F.
    """
    A1 = _as_2d(C1, "C1")
    A2 = _as_2d(C2, "C2")
    if A1.shape != A2.shape:
        raise ValueError("C1 and C2 must have the same shape.")
    A1 = ensure_symmetric(A1)
    A2 = ensure_symmetric(A2)
    vals, vecs = eigh(A1)
    vals = np.maximum(vals, eigenvalue_floor)
    inv_sqrt = (vecs * (vals ** -0.5)) @ vecs.T
    middle = inv_sqrt @ A2 @ inv_sqrt
    return float(np.linalg.norm(spd_logm(middle)))


def vech_indices(p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Row/col indices and weights for half-vectorisation of a ``p x p``
    symmetric matrix.

    The ordering matches NumPy's column-major upper-triangular (no
    explicit transpose); this is consistent with the R reference because
    the resulting ``vech_log -> unvech`` round trip is invariant to the
    chosen ordering when the same convention is used at both ends.

    Returns
    -------
    idx : (d, 2) int ndarray
        ``(i, j)`` pairs with ``i <= j``, lexicographic order.
    weight : (d,) float ndarray
        ``1`` on the diagonal, ``sqrt(2)`` off the diagonal. The weight
        preserves the Frobenius inner product when working in the
        half-vectorised representation.
    """
    if p < 1:
        raise ValueError("p must be at least 1")
    rows, cols = np.triu_indices(p)
    idx = np.column_stack([rows, cols])
    weight = np.where(rows == cols, 1.0, np.sqrt(2.0))
    return idx, weight


def vech_log(C: np.ndarray) -> np.ndarray:
    r"""Half-vectorisation of ``log(C)`` with sqrt(2) off-diagonal weight."""
    Z = spd_logm(C)
    rows, cols = np.triu_indices(Z.shape[0])
    weight = np.where(rows == cols, 1.0, np.sqrt(2.0))
    return Z[rows, cols] * weight


def unvech(z: np.ndarray, p: int) -> np.ndarray:
    """Inverse of :func:`vech_log`: return the symmetric matrix whose
    half-vectorisation (with the same sqrt(2) weights) is ``z``.
    """
    z = np.asarray(z, dtype=float)
    rows, cols = np.triu_indices(p)
    weight = np.where(rows == cols, 1.0, np.sqrt(2.0))
    if z.size != rows.size:
        raise ValueError(
            f"z has size {z.size}, expected {rows.size} for p={p}"
        )
    M = np.zeros((p, p))
    M[rows, cols] = z / weight
    M = M + M.T - np.diag(np.diag(M))
    return 0.5 * (M + M.T)
