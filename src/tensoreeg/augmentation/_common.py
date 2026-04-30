"""Helpers shared by every augmentation routine."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

CovStack = Union[np.ndarray, Sequence[np.ndarray]]


def coerce_cov_stack(cov_list: CovStack, name: str = "cov_list") -> np.ndarray:
    """Validate and stack a sequence of SPD matrices into ``(n, p, p)``.

    Accepts either a list of square 2D ndarrays (one per anchor) or an
    already-stacked 3D ndarray. The returned stack is symmetrised
    (``0.5 * (A + A.T)``) to absorb any small float asymmetry.
    """
    if isinstance(cov_list, np.ndarray) and cov_list.ndim == 3:
        arr = cov_list.astype(float, copy=True)
    else:
        items = list(cov_list)
        if len(items) == 0:
            raise ValueError(f"{name} must be a non-empty list of SPD matrices.")
        first = np.asarray(items[0], dtype=float)
        if first.ndim != 2 or first.shape[0] != first.shape[1]:
            raise ValueError(
                f"{name}[0] must be a square 2D array, got shape {first.shape}"
            )
        p = first.shape[0]
        arr = np.empty((len(items), p, p), dtype=float)
        for i, C in enumerate(items):
            Ci = np.asarray(C, dtype=float)
            if Ci.shape != (p, p):
                raise ValueError(
                    f"{name}[{i}] must have shape ({p}, {p}), got {Ci.shape}"
                )
            arr[i] = Ci
    if arr.shape[1] != arr.shape[2]:
        raise ValueError(f"{name} entries must be square, got shape {arr.shape}")
    arr = 0.5 * (arr + arr.transpose(0, 2, 1))
    return arr


def coerce_labels(
    labels: Optional[Sequence], n_anchor: int, required: bool = False
) -> Optional[np.ndarray]:
    """Validate ``labels`` against the anchor count.

    Returns a 1D ndarray of length ``n_anchor`` or ``None`` if the input
    is ``None`` and ``required=False``.
    """
    if labels is None:
        if required:
            raise ValueError("labels must be provided for this augmentation.")
        return None
    arr = np.asarray(labels)
    if arr.ndim != 1 or arr.shape[0] != n_anchor:
        raise ValueError(
            f"labels must be a 1D vector of length {n_anchor}, got shape {arr.shape}"
        )
    return arr


def make_rng(seed: Optional[int]) -> np.random.Generator:
    """Build a numpy ``Generator`` from an optional seed.

    A scalar integer seed makes the augmentation reproducible within
    Python; ``None`` defers to NumPy's default entropy. Note that even
    with the same scalar seed the output is not byte-equivalent to the R
    reference, because R and NumPy use different underlying RNGs. The
    paper's reproducibility contract is at the metric level (fidelity
    audit numbers agree to a small tolerance), not at byte level
    cross-language.
    """
    if seed is None:
        return np.random.default_rng()
    if not (np.isscalar(seed) and float(seed).is_integer()):
        raise TypeError("seed must be a single integer or None.")
    return np.random.default_rng(int(seed))
