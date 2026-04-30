"""Manifold drift: rotations on SO(n) driven by an OU/AR(1) or fBm angle path."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.linalg import expm

__all__ = [
    "generate_drift_rotations",
]


def _angle_path_ou(
    n_trials: int,
    alpha_ou: float,
    sigma_eps: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """AR(1) / Ornstein-Uhlenbeck angle path."""
    theta = np.zeros(n_trials, dtype=float)
    if n_trials >= 2:
        noise = rng.normal(scale=sigma_eps, size=n_trials)
        for k in range(1, n_trials):
            theta[k] = alpha_ou * theta[k - 1] + noise[k]
    return theta


def _angle_path_fbm(
    n_trials: int,
    hurst: float,
    sigma_eps: float,
    rng: np.random.Generator,
    kernel_size: int = 64,
) -> np.ndarray:
    """Fractional Brownian motion angle path (Mandelbrot-Van Ness kernel)."""
    if n_trials == 1:
        return np.zeros(1, dtype=float)
    j = np.arange(kernel_size, dtype=float)
    kernel = (j + 1.0) ** (hurst - 0.5) - j ** (hurst - 0.5)
    white = rng.normal(scale=sigma_eps, size=n_trials + kernel_size)
    # One-sided convolution matching R's stats::filter(..., sides = 1).
    fgn = np.convolve(white, kernel, mode="full")[: white.size]
    fgn = fgn[kernel_size : kernel_size + n_trials]
    return np.cumsum(fgn)


def generate_drift_rotations(
    n_sources: int,
    n_trials: int,
    alpha_ou: float = 0.95,
    sigma_eps: float = 0.05,
    process: str = "ou",
    hurst: float = 0.85,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    r"""Trial-wise rotation matrices on :math:`SO(n)` (manifold drift).

    Models trial-to-trial non-stationarity as a geodesic walk on the Lie
    group of rotation matrices:

    1. Sample a fixed random skew-symmetric generator
       :math:`\\Omega_{\\mathrm{base}}`, normalised in Frobenius norm.
    2. Draw a scalar angle path :math:`\\theta_k` (OU/AR(1) by default;
       fBm if ``process="fbm"``).
    3. Map to :math:`R_k = \\exp(\\theta_k \\Omega_{\\mathrm{base}})`
       via the matrix exponential.

    Parameters
    ----------
    n_sources : int
        Dimension of the rotation matrices.
    n_trials : int
        Number of trials.
    alpha_ou : float, default 0.95
        AR(1) coefficient. Used only when ``process == "ou"``.
    sigma_eps : float, default 0.05
        Standard deviation of the angle innovations.
    process : {"ou", "fbm"}, default "ou"
        ``"ou"`` is the SimEEG benchmark drift used in Shen & Degras
        (2026); ``"fbm"`` is provided for sensitivity studies.
    hurst : float in (0, 1), default 0.85
        Hurst index for the fBm path. Ignored when ``process="ou"``.
    rng : numpy Generator, optional

    Returns
    -------
    list of (n_sources, n_sources) ndarray, length ``n_trials``
        Each entry is an orthogonal matrix on :math:`SO(n)`.
    """
    if process not in ("ou", "fbm"):
        raise ValueError("process must be 'ou' or 'fbm'.")
    if n_sources < 1:
        raise ValueError("n_sources must be a positive integer.")
    if n_trials < 1:
        raise ValueError("n_trials must be a positive integer.")
    if not np.isfinite(alpha_ou) or abs(alpha_ou) >= 1:
        raise ValueError("alpha_ou must be in (-1, 1).")
    if sigma_eps < 0 or not np.isfinite(sigma_eps):
        raise ValueError("sigma_eps must be a non-negative finite number.")
    if not (0 < hurst < 1):
        raise ValueError("hurst must be in (0, 1).")
    n_sources = int(n_sources)
    n_trials = int(n_trials)
    if rng is None:
        rng = np.random.default_rng()

    if n_sources == 1:
        return [np.array([[1.0]]) for _ in range(n_trials)]

    G = rng.normal(size=(n_sources, n_sources))
    Omega = 0.5 * (G - G.T)
    frob = float(np.linalg.norm(Omega))
    if frob <= np.finfo(float).eps:
        return [np.eye(n_sources) for _ in range(n_trials)]
    Omega /= frob

    if process == "ou":
        theta = _angle_path_ou(n_trials, alpha_ou, sigma_eps, rng)
    else:
        theta = _angle_path_fbm(n_trials, hurst, sigma_eps, rng)

    return [expm(theta[k] * Omega) for k in range(n_trials)]
