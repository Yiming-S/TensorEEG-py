"""Internal helpers shared across modules.

Public-facing helpers (``rbf_kernel``, ``calc_ac_power``,
``tensor_to_cov``) are re-exported from the package root.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt

__all__ = [
    "rbf_kernel",
    "calc_ac_power",
    "tensor_to_cov",
]


def rbf_kernel(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float,
    standard_scale: bool = True,
) -> np.ndarray:
    r"""Pairwise Gaussian RBF kernel.

    .. math::

        K(x_i, y_j) = \\exp(-\\gamma \\|x_i - y_j\\|^2)

    With ``standard_scale=True`` (default), :math:`\\gamma = 1/(2\\sigma^2)`,
    matching ``stats::dnorm``-style bandwidth conventions; otherwise
    :math:`\\gamma = 1/\\sigma^2`.

    Parameters
    ----------
    x : (n_x, d) ndarray
        First set of points (rows = points).
    y : (n_y, d) ndarray
        Second set of points.
    sigma : float
        Bandwidth (must be positive).
    standard_scale : bool, default True
        See above.

    Returns
    -------
    (n_x, n_y) ndarray
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be 2D with the same number of columns.")
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError("sigma must be a positive finite number.")
    xx = np.sum(x * x, axis=1, keepdims=True)        # (n_x, 1)
    yy = np.sum(y * y, axis=1, keepdims=True).T      # (1, n_y)
    d2 = xx + yy - 2.0 * (x @ y.T)
    d2 = np.maximum(d2, 0.0)
    gamma = 1.0 / (2.0 * sigma ** 2) if standard_scale else 1.0 / (sigma ** 2)
    return np.exp(-gamma * d2)


def calc_ac_power(X: np.ndarray, fs: float) -> float:
    """Effective AC power after a 4th-order 0.1 Hz Butterworth high-pass.

    Used by :func:`tensoreeg.sim_eeg_master`'s closed-loop SNR control to
    measure neural / artifact / drift power without contamination from
    DC offset and slow drift.

    Parameters
    ----------
    X : (n_time, n_channels) ndarray
        Trial time series.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    float
        Mean squared power after high-pass filtering. Returns ``0.0`` for
        non-finite inputs or if filtering produces NaNs.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix [time x channels].")
    if X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError("X must have at least one row and one column.")
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a single positive finite number.")
    if not np.all(np.isfinite(X)):
        return 0.0
    b, a = butter(4, 0.1 / (fs / 2.0), btype="high")
    n_time = X.shape[0]
    # filtfilt requires len(x) > 3 * max(len(a), len(b)) for default padlen.
    padlen = min(3 * max(len(a), len(b)), n_time - 1)
    if padlen < 0:
        return 0.0
    X_filt = np.column_stack([
        filtfilt(b, a, X[:, k], padlen=padlen) for k in range(X.shape[1])
    ])
    val = float(np.mean(X_filt ** 2))
    return 0.0 if not np.isfinite(val) else val


def tensor_to_cov(
    X_tensor: np.ndarray,
    ridge: float = 1e-6,
    centre: bool = True,
) -> np.ndarray:
    r"""Trial-wise SPD covariance from a 3rd-order EEG tensor.

    Parameters
    ----------
    X_tensor : (n_time, n_channels, n_trials) ndarray
        Output of :func:`tensoreeg.sim_eeg_master` or any
        ``[time x channel x trial]`` tensor.
    ridge : float, default 1e-6
        Diagonal regularisation added to each covariance for strict
        positive definiteness.
    centre : bool, default True
        If True, mean-centre each trial per channel before the
        cross-product. Matches ``stats::cov`` and
        ``pyriemann.utils.covariance``.

    Returns
    -------
    cov_array : (n_trials, n_channels, n_channels) ndarray
        Stack of SPD covariance matrices, one per trial. Stored as a 3D
        array (more numpy-idiomatic than a list of 2D arrays).
    """
    if X_tensor.ndim != 3:
        raise ValueError("X_tensor must be a 3D array [time x channel x trial].")
    if not (np.isfinite(ridge) and ridge >= 0):
        raise ValueError("ridge must be a non-negative finite number.")
    n_time, n_channels, n_trials = X_tensor.shape
    if n_time < 2:
        raise ValueError("Each trial must have at least 2 time samples.")
    out = np.empty((n_trials, n_channels, n_channels), dtype=float)
    for k in range(n_trials):
        Xk = X_tensor[:, :, k]
        if centre:
            Xk = Xk - Xk.mean(axis=0, keepdims=True)
        C = (Xk.T @ Xk) / (n_time - 1)
        C = 0.5 * (C + C.T)
        out[k] = C + ridge * np.eye(n_channels)
    return out


def warn_synthetic_real_ratio(
    n_aug: int,
    n_anchor: int,
    ratio_threshold: int = 3,
    lower_anchor: int = 10,
    upper_anchor: int = 40,
) -> None:
    """Emit the budget-dependent failure warning documented in the paper.

    Restricted to the regime where the failure was empirically observed
    (``10 <= n_anchor < 40`` with ``r >= 3``); toy unit-test fixtures
    (``n_anchor < 10``) are skipped so they do not clutter test output.
    """
    if n_aug >= ratio_threshold and lower_anchor <= n_anchor < upper_anchor:
        warnings.warn(
            f"Synthetic-to-real ratio {int(n_aug)}:1 with only "
            f"{int(n_anchor)} real anchors. Isotropic tangent jitter (G0) "
            "showed a reproducible budget-dependent failure on "
            "BNCI2014_001 at ncal=30 with r=3 in Shen & Degras (2026); "
            "consider G2 (geodesic mixup) or a smaller n_aug if "
            "downstream accuracy is the goal.",
            UserWarning,
            stacklevel=3,
        )
