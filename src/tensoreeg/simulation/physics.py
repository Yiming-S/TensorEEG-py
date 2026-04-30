"""Volume-conduction geometry: Fibonacci hemisphere + Laplacian smoothing."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import squareform, pdist

from .._utils import rbf_kernel

__all__ = ["generate_geometry_mixing"]


def generate_geometry_mixing(
    n_channels: int = 64,
    n_sources: int = 10,
    sigma_geo: float = 0.3,
    lambda_smooth: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    r"""Sensor + source coordinates and a Laplacian-smoothed mixing matrix.

    Implements a graph-theoretic stand-in for full BEM/FEM forward
    modelling. Sensors are placed on a Fibonacci lattice on the upper
    hemisphere (:math:`z \\ge 0`); sources are random points inside a
    sphere of radius 0.8 (so they are 'deep' in the brain volume); the
    mixing matrix is a Tikhonov-smoothed RBF lead field

    .. math::

        A_{\\mathrm{smooth}} = (I + \\lambda L_{\\mathrm{sym}})^{-1} A_{\\mathrm{raw}},

    where :math:`L_{\\mathrm{sym}}` is the normalised graph Laplacian of
    the k-nearest-neighbour sensor graph. Columns are unit-norm so each
    source contributes equal energy.

    Parameters
    ----------
    n_channels, n_sources : int
        Counts.
    sigma_geo : float, default 0.3
        RBF bandwidth for the raw lead field.
    lambda_smooth : float, default 0.5
        Tikhonov regularisation for Laplacian smoothing. Higher =>
        smoother, more biologically plausible topographies.
    rng : numpy Generator, optional
        Source of randomness. Pass an existing generator to share RNG
        state with a calling routine; otherwise a fresh default is used.

    Returns
    -------
    dict with keys
        ``coords_sens`` (n_channels, 3),
        ``coords_src``  (n_sources, 3),
        ``A_base``      (n_channels, n_sources),
        ``L_sym``       (n_channels, n_channels) normalised graph Laplacian.
    """
    if n_channels < 1:
        raise ValueError("n_channels must be a positive integer.")
    if n_sources < 1:
        raise ValueError("n_sources must be a positive integer.")
    if not (np.isfinite(sigma_geo) and sigma_geo > 0):
        raise ValueError("sigma_geo must be a single positive finite number.")
    if not (np.isfinite(lambda_smooth) and lambda_smooth >= 0):
        raise ValueError("lambda_smooth must be a non-negative finite number.")
    n_channels = int(n_channels)
    n_sources = int(n_sources)
    if rng is None:
        rng = np.random.default_rng()

    # 1. Sensor coordinates: Fibonacci lattice on hemisphere (z uniform).
    idx = np.arange(n_channels)
    z = rng.uniform(0.0, 1.0, size=n_channels)
    theta = (np.sqrt(5.0) * np.pi * idx) % (2.0 * np.pi)
    r_xy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)
    coords_sens = np.column_stack([x, y, z])

    # 2. Source coordinates: rejection-sample uniform inside r=0.8 ball.
    coords_src = np.empty((n_sources, 3), dtype=float)
    count = 0
    while count < n_sources:
        pt = rng.uniform(-0.8, 0.8, size=3)
        if np.dot(pt, pt) < 0.8 ** 2:
            coords_src[count] = pt
            count += 1

    # 3. Raw RBF lead field.
    A_raw = rbf_kernel(coords_sens, coords_src, sigma=sigma_geo, standard_scale=True)

    # 4. Normalised graph Laplacian on the sensor graph.
    if n_channels == 1:
        L_sym = np.zeros((1, 1))
    else:
        dist_SS = squareform(pdist(coords_sens))
        k_nn = min(4, n_channels - 1)
        upper = dist_SS[np.triu_indices(n_channels, k=1)]
        sigma_w = float(np.mean(upper))
        if not np.isfinite(sigma_w) or sigma_w <= 0:
            sigma_w = 1.0
        W = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            order = np.argsort(dist_SS[i])
            nbs = order[1 : k_nn + 1]
            w = np.exp(-dist_SS[i, nbs] ** 2 / (2.0 * sigma_w ** 2))
            W[i, nbs] = w
            W[nbs, i] = w
        deg = W.sum(axis=1)
        deg = np.where(deg == 0, 1.0, deg)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        L_sym = np.eye(n_channels) - D_inv_sqrt @ W @ D_inv_sqrt

    H = np.linalg.solve(np.eye(n_channels) + lambda_smooth * L_sym, np.eye(n_channels))
    A_smooth = H @ A_raw

    col_norms = np.linalg.norm(A_smooth, axis=0)
    col_norms = np.where(col_norms == 0, 1.0, col_norms)
    A_base = A_smooth / col_norms

    return {
        "coords_sens": coords_sens,
        "coords_src": coords_src,
        "A_base": A_base,
        "L_sym": L_sym,
    }
