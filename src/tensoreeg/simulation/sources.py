"""Source-level signal generation: VAR(2) background + Gabor task ERP."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "setup_var2_system",
    "sim_source_var2",
    "sim_source_task",
]


def setup_var2_system(
    n_sources: int,
    fs: float,
    target_freqs: Sequence[float],
    rng: Optional[np.random.Generator] = None,
) -> dict:
    r"""Construct a stable VAR(2) coefficient matrix pair.

    Diagonal entries place poles at magnitude ``r_pole = 0.95`` so each
    source oscillates at the corresponding ``target_freqs`` value;
    off-diagonal entries draw a sparse random coupling pattern. The
    coupling magnitude is iteratively scaled down (``gamma``) until the
    spectral radius of the companion matrix drops below 0.995, ensuring
    a stable VAR(2) process.

    Returns
    -------
    dict with keys
        ``Phi1``, ``Phi2``: ``(n_sources, n_sources)`` lag-1 / lag-2
        coefficient matrices.
    """
    if n_sources < 1:
        raise ValueError("n_sources must be a positive integer.")
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a single positive finite number.")
    target = np.asarray(target_freqs, dtype=float)
    if target.shape != (n_sources,):
        raise ValueError("target_freqs must be a vector of length n_sources.")
    nyq = fs / 2.0
    if np.any(target <= 0) or np.any(target >= nyq):
        raise ValueError(f"target_freqs values must be in (0, fs/2 = {nyq:.3f}).")
    if rng is None:
        rng = np.random.default_rng()
    n_sources = int(n_sources)

    Phi1 = np.zeros((n_sources, n_sources))
    Phi2 = np.zeros((n_sources, n_sources))
    r_pole = 0.95
    for i in range(n_sources):
        omega = 2.0 * np.pi * target[i] / fs
        Phi1[i, i] = 2.0 * r_pole * np.cos(omega)
        Phi2[i, i] = -(r_pole ** 2)

    coupling_mask = (rng.uniform(size=(n_sources, n_sources)) < 0.2).astype(float)
    np.fill_diagonal(coupling_mask, 0.0)
    couplings_base = rng.normal(scale=0.05, size=(n_sources, n_sources)) * coupling_mask

    is_stable = False
    gamma = 1.0
    Phi1_curr = Phi1.copy()
    while (not is_stable) and (gamma > 0.01):
        Phi1_curr = np.diag(np.diag(Phi1)) + gamma * couplings_base
        top = np.hstack([Phi1_curr, Phi2])
        bot = np.hstack([np.eye(n_sources), np.zeros((n_sources, n_sources))])
        comp = np.vstack([top, bot])
        rho = float(np.max(np.abs(np.linalg.eigvals(comp))))
        if rho < 0.995:
            is_stable = True
            Phi1 = Phi1_curr
        else:
            gamma *= 0.9
    if not is_stable:
        Phi1 = np.diag(np.diag(Phi1))

    return {"Phi1": Phi1, "Phi2": Phi2}


def sim_source_var2(
    n_time: int,
    n_sources: int,
    var_params: dict,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""Simulate a VAR(2) source-time series.

    Iterates :math:`s_t = \\Phi_1 s_{t-1} + \\Phi_2 s_{t-2} + \\epsilon_t`
    with a 500-sample burn-in, then mean-centres each source.
    """
    if n_time < 1 or n_sources < 1:
        raise ValueError("n_time and n_sources must be positive integers.")
    Phi1 = np.asarray(var_params["Phi1"], dtype=float)
    Phi2 = np.asarray(var_params["Phi2"], dtype=float)
    if Phi1.shape != (n_sources, n_sources) or Phi2.shape != (n_sources, n_sources):
        raise ValueError("Phi1 and Phi2 must both be n_sources x n_sources matrices.")
    if not np.all(np.isfinite(Phi1)) or not np.all(np.isfinite(Phi2)):
        raise ValueError("Phi1 and Phi2 must contain only finite values.")
    if rng is None:
        rng = np.random.default_rng()

    burn_in = 500
    n_total = n_time + burn_in
    noise = rng.normal(size=(n_total, n_sources))
    S = np.zeros((n_total, n_sources))
    for t in range(2, n_total):
        val = Phi1 @ S[t - 1] + Phi2 @ S[t - 2] + noise[t]
        # Replace any infinite values with sign * 1e10, matching the R safety net.
        if not np.all(np.isfinite(val)):
            val = np.where(np.isfinite(val), val, np.sign(val) * 1e10)
        S[t] = val
    out = S[burn_in:burn_in + n_time]
    return out - out.mean(axis=0, keepdims=True)


def sim_source_task(
    n_time: int,
    n_sources: int,
    fs: float,
    tau_ms: float = 0.0,
    gamma: float = 1.0,
    active_idx: Optional[Sequence[int]] = None,
) -> np.ndarray:
    r"""Task source: time-warped Gabor wavelet at 20 Hz.

    The wavelet is centred at the trial midpoint, with optional latency
    shift ``tau_ms`` and time-scaling ``gamma`` (>1 stretches in time;
    <1 compresses). Active sources receive the same waveform; the rest
    are zero.

    Parameters
    ----------
    n_time, n_sources : int
    fs : float
        Sampling rate in Hz.
    tau_ms : float, default 0
        Latency shift in milliseconds.
    gamma : float, default 1.0
        Time-scaling factor.
    active_idx : sequence of int, optional
        1-based or 0-based source indices? **0-based** (Python). The R
        version uses 1-based ``1:3``; this Python version uses ``[0, 1, 2]``.
        ``None`` -> default ``[0, 1, 2]``.
    """
    if n_time < 1 or n_sources < 1:
        raise ValueError("n_time and n_sources must be positive integers.")
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a single positive finite number.")
    if not np.isfinite(tau_ms):
        raise ValueError("tau_ms must be a single finite number.")
    if not (np.isfinite(gamma) and gamma > 0):
        raise ValueError("gamma must be a single positive finite number.")

    if active_idx is None:
        active_idx = [0, 1, 2]
    active = np.asarray(active_idx, dtype=int)
    active = active[(active >= 0) & (active < n_sources)]
    active = np.unique(active)

    T_dur = n_time / fs
    t_vec = np.arange(n_time) / fs
    centre = T_dur / 2.0
    tau_s = tau_ms / 1000.0

    S = np.zeros((n_time, n_sources), dtype=float)
    if active.size == 0:
        return S

    t_prime = (t_vec - centre - tau_s) / gamma
    waveform = np.exp(-(t_prime ** 2) / (2.0 * 0.05 ** 2)) * np.cos(
        2.0 * np.pi * 20.0 * t_prime
    )
    for i in active:
        S[:, i] = waveform
    return S
