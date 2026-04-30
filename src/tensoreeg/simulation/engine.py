"""Master controller: ``sim_eeg_master`` (orchestrates the full pipeline)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .._utils import calc_ac_power
from .physics import generate_geometry_mixing
from .manifold import generate_drift_rotations
from .sources import setup_var2_system, sim_source_var2, sim_source_task
from .artifacts import sim_artifacts

__all__ = ["sim_eeg_master"]


def sim_eeg_master(
    n_trials: int = 20,
    n_time: int = 500,
    n_channels: int = 64,
    n_sources: int = 10,
    fs: float = 250.0,
    snr_neural_db: float = 5.0,
    snr_artifact_db: float = 0.0,
    drift_power_ratio: float = 0.5,
    target_freqs: Optional[Sequence[float]] = None,
    class_labels: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    r"""Physics-constrained 3rd-order EEG tensor simulator.

    Pipeline:

    1. Generate sensor geometry and Laplacian-smoothed mixing
       (:func:`generate_geometry_mixing`).
    2. Generate trial-wise rotation matrices on :math:`SO(n)`
       (:func:`generate_drift_rotations`).
    3. For each trial, generate task ERP (Gabor) + VAR(2) background,
       add EOG / EMG / drift artifacts, and apply closed-loop SNR
       control measured against ``calc_ac_power`` (HP > 0.1 Hz).

    Class-dependent logic:

    - Class 1: 1.5x task amplitude, 0.8x background amplitude (ERD-like).
    - Class 0: baseline 1.0x / 1.0x.

    Parameters
    ----------
    n_trials, n_time, n_channels, n_sources : int
        Tensor shape.
    fs : float
        Sampling rate in Hz.
    snr_neural_db, snr_artifact_db : float
        Target SNRs in dB. ``snr_neural_db`` is task vs. background;
        ``snr_artifact_db`` is total neural vs. fast artifacts.
    drift_power_ratio : float
        Drift AC power as a fraction of neural AC power.
    target_freqs : sequence of float, optional
        Per-source peak frequencies in Hz. Length must equal
        ``n_sources``. ``None`` -> uniform random in 8-20 Hz (or
        clipped to fs/2).
    class_labels : sequence of int, optional
        Per-trial labels in ``{0, 1}``. ``None`` -> alternating
        ``[0, 1, 0, 1, ...]``.
    seed : int, optional
        RNG seed (reproducible within Python; not byte-equivalent to R).
    verbose : bool, default True
        If True, print progress messages every 10 trials.

    Returns
    -------
    dict with keys
        ``data``     : (n_time, n_channels, n_trials) ndarray,
        ``geometry`` : dict from :func:`generate_geometry_mixing`,
        ``audit``    : list of dicts (per-trial realised SNR),
        ``labels``   : (n_trials,) int ndarray,
        ``params``   : dict with ``fs``, ``target_freqs``, ``seed``.
    """
    for name, val in (
        ("n_trials", n_trials),
        ("n_time", n_time),
        ("n_channels", n_channels),
        ("n_sources", n_sources),
    ):
        if val < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a single positive finite number.")
    if not np.isfinite(snr_neural_db):
        raise ValueError("snr_neural_db must be a single finite number.")
    if not np.isfinite(snr_artifact_db):
        raise ValueError("snr_artifact_db must be a single finite number.")
    if not (np.isfinite(drift_power_ratio) and drift_power_ratio >= 0):
        raise ValueError("drift_power_ratio must be a non-negative finite number.")
    n_trials = int(n_trials)
    n_time = int(n_time)
    n_channels = int(n_channels)
    n_sources = int(n_sources)

    rng = np.random.default_rng(seed)

    nyq = fs / 2.0
    if target_freqs is None:
        lower = min(8.0, nyq * 0.25)
        upper = min(20.0, nyq * 0.9)
        if lower <= 0 or upper <= 0 or lower >= upper:
            raise ValueError(
                "fs is too low for default target_freqs; please provide "
                "target_freqs in (0, fs/2)."
            )
        target_freqs_arr = rng.uniform(lower, upper, size=n_sources)
    else:
        target_freqs_arr = np.asarray(target_freqs, dtype=float)
        if target_freqs_arr.shape != (n_sources,):
            raise ValueError("target_freqs must have length n_sources.")
        if np.any(~np.isfinite(target_freqs_arr)) or np.any(
            (target_freqs_arr <= 0) | (target_freqs_arr >= nyq)
        ):
            raise ValueError("target_freqs must contain finite values in (0, fs/2).")

    if class_labels is None:
        labels_arr = np.tile([0, 1], (n_trials // 2) + 1)[:n_trials].astype(int)
    else:
        labels_arr = np.asarray(class_labels, dtype=int)
        if labels_arr.shape != (n_trials,):
            raise ValueError("class_labels must have length n_trials.")
        if not np.all(np.isin(labels_arr, [0, 1])):
            raise ValueError("class_labels must only contain 0 and 1.")

    geo = generate_geometry_mixing(n_channels, n_sources, rng=rng)
    A_base = geo["A_base"]
    coords = geo["coords_sens"]
    rotations = generate_drift_rotations(n_sources, n_trials, rng=rng)
    var_system = setup_var2_system(n_sources, fs, target_freqs_arr, rng=rng)

    X_tensor = np.zeros((n_time, n_channels, n_trials), dtype=float)
    audit: List[dict] = []

    if verbose:
        print(f"Simulating {n_trials} trials...")

    for k in range(n_trials):
        if verbose and ((k + 1) % 10 == 0 or k == n_trials - 1):
            print(f"  trial {k + 1}/{n_trials}")
        A_k = A_base @ rotations[k]
        cls = int(labels_arr[k])
        task_amp = 1.5 if cls == 1 else 1.0
        bg_amp = 0.8 if cls == 1 else 1.0

        tau = float(rng.normal(0.0, 20.0))
        gamma = float(rng.lognormal(0.0, 0.1))
        S_task = sim_source_task(n_time, n_sources, fs, tau_ms=tau, gamma=gamma) * task_amp
        S_bg = sim_source_var2(n_time, n_sources, var_system, rng=rng) * bg_amp

        X_task_pure = S_task @ A_k.T
        X_bg_pure = S_bg @ A_k.T

        P_task = calc_ac_power(X_task_pure, fs)
        P_bg = calc_ac_power(X_bg_pure, fs)
        if P_bg > 1e-9:
            g_bg = float(np.sqrt(P_task / (P_bg * 10 ** (snr_neural_db / 10.0))))
        else:
            g_bg = 0.0
        X_neural = X_task_pure + g_bg * X_bg_pure
        P_neural = calc_ac_power(X_neural, fs)

        N_fast, N_drift_raw = sim_artifacts(n_time, n_channels, fs, coords, rng=rng)
        P_fast = calc_ac_power(N_fast, fs)
        if P_fast > 1e-9:
            g_art = float(np.sqrt(P_neural / (P_fast * 10 ** (snr_artifact_db / 10.0))))
        else:
            g_art = 0.0
        P_drift_ac = calc_ac_power(N_drift_raw, fs)
        if P_drift_ac > 1e-9:
            g_drift = float(np.sqrt((P_neural * drift_power_ratio) / P_drift_ac))
        else:
            g_drift = 0.0

        X_final = X_neural + g_art * N_fast + g_drift * N_drift_raw
        X_final = np.where(np.isnan(X_final), 0.0, X_final)
        X_tensor[:, :, k] = X_final

        snr_denom = (g_bg ** 2) * P_bg
        if P_task > 1e-12 and snr_denom > 1e-12:
            realised_snr = float(10.0 * np.log10(P_task / snr_denom))
        else:
            realised_snr = float("nan")
        audit.append({
            "trial": k + 1,
            "class": cls,
            "Realized_SNR_Neural": realised_snr,
        })

    return {
        "data": X_tensor,
        "geometry": geo,
        "audit": audit,
        "labels": labels_arr,
        "params": {
            "fs": float(fs),
            "target_freqs": target_freqs_arr,
            "seed": seed,
        },
    }
