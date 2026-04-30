"""Optional matplotlib visualisation helpers.

These functions are siblings of the R ``plot_*`` and
``validate_sim_eeg`` helpers in the R reference implementation. They
require matplotlib and signal at runtime, so the import is deferred to
function-call time and the package itself does **not** declare
matplotlib as a hard dependency.

Install matplotlib via the ``[plot]`` extra::

    pip install "tensoreeg[plot]"
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Visualisation helpers require matplotlib. Install with "
            "'pip install \"tensoreeg[plot]\"'."
        ) from exc
    return plt


def plot_topomap(values: np.ndarray, coords_sens: np.ndarray, ax=None,
                 title: Optional[str] = None):
    """Scatter-plot a per-channel value over a 2D sensor projection.

    A simple matplotlib scatter on the (x, y) sensor positions, with
    point colour driven by ``values``. Suitable for quickly inspecting
    spatial patterns; not a substitute for MNE-style topographic
    interpolation.
    """
    plt = _require_matplotlib()
    values = np.asarray(values, dtype=float)
    coords = np.asarray(coords_sens, dtype=float)
    if values.shape[0] != coords.shape[0]:
        raise ValueError("values and coords_sens must have matching length.")
    own_axis = ax is None
    if own_axis:
        fig, ax = plt.subplots(figsize=(4, 4))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values,
                    s=80, edgecolors="black", cmap="RdBu_r")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)
    if own_axis:
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
    return ax


def plot_run_drift(rotations, ax=None, title: Optional[str] = None):
    """Plot the angle path of a sequence of SO(n) rotation matrices.

    ``rotations`` is the list returned by
    :func:`tensoreeg.simulation.generate_drift_rotations`.

    The angle is recovered from the matrix logarithm via
    :math:`\\theta_k \\approx \\|\\log R_k\\|_F / \\sqrt{2}` (since
    :math:`\\log R_k = \\theta_k \\Omega` with
    :math:`\\|\\Omega\\|_F = 1` after normalisation in the generator).
    """
    plt = _require_matplotlib()
    from scipy.linalg import logm
    angles = np.array([float(np.linalg.norm(logm(R).real) / np.sqrt(2.0))
                       for R in rotations])
    own_axis = ax is None
    if own_axis:
        fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(angles.size) + 1, angles, marker="o", linewidth=1.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel(r"$\theta_k$ (rad)")
    if title is not None:
        ax.set_title(title)
    if own_axis:
        plt.tight_layout()
    return ax


def plot_sim_dashboard(sim_result: dict, channel: int = 0, fmax: float = 60.0,
                       savepath: Optional[str] = None):
    """A 2 x 2 validation dashboard analogous to ``validate_sim_eeg`` in R.

    Panels:
      1. Class contrast trace (1 Hz HP, single channel),
      2. Power spectral density of a class-1 trial (with 20 Hz target),
      3. Spatial covariance heatmap of a class-1 trial,
      4. Realised SNR vs trial.
    """
    plt = _require_matplotlib()
    from scipy.signal import butter, filtfilt, welch

    data = sim_result["data"]
    fs = sim_result["params"]["fs"]
    target_freqs = sim_result["params"]["target_freqs"]
    labels = sim_result["labels"]
    audit = sim_result["audit"]
    n_time, n_channels, n_trials = data.shape
    if not (0 <= channel < n_channels):
        raise ValueError(f"channel must be in [0, {n_channels})")

    idx0 = next((i for i, c in enumerate(labels) if c == 0), 0)
    idx1 = next((i for i, c in enumerate(labels) if c == 1), min(1, n_trials - 1))

    b, a = butter(4, 1.0 / (fs / 2.0), btype="high")
    pad = min(3 * max(len(a), len(b)), n_time - 1)
    x_c0 = filtfilt(b, a, data[:, channel, idx0], padlen=pad)
    x_c1 = filtfilt(b, a, data[:, channel, idx1], padlen=pad)
    t = np.arange(n_time) / fs

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(t, x_c1, color="red", label="Class 1 (Task High)")
    axes[0, 0].plot(t, x_c0, color="blue", label="Class 0 (Task Low)")
    axes[0, 0].set_title(f"Class Contrast (Ch{channel})")
    axes[0, 0].set_xlabel("s")
    axes[0, 0].set_ylabel("uV")
    axes[0, 0].legend(fontsize=8)

    f, p = welch(x_c1, fs=fs, nperseg=min(256, n_time))
    axes[0, 1].plot(f, 10 * np.log10(np.maximum(p, 1e-12)))
    if target_freqs is not None:
        for tf in np.asarray(target_freqs):
            axes[0, 1].axvline(tf, color="gray", linestyle=":", alpha=0.5)
    axes[0, 1].axvline(20.0, color="red", linestyle="--", label="Task (20 Hz)")
    axes[0, 1].set_xlim(0, min(fmax, fs / 2))
    axes[0, 1].set_xlabel("Hz")
    axes[0, 1].set_ylabel("dB")
    axes[0, 1].set_title("PSD (Class 1)")
    axes[0, 1].legend(fontsize=8)

    X_hp = np.column_stack([
        filtfilt(b, a, data[:, k, idx1], padlen=pad) for k in range(n_channels)
    ])
    cov_mat = np.cov(X_hp, rowvar=False)
    axes[1, 0].imshow(cov_mat, aspect="auto", cmap="viridis")
    axes[1, 0].set_title("Spatial Cov (1 Hz HP)")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    snr = np.array([row.get("Realized_SNR_Neural", np.nan) for row in audit])
    axes[1, 1].plot(np.arange(len(snr)) + 1, snr, marker="o")
    axes[1, 1].axhline(np.nanmedian(snr), color="green", linestyle="--",
                       label="Median")
    axes[1, 1].set_xlabel("Trial")
    axes[1, 1].set_ylabel("dB")
    axes[1, 1].set_title("SNR Audit")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
    return fig


__all__ = [
    "plot_topomap",
    "plot_run_drift",
    "plot_sim_dashboard",
]
