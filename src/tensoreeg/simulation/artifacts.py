"""Synthetic physiological artifacts: EOG, EMG, slow drift."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import squareform, pdist

__all__ = ["sim_artifacts"]


def sim_artifacts(
    n_time: int,
    n_channels: int,
    fs: float,
    coords_sens: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate fast (EOG + EMG) and slow (drift) artifact components.

    Returns
    -------
    N_fast : (n_time, n_channels) ndarray
        EOG (~15/min Hanning blinks projected from a frontal source) +
        EMG (~5/min ~100 ms 20+ Hz bursts on 3-electrode clusters).
    N_drift : (n_time, n_channels) ndarray
        Random walk + linear trend, normalised so the maximum absolute
        amplitude is 100.
    """
    if n_time < 1 or n_channels < 1:
        raise ValueError("n_time and n_channels must be positive integers.")
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a single positive finite number.")
    coords = np.asarray(coords_sens, dtype=float)
    if coords.shape[0] != n_channels or coords.shape[1] < 3:
        raise ValueError(
            f"coords_sens must have shape ({n_channels}, >=3); "
            f"got {coords.shape}"
        )
    if rng is None:
        rng = np.random.default_rng()
    n_time = int(n_time)
    n_channels = int(n_channels)

    N_fast = np.zeros((n_time, n_channels), dtype=float)

    # 1. EOG (eye-blink) component.
    fpz = np.array([0.9, 0.0, 0.4])
    dists_fpz = np.linalg.norm(coords - fpz, axis=1)
    proj_eog = np.exp(-(dists_fpz ** 2) / 0.1)

    n_blink = max(1, int(round(0.3 * fs)))
    t_blink = np.linspace(-np.pi, np.pi, n_blink)
    blink_shape = (1.0 + np.cos(t_blink)) / 2.0

    n_events_eog = int(rng.poisson((n_time / fs) * (15.0 / 60.0)))
    if n_events_eog > 0 and n_time >= n_blink:
        n_starts = n_time - n_blink + 1
        if n_starts > 0:
            n_pick = min(n_events_eog, n_starts)
            onsets = rng.choice(n_starts, size=n_pick, replace=False)
            ts_eog = np.zeros(n_time, dtype=float)
            for t_start in onsets:
                ts_eog[t_start : t_start + n_blink] += blink_shape
            N_fast = N_fast + np.outer(ts_eog, proj_eog) * 50.0

    # 2. EMG bursts.
    dist_SS = squareform(pdist(coords))
    n_bursts = int(rng.poisson((n_time / fs) * (5.0 / 60.0)))
    if n_bursts > 0:
        b, a = butter(4, 20.0 / (fs / 2.0), btype="high")
        for _ in range(n_bursts):
            dur_samps = max(1, int(round(0.1 * fs)))
            if n_time < dur_samps:
                continue
            centre_ch = int(rng.integers(0, n_channels))
            cluster_size = min(3, n_channels)
            cluster = np.argsort(dist_SS[centre_ch])[:cluster_size]
            n_starts = n_time - dur_samps + 1
            start_t = int(rng.integers(0, n_starts))
            buffer_len = max(int(round(fs)), dur_samps + 8)
            noise_buffer = rng.normal(size=(buffer_len, cluster_size))
            padlen = min(3 * max(len(a), len(b)), buffer_len - 1)
            if padlen < 0:
                continue
            noise_hp = np.column_stack([
                filtfilt(b, a, noise_buffer[:, k], padlen=padlen)
                for k in range(cluster_size)
            ])
            crop_start = (buffer_len - dur_samps) // 2
            burst = noise_hp[crop_start : crop_start + dur_samps]
            N_fast[start_t : start_t + dur_samps, cluster] += burst * 20.0

    # 3. Slow drift: random walk + linear trend.
    drift = np.cumsum(rng.normal(size=(n_time, n_channels)), axis=0)
    trend = np.linspace(0.0, 1.0, n_time)
    trend_scale = rng.normal(scale=50.0, size=n_channels)
    drift = drift + np.outer(trend, trend_scale)
    max_exc = float(np.max(np.abs(drift)))
    if max_exc > 0:
        drift = drift / max_exc * 100.0
    return N_fast, drift
