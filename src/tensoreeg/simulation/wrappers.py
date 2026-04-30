"""Multi-run session wrapper (cross-run drift / leave-one-run-out CV)."""

from __future__ import annotations

from typing import List

import numpy as np

from .engine import sim_eeg_master

__all__ = ["sim_multirun_session"]


def sim_multirun_session(
    n_runs: int = 3,
    trials_per_run: int = 20,
    gap_trials: int = 15,
    verbose: bool = True,
    **sim_kwargs,
) -> dict:
    """Simulate a multi-run session by extracting non-contiguous trials.

    Internally runs **one** ``sim_eeg_master`` call long enough to cover
    ``n_runs * trials_per_run + (n_runs - 1) * gap_trials`` trials, then
    extracts each run while discarding the gap trials between them. The
    underlying manifold drift continues to evolve during the gaps, so
    the output simulates a session with naturally-evolving cross-run
    distributional shift.

    Parameters
    ----------
    n_runs, trials_per_run, gap_trials : int
    verbose : bool
        Forwarded to :func:`sim_eeg_master`.
    **sim_kwargs
        All other keyword arguments forwarded to
        :func:`sim_eeg_master` (e.g., ``n_time``, ``n_channels``,
        ``target_freqs``).

    Returns
    -------
    dict with keys
        ``x``       : list of (n_time, n_channels) per-trial arrays,
        ``y``       : (total_trials,) int label ndarray,
        ``run``     : (total_trials,) int run-id ndarray (1-based),
        ``fs``      : float,
        ``ch_names`` : list of str.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be a positive integer.")
    if trials_per_run < 1:
        raise ValueError("trials_per_run must be a positive integer.")
    if gap_trials < 0:
        raise ValueError("gap_trials must be a non-negative integer.")
    if "verbose" in sim_kwargs:
        raise ValueError(
            "Pass verbose via sim_multirun_session(verbose=...), not via **sim_kwargs."
        )

    grand_total = n_runs * trials_per_run + (n_runs - 1) * gap_trials
    if verbose:
        print("--- Simulating Multi-Run Session ---")
        print(
            f"Config: {n_runs} Runs, {trials_per_run} Trials/Run, "
            f"{gap_trials} Gap Trials (Drift Injection)"
        )

    giant = sim_eeg_master(n_trials=grand_total, verbose=verbose, **sim_kwargs)

    x_list: List[np.ndarray] = []
    y_vec: List[int] = []
    run_id: List[int] = []

    cursor = 0
    for r in range(n_runs):
        end = cursor + trials_per_run
        if verbose:
            print(f"Extracting Run {r + 1}: Indices [{cursor + 1} - {end}]")
        for k in range(cursor, end):
            x_list.append(giant["data"][:, :, k])
        y_vec.extend(int(v) for v in giant["labels"][cursor:end])
        run_id.extend([r + 1] * trials_per_run)
        cursor = end + gap_trials

    return {
        "x": x_list,
        "y": np.asarray(y_vec, dtype=int),
        "run": np.asarray(run_id, dtype=int),
        "fs": giant["params"]["fs"],
        "ch_names": [f"Ch{i + 1}" for i in range(giant["data"].shape[1])],
    }
