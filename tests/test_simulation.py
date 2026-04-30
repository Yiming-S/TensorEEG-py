"""Tests for the SimEEG simulation engine."""

from __future__ import annotations

import numpy as np
import pytest

import tensoreeg as te


# -------------------------------------------------------------------------
# Geometry / mixing


def test_generate_geometry_mixing_shapes():
    geo = te.generate_geometry_mixing(n_channels=16, n_sources=5)
    assert geo["coords_sens"].shape == (16, 3)
    assert geo["coords_src"].shape == (5, 3)
    assert geo["A_base"].shape == (16, 5)
    assert geo["L_sym"].shape == (16, 16)


def test_geometry_columns_unit_norm():
    geo = te.generate_geometry_mixing(n_channels=20, n_sources=4)
    norms = np.linalg.norm(geo["A_base"], axis=0)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_sources_inside_brain_volume():
    geo = te.generate_geometry_mixing(n_channels=10, n_sources=8)
    radii = np.linalg.norm(geo["coords_src"], axis=1)
    assert np.all(radii < 0.8 + 1e-9)


def test_sensors_on_upper_hemisphere():
    geo = te.generate_geometry_mixing(n_channels=32, n_sources=4)
    z = geo["coords_sens"][:, 2]
    assert np.all(z >= 0 - 1e-9)
    assert np.all(z <= 1 + 1e-9)


# -------------------------------------------------------------------------
# Drift rotations


def test_drift_rotations_orthogonal_ou():
    rots = te.generate_drift_rotations(
        n_sources=5, n_trials=10, alpha_ou=0.95,
        sigma_eps=0.05, process="ou",
        rng=np.random.default_rng(0),
    )
    assert len(rots) == 10
    for R in rots:
        # Orthogonal: R @ R.T == I.
        err = np.linalg.norm(R @ R.T - np.eye(5))
        assert err < 1e-8


def test_drift_rotations_orthogonal_fbm():
    rots = te.generate_drift_rotations(
        n_sources=4, n_trials=8, hurst=0.85,
        sigma_eps=0.05, process="fbm",
        rng=np.random.default_rng(0),
    )
    for R in rots:
        err = np.linalg.norm(R @ R.T - np.eye(4))
        assert err < 1e-8


def test_drift_rotations_n_sources_one_returns_identity():
    rots = te.generate_drift_rotations(n_sources=1, n_trials=4)
    for R in rots:
        assert R.shape == (1, 1)
        assert R[0, 0] == 1.0


# -------------------------------------------------------------------------
# VAR(2) sources


def test_var2_system_stable():
    rng = np.random.default_rng(0)
    sys = te.setup_var2_system(n_sources=5, fs=250.0,
                                target_freqs=[10, 12, 8, 14, 11], rng=rng)
    assert sys["Phi1"].shape == (5, 5)
    assert sys["Phi2"].shape == (5, 5)
    # Spectral radius of companion matrix < 1.
    top = np.hstack([sys["Phi1"], sys["Phi2"]])
    bot = np.hstack([np.eye(5), np.zeros((5, 5))])
    comp = np.vstack([top, bot])
    rho = float(np.max(np.abs(np.linalg.eigvals(comp))))
    assert rho < 1.0, f"VAR(2) is unstable, spectral radius {rho}"


def test_sim_source_var2_runs():
    rng = np.random.default_rng(0)
    sys = te.setup_var2_system(5, 250.0, [10, 12, 8, 14, 11], rng=rng)
    S = te.sim_source_var2(200, 5, sys, rng=rng)
    assert S.shape == (200, 5)
    assert np.all(np.isfinite(S))
    # Mean-centred.
    assert np.allclose(S.mean(axis=0), 0.0, atol=1e-10)


def test_sim_source_task_default_active():
    S = te.sim_source_task(n_time=200, n_sources=5, fs=250.0)
    assert S.shape == (200, 5)
    # Default active_idx = [0, 1, 2] -> first three columns active, others zero.
    assert np.allclose(S[:, 3], 0.0)
    assert np.allclose(S[:, 4], 0.0)
    assert not np.allclose(S[:, 0], 0.0)


# -------------------------------------------------------------------------
# Master engine


def test_sim_eeg_master_smoke():
    sim = te.sim_eeg_master(
        n_trials=4, n_time=200, n_channels=8, n_sources=4, fs=250.0,
        target_freqs=[10, 12, 11, 9], seed=42, verbose=False,
    )
    assert sim["data"].shape == (200, 8, 4)
    assert sim["labels"].shape == (4,)
    assert len(sim["audit"]) == 4
    assert sim["geometry"]["A_base"].shape == (8, 4)


def test_sim_eeg_master_reproducible():
    s1 = te.sim_eeg_master(
        n_trials=2, n_time=100, n_channels=4, n_sources=2, fs=250.0,
        target_freqs=[10, 12], seed=7, verbose=False,
    )
    s2 = te.sim_eeg_master(
        n_trials=2, n_time=100, n_channels=4, n_sources=2, fs=250.0,
        target_freqs=[10, 12], seed=7, verbose=False,
    )
    assert np.allclose(s1["data"], s2["data"])


def test_sim_eeg_master_realised_snr_close_to_target():
    """Closed-loop SNR should land within ~0.5 dB of the target."""
    sim = te.sim_eeg_master(
        n_trials=4, n_time=400, n_channels=8, n_sources=4, fs=250.0,
        snr_neural_db=5.0,
        target_freqs=[10, 12, 11, 9], seed=42, verbose=False,
    )
    snrs = np.array([row["Realized_SNR_Neural"] for row in sim["audit"]])
    snrs = snrs[np.isfinite(snrs)]
    assert np.all(np.abs(snrs - 5.0) < 1.0)


def test_sim_eeg_master_alternating_labels():
    sim = te.sim_eeg_master(
        n_trials=6, n_time=100, n_channels=4, n_sources=2, fs=250.0,
        target_freqs=[10, 12], seed=1, verbose=False,
    )
    assert list(sim["labels"]) == [0, 1, 0, 1, 0, 1]


# -------------------------------------------------------------------------
# Multi-run wrapper


def test_sim_multirun_session():
    multi = te.sim_multirun_session(
        n_runs=3, trials_per_run=3, gap_trials=2,
        n_time=100, n_channels=4, n_sources=2,
        target_freqs=[10, 12], seed=11, verbose=False,
    )
    assert len(multi["x"]) == 9
    assert list(multi["run"]) == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert multi["fs"] == 250.0
    assert len(multi["ch_names"]) == 4


# -------------------------------------------------------------------------
# tensor_to_cov utility


def test_tensor_to_cov_spd_output():
    sim = te.sim_eeg_master(
        n_trials=4, n_time=200, n_channels=6, n_sources=3, fs=250.0,
        target_freqs=[10, 12, 11], seed=0, verbose=False,
    )
    cov = te.tensor_to_cov(sim["data"], ridge=1e-6, centre=True)
    assert cov.shape == (4, 6, 6)
    for k in range(cov.shape[0]):
        e = np.linalg.eigvalsh(cov[k])
        assert np.all(e > 0), f"trial {k} not SPD"
