"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def example_anchors() -> np.ndarray:
    """The bundled example anchor stack (8, 6, 6)."""
    import tensoreeg as te
    return te.data.load_example_anchors()


@pytest.fixture(scope="session")
def example_labels() -> np.ndarray:
    """The bundled example label vector (8,)."""
    import tensoreeg as te
    return te.data.load_example_labels()


@pytest.fixture(scope="session")
def example_manifest_path() -> str:
    """Path to the bundled example manifest CSV."""
    import tensoreeg as te
    return te.data.example_manifest_path()


def _build_synthetic_anchors(seed: int = 7, n_per_class: int = 4,
                             p: int = 6, n_time: int = 400) -> tuple:
    """Helper: deterministic two-class SPD anchor stack.

    Used by integration tests that want a different anchor stack from
    the bundled fixture (e.g., to verify size invariance).
    """
    rng = np.random.default_rng(seed)

    def build(base_freq: float, sub_seed: int) -> np.ndarray:
        local = np.random.default_rng(sub_seed)
        fs = 250
        t = np.arange(n_time) / fs
        X = np.zeros((n_time, p))
        for k in range(p):
            f_k = base_freq + local.uniform(-0.5, 0.5)
            ph = local.uniform(0, 2 * np.pi)
            X[:, k] = np.cos(2 * np.pi * f_k * t + ph) + 0.3 * local.standard_normal(n_time)
        return np.cov(X, rowvar=False, ddof=1) + 1e-3 * np.eye(p)

    anchors = np.empty((2 * n_per_class, p, p))
    for i in range(n_per_class):
        anchors[i] = build(10.0, seed * 100 + i)
        anchors[n_per_class + i] = build(6.0, seed * 200 + i)
    labels = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)
    return anchors, labels


@pytest.fixture(scope="session")
def synthetic_anchors():
    return _build_synthetic_anchors()
