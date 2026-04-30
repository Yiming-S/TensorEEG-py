"""TensorEEG-py: covariance-aware EEG simulation and audit (Python port).

Top-level package re-exports every user-facing function so that the
Python API mirrors the R package's flat namespace::

    import tensoreeg as te

    sim = te.sim_eeg_master(n_trials=8, n_time=200, n_channels=6,
                            n_sources=4, seed=1, verbose=False)
    cov = te.tensor_to_cov(sim["data"])
    aug = te.augment_cov_geodesic_mixup(cov, sim["labels"], n_aug=3,
                                        seed=42)
    audit = te.audit_covariance_fidelity(cov, aug["cov"], aug["anchor"])

For the four-method audit story see
``examples/covariance_audit.py``.

For cross-language reproducibility against the Python protocol driver
of Shen & Degras (2026) see ``examples/manifest_replay.py``.
"""

from __future__ import annotations

__version__ = "0.10.0"

# --- Foundation: SPD utilities, covariance helpers ----------------------------
from .spd import (
    ensure_symmetric,
    spd_logm,
    spd_expm,
    spd_project,
    log_euclidean_distance,
    affine_invariant_distance,
    vech_log,
    unvech,
)
from ._utils import rbf_kernel, calc_ac_power, tensor_to_cov

# --- Augmentation family (E0 / G0 / G1 / G2 / A0) -----------------------------
from .augmentation import (
    augment_cov_amplitude_matched_euclidean,
    augment_cov_riemannian,
    augment_cov_empirical_tangent,
    augment_cov_geodesic_mixup,
    augment_cov_alignment_riemannian,
    augment_e0,
    augment_g0,
    augment_g1,
    augment_g2,
    augment_a0,
)

# --- Six-metric fidelity audit ------------------------------------------------
from .fidelity import (
    cov_logeuclidean_distance,
    cov_affine_invariant_distance,
    cov_eigenvalue_correlation,
    cov_trace_ratio,
    cov_condition_ratio,
    cov_anchor_perturbation_distance,
    audit_covariance_fidelity,
)

# --- Manifest read / replay ---------------------------------------------------
from .manifest import (
    MANIFEST_COLUMNS,
    read_calibration_manifest,
    replay_from_manifest,
)

# --- Simulation engine --------------------------------------------------------
from .simulation import (
    generate_geometry_mixing,
    generate_drift_rotations,
    setup_var2_system,
    sim_source_var2,
    sim_source_task,
    sim_artifacts,
    sim_eeg_master,
    sim_multirun_session,
)

# --- Bundled example data -----------------------------------------------------
from . import data as data  # noqa: F401  re-export submodule

__all__ = [
    "__version__",
    # spd
    "ensure_symmetric",
    "spd_logm",
    "spd_expm",
    "spd_project",
    "log_euclidean_distance",
    "affine_invariant_distance",
    "vech_log",
    "unvech",
    # utilities
    "rbf_kernel",
    "calc_ac_power",
    "tensor_to_cov",
    # augmentation (long names)
    "augment_cov_amplitude_matched_euclidean",
    "augment_cov_riemannian",
    "augment_cov_empirical_tangent",
    "augment_cov_geodesic_mixup",
    "augment_cov_alignment_riemannian",
    # augmentation (short aliases)
    "augment_e0",
    "augment_g0",
    "augment_g1",
    "augment_g2",
    "augment_a0",
    # fidelity
    "cov_logeuclidean_distance",
    "cov_affine_invariant_distance",
    "cov_eigenvalue_correlation",
    "cov_trace_ratio",
    "cov_condition_ratio",
    "cov_anchor_perturbation_distance",
    "audit_covariance_fidelity",
    # manifest
    "MANIFEST_COLUMNS",
    "read_calibration_manifest",
    "replay_from_manifest",
    # simulation
    "generate_geometry_mixing",
    "generate_drift_rotations",
    "setup_var2_system",
    "sim_source_var2",
    "sim_source_task",
    "sim_artifacts",
    "sim_eeg_master",
    "sim_multirun_session",
    # data
    "data",
]
