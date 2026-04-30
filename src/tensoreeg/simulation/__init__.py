"""SimEEG: physics-constrained synthetic EEG generator.

This sub-package mirrors the R simulation engine. Each module is the
Python sibling of its R counterpart:

================  ============================================
Python module     R source
================  ============================================
``physics``       ``R/physics.R``  + ``R/utils.R::rbf_kernel``
``manifold``      ``R/manifold.R``
``sources``       ``R/sources.R``
``artifacts``     ``R/artifacts.R``
``engine``        ``R/engine.R``  (``sim_eeg_master``)
``wrappers``      ``R/wrappers.R``  (``sim_multirun_session``)
================  ============================================

Note on cross-language reproducibility: even at identical seeds, the R
and Python simulators return numerically different tensors because the
two languages use different RNGs. The simulator is reproducible **within
Python** via the ``seed`` argument. Algorithm-level equivalence is
preserved (volume conduction physics, VAR(2) dynamics, OU/fBm drift,
artifact injection, closed-loop SNR).
"""

from .physics import generate_geometry_mixing
from .manifold import generate_drift_rotations
from .sources import setup_var2_system, sim_source_var2, sim_source_task
from .artifacts import sim_artifacts
from .engine import sim_eeg_master
from .wrappers import sim_multirun_session

__all__ = [
    "generate_geometry_mixing",
    "generate_drift_rotations",
    "setup_var2_system",
    "sim_source_var2",
    "sim_source_task",
    "sim_artifacts",
    "sim_eeg_master",
    "sim_multirun_session",
]
