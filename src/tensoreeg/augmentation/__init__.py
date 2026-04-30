"""Covariance-aware augmentation family.

Each routine maps a stack of SPD anchor covariances to a stack of
synthetic SPD covariances and returns a structured result containing

- ``cov``: ``(n_anchor * n_aug, p, p)`` SPD array,
- ``anchor``: ``(n_anchor * n_aug,)`` index into the original anchor
  stack,
- ``replicate``: per-anchor replicate id,
- ``labels``: anchor labels propagated through to synthetics, or
  ``None`` if no labels were supplied,
- ``params``: a dict of resolved parameters.

E0 / G0 / G1 / G2 are non-transductive (they need only labelled source
covariances and a seed). A0 is transductive: it consults the unlabeled
target-session covariance stack at fit time and is reported as a
transfer-learning upper bound rather than a non-inferiority candidate.
"""

from .e0 import augment_cov_amplitude_matched_euclidean
from .g0 import augment_cov_riemannian
from .g1 import augment_cov_empirical_tangent
from .g2 import augment_cov_geodesic_mixup
from .a0 import augment_cov_alignment_riemannian

# Short aliases that mirror the method codes used in Shen & Degras (2026).
augment_e0 = augment_cov_amplitude_matched_euclidean
augment_g0 = augment_cov_riemannian
augment_g1 = augment_cov_empirical_tangent
augment_g2 = augment_cov_geodesic_mixup
augment_a0 = augment_cov_alignment_riemannian

__all__ = [
    "augment_cov_amplitude_matched_euclidean",
    "augment_cov_riemannian",
    "augment_cov_empirical_tangent",
    "augment_cov_geodesic_mixup",
    "augment_cov_alignment_riemannian",
    "augment_e0",
    "augment_g0",
    "augment_g1",
    "augment_g2",
    "augment_a0",
]
