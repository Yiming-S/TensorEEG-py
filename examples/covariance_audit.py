"""End-to-end covariance audit on the bundled example anchors.

Reproduces the four-method fidelity ranking E0 > G0 > G1 > G2 on the
8-anchor toy stack shipped with the package. Mirrors the ``covariance-
audit`` vignette of the R package.

Run::

    python examples/covariance_audit.py
"""

from __future__ import annotations

import numpy as np

import tensoreeg as te


def main() -> None:
    anchors = te.data.load_example_anchors()
    labels = te.data.load_example_labels()
    print(f"Loaded {anchors.shape[0]} anchors of shape {anchors.shape[1:]}, "
          f"labels = {labels.tolist()}")

    n_aug = 3
    g0_sigma = 0.15

    aug_e0 = te.augment_e0(anchors, n_aug=n_aug, g0_sigma=g0_sigma,
                           labels=labels, seed=1001)
    aug_g0 = te.augment_g0(anchors, n_aug=n_aug, sigma=g0_sigma,
                           labels=labels, seed=1002)
    aug_g1 = te.augment_g1(anchors, labels, n_aug=n_aug, sigma=g0_sigma,
                           seed=1003)
    aug_g2 = te.augment_g2(anchors, labels, n_aug=n_aug, beta_alpha=1.0,
                           seed=1004)

    print(f"\nE0 amplitude-match rho = {aug_e0['diagnostic']['rho']:.4f} "
          f"(success={aug_e0['diagnostic']['success']})")

    methods = {"E0": aug_e0, "G0": aug_g0, "G1": aug_g1, "G2": aug_g2}
    print("\nSix-metric fidelity audit (lower distance = closer to class mean):")
    header = (
        f"{'method':<5}"
        f"{'LE':>10}{'AI':>10}{'trace':>10}{'cond':>10}{'anchor':>10}"
        f"{'LE/dn':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, aug in methods.items():
        res = te.audit_covariance_fidelity(anchors, aug["cov"], aug["anchor"])
        print(
            f"{name:<5}"
            f"{res['log_euclidean_to_reference']:>10.4f}"
            f"{res['affine_invariant_to_reference']:>10.4f}"
            f"{res['trace_ratio']:>10.4f}"
            f"{res['condition_number_ratio']:>10.4f}"
            f"{res['anchor_perturbation_distance']:>10.4f}"
            f"{res['normalized']['log_euclidean']:>10.4f}"
        )

    print(
        "\nLE-distance ranking should be E0 > (G0, G1) > G2 -- with G2 the "
        "most fidelity-preserving variant.\n"
    )


if __name__ == "__main__":
    main()
