"""Cross-language reproducibility demo: read manifest CSV, replay each method.

Reads the bundled ``example_manifest.csv``, replays the synthetic stack
for every method code (R0 / E0 / G0 / G1 / G2), and verifies byte-
equivalent reproducibility against a direct seeded augmentation call.

Mirrors the ``manifest-replay`` vignette of the R package.

Run::

    python examples/manifest_replay.py
"""

from __future__ import annotations

import numpy as np

import tensoreeg as te


def main() -> None:
    anchors = te.data.load_example_anchors()
    labels = te.data.load_example_labels()
    manifest_path = te.data.example_manifest_path()
    print(f"Manifest path: {manifest_path}")

    manifest = te.read_calibration_manifest(manifest_path)
    print(f"Loaded {len(manifest)} manifest rows; "
          f"{len({r['method_code'] for r in manifest})} method codes.")

    # 1. Replay every method code on the same source covariances.
    print("\nReplay sizes:")
    for method in ("R0", "E0", "G0", "G1", "G2"):
        group = [r for r in manifest if r["method_code"] == method]
        res = te.replay_from_manifest(group, anchors, labels,
                                       ratio_per_anchor=3,
                                       g0_sigma=0.15, g1_sigma=0.15,
                                       g2_beta_alpha=1.0)
        print(f"  {method}: {res['cov'].shape}")

    # 2. Byte-equivalence check against a direct seeded G2 call.
    g2_group = [r for r in manifest if r["method_code"] == "G2"]
    g2_seed = g2_group[0]["method_seed"]
    replayed = te.replay_from_manifest(g2_group, anchors, labels,
                                        ratio_per_anchor=3,
                                        g2_beta_alpha=1.0)
    direct = te.augment_g2(anchors, labels, n_aug=3, beta_alpha=1.0,
                            seed=g2_seed)
    max_diff = float(np.max(np.abs(replayed["cov"] - direct["cov"])))
    print(f"\nG2 manifest replay vs direct seeded call: "
          f"max |diff| = {max_diff:.2e}")
    assert max_diff == 0.0, "G2 replay drifted from direct call"

    # 3. Audit every replayed stack against the same anchor reference.
    print("\nFidelity audit (LE / AI / anchor distance):")
    print(f"  {'method':<5}{'LE':>10}{'AI':>10}{'anchor':>10}")
    for method in ("E0", "G0", "G1", "G2"):
        group = [r for r in manifest if r["method_code"] == method]
        res = te.replay_from_manifest(group, anchors, labels,
                                       ratio_per_anchor=3,
                                       g0_sigma=0.15, g1_sigma=0.15,
                                       g2_beta_alpha=1.0)
        aud = te.audit_covariance_fidelity(anchors, res["cov"], res["anchor"])
        print(
            f"  {method:<5}"
            f"{aud['log_euclidean_to_reference']:>10.4f}"
            f"{aud['affine_invariant_to_reference']:>10.4f}"
            f"{aud['anchor_perturbation_distance']:>10.4f}"
        )

    print("\nThe Python protocol driver in the manuscript writes the "
          "same manifest format; this script is the audit trail you "
          "give a reviewer.\n")


if __name__ == "__main__":
    main()
