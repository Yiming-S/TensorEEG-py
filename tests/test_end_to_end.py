"""End-to-end integration tests.

Verifies that the documented user-facing pipelines compose correctly:

(1) ``sim_eeg_master() -> tensor_to_cov() -> 4 augmentations ->
    audit_covariance_fidelity()`` runs without error and produces a sane
    multi-metric profile.

(2) The shipped example assets drive the same pipeline as the
    ``examples/`` scripts and the README quick-start.

(3) Manifest-driven replay is byte-equivalent to a direct seeded call
    to the underlying augmentation routine.
"""

from __future__ import annotations

import numpy as np

import tensoreeg as te


def test_full_pipeline_sim_to_audit():
    sim = te.sim_eeg_master(
        n_trials=8, n_time=200, n_channels=6, n_sources=4, fs=250.0,
        target_freqs=[10, 12, 11, 9], seed=42, verbose=False,
    )
    cov = te.tensor_to_cov(sim["data"])
    labels = sim["labels"]
    n_aug = 3
    g0 = te.augment_g0(cov, n_aug=n_aug, sigma=0.15, labels=labels, seed=1)
    e0 = te.augment_e0(cov, n_aug=n_aug, g0_sigma=0.15, labels=labels, seed=2)
    g1 = te.augment_g1(cov, labels, n_aug=n_aug, sigma=0.15, seed=3)
    g2 = te.augment_g2(cov, labels, n_aug=n_aug, beta_alpha=1.0, seed=4)

    for aug, name in [(e0, "E0"), (g0, "G0"), (g1, "G1"), (g2, "G2")]:
        assert aug["cov"].shape == (cov.shape[0] * n_aug, cov.shape[1], cov.shape[2])
        # SPD invariant.
        for C in aug["cov"]:
            assert np.all(np.linalg.eigvalsh(0.5 * (C + C.T)) > 0), name

    aud_e0 = te.audit_covariance_fidelity(cov, e0["cov"], e0["anchor"])
    aud_g2 = te.audit_covariance_fidelity(cov, g2["cov"], g2["anchor"])
    # E0 sits off-manifold by construction; LE distance > G2.
    assert (
        aud_e0["log_euclidean_to_reference"]
        > aud_g2["log_euclidean_to_reference"]
    )


def test_shipped_assets_drive_full_pipeline(
    example_anchors, example_labels, example_manifest_path
):
    """Same pipeline as the README quick-start / vignette."""
    manifest = te.read_calibration_manifest(example_manifest_path)
    assert {r["method_code"] for r in manifest} >= {"R0", "E0", "G0", "G1", "G2"}

    for method in ("E0", "G0", "G1", "G2"):
        group = [r for r in manifest if r["method_code"] == method]
        res = te.replay_from_manifest(
            group, example_anchors, example_labels, ratio_per_anchor=3
        )
        assert res["cov"].shape == (
            example_anchors.shape[0] * 3,
            example_anchors.shape[1],
            example_anchors.shape[2],
        )

    # R0 is empty by design.
    r0_group = [r for r in manifest if r["method_code"] == "R0"]
    res = te.replay_from_manifest(r0_group, example_anchors, example_labels)
    assert res["cov"].shape[0] == 0


def test_manifest_replay_byte_equivalent_for_each_method(
    example_anchors, example_labels, example_manifest_path
):
    """All non-A0 methods should replay byte-equivalently to a direct seeded call."""
    manifest = te.read_calibration_manifest(example_manifest_path)
    for method in ("E0", "G0", "G1", "G2"):
        group = [r for r in manifest if r["method_code"] == method]
        seed = group[0]["method_seed"]
        replayed = te.replay_from_manifest(
            group, example_anchors, example_labels,
            ratio_per_anchor=3, g0_sigma=0.15, g1_sigma=0.15, g2_beta_alpha=1.0,
        )
        if method == "E0":
            direct = te.augment_e0(
                example_anchors, n_aug=3, g0_sigma=0.15,
                labels=example_labels, seed=seed,
            )
        elif method == "G0":
            direct = te.augment_g0(
                example_anchors, n_aug=3, sigma=0.15,
                labels=example_labels, seed=seed,
            )
        elif method == "G1":
            direct = te.augment_g1(
                example_anchors, example_labels,
                n_aug=3, sigma=0.15, seed=seed,
            )
        else:  # G2
            direct = te.augment_g2(
                example_anchors, example_labels,
                n_aug=3, beta_alpha=1.0, seed=seed,
            )
        max_diff = float(np.max(np.abs(replayed["cov"] - direct["cov"])))
        assert max_diff == 0.0, f"{method} replay drifted by {max_diff}"


def test_fidelity_ranking_on_shipped_anchors(example_anchors, example_labels):
    """The bundled anchors should reproduce E0 worst, G2 best."""
    n_aug = 6
    aud = lambda a: te.audit_covariance_fidelity(
        example_anchors, a["cov"], a["anchor"]
    )["log_euclidean_to_reference"]

    e0 = te.augment_e0(example_anchors, n_aug=n_aug, g0_sigma=0.15,
                       labels=example_labels, seed=1001)
    g0 = te.augment_g0(example_anchors, n_aug=n_aug, sigma=0.15,
                       labels=example_labels, seed=1002)
    g1 = te.augment_g1(example_anchors, example_labels, n_aug=n_aug,
                       sigma=0.15, seed=1003)
    g2 = te.augment_g2(example_anchors, example_labels, n_aug=n_aug,
                       beta_alpha=1.0, seed=1004)
    d_e0, d_g0, d_g1, d_g2 = aud(e0), aud(g0), aud(g1), aud(g2)
    # Loose ordering: E0 not the smallest, G2 the smallest.
    assert d_g2 == min(d_e0, d_g0, d_g1, d_g2)
    assert d_e0 != min(d_e0, d_g0, d_g1, d_g2)
