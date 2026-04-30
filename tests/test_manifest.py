"""Tests for tensoreeg.manifest (read CSV + replay)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import tensoreeg as te


def test_read_calibration_manifest_basic(example_manifest_path):
    rows = te.read_calibration_manifest(example_manifest_path)
    assert len(rows) == 10  # 5 methods x 2 classes
    methods = {r["method_code"] for r in rows}
    assert methods == {"R0", "E0", "G0", "G1", "G2"}
    classes = {r["class_label"] for r in rows}
    assert classes == {0, 1}
    # real_trial_ids parses to a Python list of ints.
    for r in rows:
        assert isinstance(r["real_trial_ids"], list)
        for tid in r["real_trial_ids"]:
            assert isinstance(tid, int)


def test_read_calibration_manifest_missing_file():
    with pytest.raises(FileNotFoundError):
        te.read_calibration_manifest("/nonexistent/path.csv")


def test_read_calibration_manifest_missing_columns():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.csv")
        with open(path, "w") as fh:
            fh.write("dataset,subject\nfoo,1\n")
        with pytest.raises(ValueError, match="missing required columns"):
            te.read_calibration_manifest(path)


def test_replay_g2_byte_equivalent_to_direct(
    example_manifest_path, example_anchors, example_labels
):
    """Manifest replay must produce identical bytes to a direct seeded call.

    (R-side test calls this 'byte-equivalence within Python'; the
    cross-language analogue is checked in test_cross_language.py at the
    metric level.)
    """
    manifest = te.read_calibration_manifest(example_manifest_path)
    g2_group = [r for r in manifest if r["method_code"] == "G2"]
    g2_seed = g2_group[0]["method_seed"]

    replayed = te.replay_from_manifest(
        g2_group, example_anchors, example_labels,
        ratio_per_anchor=3, g2_beta_alpha=1.0,
    )
    direct = te.augment_g2(
        example_anchors, example_labels,
        n_aug=3, beta_alpha=1.0, seed=g2_seed,
    )
    assert replayed["cov"].shape == direct["cov"].shape
    max_diff = float(np.max(np.abs(replayed["cov"] - direct["cov"])))
    assert max_diff == 0.0


def test_replay_r0_returns_empty_stack(
    example_manifest_path, example_anchors, example_labels
):
    manifest = te.read_calibration_manifest(example_manifest_path)
    r0_group = [r for r in manifest if r["method_code"] == "R0"]
    res = te.replay_from_manifest(r0_group, example_anchors, example_labels)
    assert res["cov"].shape == (0, example_anchors.shape[1], example_anchors.shape[1])


def test_replay_a0_raises(
    example_manifest_path, example_anchors, example_labels
):
    """A0 is not replayable from manifest seeds."""
    manifest = te.read_calibration_manifest(example_manifest_path)
    # The bundled manifest has no A0 row, but we synthesise one to test.
    fake = dict(manifest[0])
    fake["method_code"] = "A0"
    with pytest.raises(NotImplementedError, match="transductive"):
        te.replay_from_manifest([fake], example_anchors, example_labels)


def test_replay_validates_manifest_group_uniqueness(
    example_manifest_path, example_anchors, example_labels
):
    manifest = te.read_calibration_manifest(example_manifest_path)
    g0_rows = [r for r in manifest if r["method_code"] == "G0"]
    g1_rows = [r for r in manifest if r["method_code"] == "G1"]
    mixed = g0_rows + g1_rows
    with pytest.raises(ValueError, match="exactly one"):
        te.replay_from_manifest(mixed, example_anchors, example_labels)


def test_replay_unknown_method_raises(
    example_manifest_path, example_anchors, example_labels
):
    manifest = te.read_calibration_manifest(example_manifest_path)
    fake = dict(manifest[0])
    fake["method_code"] = "ZZ"
    with pytest.raises(ValueError, match="unknown method_code"):
        te.replay_from_manifest([fake], example_anchors, example_labels)


def test_replay_each_method_has_correct_count(
    example_manifest_path, example_anchors, example_labels
):
    manifest = te.read_calibration_manifest(example_manifest_path)
    n = example_anchors.shape[0]
    for method in ("E0", "G0", "G1", "G2"):
        group = [r for r in manifest if r["method_code"] == method]
        res = te.replay_from_manifest(
            group, example_anchors, example_labels, ratio_per_anchor=3
        )
        assert res["cov"].shape == (n * 3, example_anchors.shape[1], example_anchors.shape[2]), method
