"""Calibration-manifest reading and seeded replay.

The Python protocol driver in ``scripts/protocol/`` of the manuscript
writes a per-cell calibration manifest CSV that records, for every
``(dataset, subject, budget, resample, method, class)`` row,

- the source-trial ids that served as anchors (0-based, JSON list),
- a per-method seed,
- the augmentation ratio,
- a ``target_session_ids_used`` flag for the leakage audit.

The two functions in this module are the R analogues' Python siblings:

- :func:`read_calibration_manifest` parses the CSV without requiring
  pandas (uses the stdlib ``csv`` module).
- :func:`replay_from_manifest` rebuilds the synthetic stack for a single
  cell using the recorded seed and the supplied source covariances. The
  output is byte-equivalent to a direct seeded call to the underlying
  augmentation routine inside Python; cross-language byte equivalence
  is not guaranteed (see ``augmentation/_common.py:make_rng``).
"""

from __future__ import annotations

import csv
import os
import re
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from . import augmentation
from .augmentation._common import CovStack, coerce_cov_stack, coerce_labels

__all__ = [
    "MANIFEST_COLUMNS",
    "read_calibration_manifest",
    "replay_from_manifest",
]


MANIFEST_COLUMNS: List[str] = [
    "dataset",
    "subject",
    "budget_per_class",
    "resample_id",
    "method_code",
    "class_label",
    "n_real_trials",
    "real_trial_ids",
    "real_seed",
    "method_seed",
    "n_synthetic_per_anchor",
    "target_session_ids_used",
    "status",
    "notes",
]


_INT_LIST_RE = re.compile(r"-?\d+")


def _parse_int_list(s: str) -> List[int]:
    """Parse a JSON-like integer list ``"[1, 2, 3]"`` to ``[1, 2, 3]``.

    Empty / blank input maps to an empty list. We avoid a JSON dependency
    so the package stays numpy + scipy only, but accept any whitespace
    or bracket variation that the Python or R writers might produce.
    """
    if s is None:
        return []
    text = str(s).strip()
    if not text:
        return []
    return [int(m.group(0)) for m in _INT_LIST_RE.finditer(text)]


def _coerce_int(s: str, name: str) -> int:
    try:
        return int(float(s))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest column {name!r} value {s!r} is not an integer.") from exc


def read_calibration_manifest(path: Union[str, os.PathLike]) -> List[Dict]:
    """Read a calibration-manifest CSV and return a list of row dicts.

    Each returned dict has the keys in :data:`MANIFEST_COLUMNS`, with
    ``real_trial_ids`` parsed into a Python list of ints and integer
    columns coerced to ``int``. ``method_code`` and ``status`` are kept
    as strings; ``target_session_ids_used`` is kept as the original
    string (``"True"`` / ``"False"``) to remain faithful to what the
    Python driver writes.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest file not found: {path}")
    rows: List[Dict] = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"manifest at {path} is empty.")
        missing = [c for c in MANIFEST_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"manifest is missing required columns: {', '.join(missing)}"
            )
        for raw in reader:
            row: Dict = {
                "dataset": str(raw["dataset"]),
                "subject": _coerce_int(raw["subject"], "subject"),
                "budget_per_class": _coerce_int(
                    raw["budget_per_class"], "budget_per_class"
                ),
                "resample_id": _coerce_int(raw["resample_id"], "resample_id"),
                "method_code": str(raw["method_code"]),
                "class_label": _coerce_int(raw["class_label"], "class_label"),
                "n_real_trials": _coerce_int(raw["n_real_trials"], "n_real_trials"),
                "real_trial_ids": _parse_int_list(raw["real_trial_ids"]),
                "real_seed": _coerce_int(raw["real_seed"], "real_seed"),
                "method_seed": _coerce_int(raw["method_seed"], "method_seed"),
                "n_synthetic_per_anchor": _coerce_int(
                    raw["n_synthetic_per_anchor"], "n_synthetic_per_anchor"
                ),
                "target_session_ids_used": str(raw["target_session_ids_used"]),
                "status": str(raw["status"]),
                "notes": str(raw["notes"]),
            }
            rows.append(row)
    return rows


def _filter_group(
    manifest: List[Dict],
    method_code: Optional[str] = None,
    subject: Optional[int] = None,
    budget_per_class: Optional[int] = None,
    resample_id: Optional[int] = None,
    dataset: Optional[str] = None,
) -> List[Dict]:
    """Convenience filter for picking out the rows of one manifest cell.

    Returns the rows (one per class) that match the supplied keys.
    """
    out = list(manifest)
    if dataset is not None:
        out = [r for r in out if r["dataset"] == dataset]
    if subject is not None:
        out = [r for r in out if r["subject"] == subject]
    if budget_per_class is not None:
        out = [r for r in out if r["budget_per_class"] == budget_per_class]
    if resample_id is not None:
        out = [r for r in out if r["resample_id"] == resample_id]
    if method_code is not None:
        out = [r for r in out if r["method_code"] == method_code]
    return out


def replay_from_manifest(
    manifest_group: List[Dict],
    source_cov_list: CovStack,
    source_labels: Sequence,
    ratio_per_anchor: int = 3,
    g0_sigma: float = 0.15,
    g1_sigma: float = 0.15,
    g2_beta_alpha: float = 1.0,
) -> dict:
    """Replay one ``(dataset, subject, budget, resample, method)`` cell.

    Parameters
    ----------
    manifest_group : list of dict
        Rows for a single cell (typically one row per class label).
        Filter using :func:`read_calibration_manifest` plus a list
        comprehension or :func:`_filter_group`.
    source_cov_list : list of (p, p) ndarray, or (n_src, p, p) ndarray
        Source-session SPD covariances, indexed by trial id (Python uses
        0-based indexing, which matches the manifest convention).
    source_labels : sequence
        Labels for the source covariances (length ``n_src``).
    ratio_per_anchor : int, default 3
        Augmentation ratio used in the original run.
    g0_sigma, g1_sigma : float, default 0.15
        Tangent perturbation scales for G0 / G1 / E0.
    g2_beta_alpha : float, default 1.0
        Symmetric Beta concentration used by G2 mixup.

    Returns
    -------
    result : dict
        Same shape as the underlying augmentation routine, plus an
        ``anchors_used`` field listing the actual anchor covariances
        consumed by this cell.

    Raises
    ------
    ValueError
        If the group does not contain exactly one row per class, or if
        ``real_trial_ids`` reference indices outside the source range,
        or if ``method_code`` is not one of the supported codes.
    NotImplementedError
        For ``method_code == "A0"`` (transductive; cannot be replayed
        from seed alone).
    """
    if not manifest_group:
        raise ValueError("manifest_group must be a non-empty list of row dicts.")
    cells = {
        (r["dataset"], r["subject"], r["budget_per_class"],
         r["resample_id"], r["method_code"])
        for r in manifest_group
    }
    if len(cells) != 1:
        raise ValueError(
            "manifest_group must contain exactly one "
            "(dataset, subject, budget, resample, method) cell."
        )
    method = manifest_group[0]["method_code"]
    method_seed = int(manifest_group[0]["method_seed"])

    source = coerce_cov_stack(source_cov_list, "source_cov_list")
    n_src = source.shape[0]
    labels_arr = coerce_labels(source_labels, n_src, required=True)

    # Assemble anchors from real_trial_ids in sorted-class order.
    classes = sorted({r["class_label"] for r in manifest_group})
    anchor_indices: List[int] = []
    anchor_labels: List[int] = []
    for cls in classes:
        rows = [r for r in manifest_group if r["class_label"] == cls]
        if len(rows) != 1:
            raise ValueError(
                f"manifest cell has {len(rows)} rows for class {cls}; expected 1."
            )
        ids = list(rows[0]["real_trial_ids"])
        for tid in ids:
            if not (0 <= tid < n_src):
                raise ValueError(
                    f"manifest real_trial_ids reference index {tid} "
                    f"outside source range [0, {n_src})."
                )
        anchor_indices.extend(ids)
        anchor_labels.extend([int(cls)] * len(ids))

    anchors = source[anchor_indices]
    anchor_labels_arr = np.asarray(anchor_labels, dtype=np.int64)

    if method == "R0":
        return {
            "cov": np.empty((0, source.shape[1], source.shape[2]), dtype=float),
            "anchor": np.empty(0, dtype=np.int64),
            "replicate": np.empty(0, dtype=np.int64),
            "labels": np.empty(0, dtype=anchor_labels_arr.dtype),
            "anchors_used": anchors,
            "params": {"method": "R0", "method_seed": method_seed},
        }
    if method == "G0":
        res = augmentation.augment_cov_riemannian(
            anchors,
            n_aug=ratio_per_anchor,
            sigma=g0_sigma,
            labels=anchor_labels_arr,
            seed=method_seed,
        )
    elif method == "E0":
        res = augmentation.augment_cov_amplitude_matched_euclidean(
            anchors,
            n_aug=ratio_per_anchor,
            g0_sigma=g0_sigma,
            labels=anchor_labels_arr,
            seed=method_seed,
        )
    elif method == "G1":
        res = augmentation.augment_cov_empirical_tangent(
            anchors,
            anchor_labels_arr,
            n_aug=ratio_per_anchor,
            sigma=g1_sigma,
            seed=method_seed,
        )
    elif method == "G2":
        res = augmentation.augment_cov_geodesic_mixup(
            anchors,
            anchor_labels_arr,
            n_aug=ratio_per_anchor,
            beta_alpha=g2_beta_alpha,
            seed=method_seed,
        )
    elif method == "A0":
        raise NotImplementedError(
            "A0 (Riemannian alignment) is transductive and cannot be "
            "replayed from manifest seeds alone. Use "
            "augment_cov_alignment_riemannian() with the unlabeled target "
            "covariance stack instead."
        )
    else:
        raise ValueError(f"unknown method_code in manifest_group: {method}")

    res["anchors_used"] = anchors
    return res
