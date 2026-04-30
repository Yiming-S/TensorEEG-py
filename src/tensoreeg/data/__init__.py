"""Bundled example data for the vignettes, demos, and integration tests.

The data files in this directory are deterministic and small enough
that the package's ``[test]`` suite, the ``examples/`` scripts, and the
README quick-start can all run without external downloads.

Public API
----------
- :func:`load_example_anchors` -> ``(8, 6, 6)`` SPD ndarray
- :func:`load_example_labels` -> ``(8,)`` int ndarray (0 = class A, 1 = B)
- :func:`example_manifest_path` -> filesystem path to the bundled CSV

The contents mirror those of the R package's ``data(example_anchors)``
and ``inst/extdata/example_manifest.csv``: 8 anchors, p = 6 channels,
two classes (10 Hz alpha-like vs 6 Hz lower-frequency). The companion
manifest describes a single ``(Example, subject 1, budget 4, resample 0)``
cell with five method codes.
"""

from __future__ import annotations

import os
from importlib.resources import files

import numpy as np

__all__ = [
    "load_example_anchors",
    "load_example_labels",
    "example_manifest_path",
]


def _data_path(name: str) -> str:
    return os.fspath(files(__name__).joinpath(name))


def load_example_anchors() -> np.ndarray:
    """Return the bundled ``(n_anchor, p, p)`` SPD anchor stack."""
    arr = np.load(_data_path("example_anchors.npz"))
    return arr["anchors"]


def load_example_labels() -> np.ndarray:
    """Return the bundled length-``n_anchor`` integer label vector."""
    return np.load(_data_path("example_labels.npy"))


def example_manifest_path() -> str:
    """Return the filesystem path to the bundled calibration manifest."""
    return _data_path("example_manifest.csv")
