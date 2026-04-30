# TensorEEG-py

**Physics-Constrained EEG Simulation and Covariance-Aware Augmentation (Python)**

<p align="center">
  <img src="docs/cover.png"
       alt="TensorEEG cover: Simulate, SPD covariance, augmentation footprint, fidelity audit"
       width="100%">
</p>

[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.10.0-orange)](pyproject.toml)
[![R sibling](https://img.shields.io/badge/R-TensorEEG-yellow)](https://github.com/Yiming-S/TensorEEG)

TensorEEG-py is the Python port of the [TensorEEG R
package](https://github.com/Yiming-S/TensorEEG). It does two things
end-to-end:

1. **Simulate** physically-consistent synthetic EEG as a 3rd-order
   tensor :math:`\mathcal{X} \in \mathbb{R}^{T \times C \times K}` with
   volume-conduction geometry, structured oscillations, and
   trial-wise manifold drift.
2. **Audit** covariance-level data augmentation for cross-session BCI
   transfer through a covariance-aware augmentation family
   (E0 / G0 / G1 / G2 / A0), a six-metric covariance fidelity audit,
   and manifest-based replay for cross-language reproducibility.

It is the Python reference implementation for Shen & Degras (2026),
*Covariance Geometry as a Safety Constraint for Cross-Session BCI
Augmentation: A Multi-Dataset Non-Inferiority and Fidelity Audit.*

---

## Why a Python sibling?

The R package is canonical; TensorEEG-py mirrors its full API so that:

- **Python pipelines** (PyTorch / scikit-learn / pyriemann / MOABB) can
  consume the same simulation and audit primitives without an R-Python
  bridge,
- the manuscript's `scripts/protocol/` driver and any user's R or
  Python audit can talk to each other through a shared
  `calibration_manifest.csv` schema,
- a reviewer can byte-equivalently reproduce a synthetic stack from
  manifest seeds **inside Python**, and check that the metric-level
  fidelity output agrees with the R run.

Cross-language reproducibility is at the metric level (fidelity numbers
agree to small relative tolerance), not at byte level: NumPy and R use
different RNGs, so the same scalar seed yields different bytes. Within
each language, byte-equivalence is enforced.

---

## What's in 0.10.0

| Module | Functions | What they do |
|---|---|---|
| `tensoreeg.augmentation` | `augment_cov_amplitude_matched_euclidean` (E0), `augment_cov_riemannian` (G0), `augment_cov_empirical_tangent` (G1), `augment_cov_geodesic_mixup` (G2), `augment_cov_alignment_riemannian` (A0) + short aliases `augment_e0/g0/g1/g2/a0` | Five SPD-aware ways to expand a covariance set, ranging from off-manifold Euclidean control to log-Euclidean geodesic mixup |
| `tensoreeg.fidelity` | `audit_covariance_fidelity` + 5 standalone metrics | Six-metric audit, raw + dimension-normalised |
| `tensoreeg.manifest` | `read_calibration_manifest`, `replay_from_manifest` | Read the per-cell manifest CSV and reconstruct the synthetic stack byte-equivalently in Python |
| `tensoreeg.simulation` | `sim_eeg_master`, `sim_multirun_session`, `generate_geometry_mixing`, `generate_drift_rotations`, `setup_var2_system`, `sim_source_var2`, `sim_source_task`, `sim_artifacts` | Full SimEEG generator: Fibonacci-grid hemispherical sensors, Tikhonov-Laplacian volume conduction, OU/AR(1) or fBm manifold drift, VAR(2) backgrounds, Gabor task ERP, EOG/EMG/drift artifacts, closed-loop SNR |
| `tensoreeg.spd` | `spd_logm`, `spd_expm`, `spd_project`, `log_euclidean_distance`, `affine_invariant_distance`, `vech_log`, `unvech` | Numerical SPD primitives that all of the above are built on |
| `tensoreeg.visualization` | `plot_topomap`, `plot_run_drift`, `plot_sim_dashboard` | Optional matplotlib helpers (install with `pip install "tensoreeg[plot]"`) |
| `tensoreeg.data` | `load_example_anchors`, `load_example_labels`, `example_manifest_path` | Eight 6 x 6 SPD anchors + a 10-row demo manifest CSV |

---

## Installation

```bash
# from PyPI (when published; not yet uploaded for 0.10.0)
pip install tensoreeg

# from source / GitHub
pip install git+https://github.com/Yiming-S/TensorEEG-py.git

# editable / dev install (recommended while contributing)
git clone https://github.com/Yiming-S/TensorEEG-py.git
cd TensorEEG-py
pip install -e .[dev]
```

Hard runtime dependencies are NumPy >= 1.21 and SciPy >= 1.7. The
`[plot]` extra pulls matplotlib for the visualisation helpers; `[test]`
pulls pytest.

---

## Quick start

The shortest end-to-end audit you can run after installation. No
external data required --- the bundled `example_anchors` is enough.

```python
import tensoreeg as te

anchors = te.data.load_example_anchors()   # (8, 6, 6) SPD
labels  = te.data.load_example_labels()    # (8,) class labels {0, 1}

# 1. Augment the anchors four ways.
aug_e0 = te.augment_e0(anchors, n_aug=3, g0_sigma=0.15, labels=labels, seed=1001)
aug_g0 = te.augment_g0(anchors, n_aug=3, sigma=0.15,    labels=labels, seed=1002)
aug_g1 = te.augment_g1(anchors, labels, n_aug=3, sigma=0.15,        seed=1003)
aug_g2 = te.augment_g2(anchors, labels, n_aug=3, beta_alpha=1.0,    seed=1004)

# 2. Run the six-metric fidelity audit on each.
def le(a):
    return te.audit_covariance_fidelity(
        anchors, a["cov"], a["anchor"]
    )["normalized"]["log_euclidean"]

print({name: round(le(a), 4) for name, a in
       [("E0", aug_e0), ("G0", aug_g0), ("G1", aug_g1), ("G2", aug_g2)]})
# {'E0': 0.6014, 'G0': 0.5927, 'G1': 0.5763, 'G2': 0.4717}
```

Lower distance = closer to the class-mean covariance geometry. The
expected ranking E0 > G0 > G1 > G2 is reproduced in seconds on the
bundled toy data.

---

## The four pieces

### Simulate

`tensoreeg.sim_eeg_master` generates a 3rd-order tensor
:math:`\mathcal{X} \in \mathbb{R}^{T \times C \times K}` from a
physically-constrained generative process:

- **Volume-conduction physics.** Sources are placed on a Fibonacci grid
  on the unit sphere and projected through a forward mixing matrix
  regularised with Tikhonov smoothing on a normalised graph Laplacian.
- **Manifold drift.** Trial-to-trial non-stationarity is modelled as a
  geodesic walk on :math:`SO(n)`, with the angle path driven by an
  OU/AR(1) process by default or a long-memory fBm path
  (`process="fbm"` in `generate_drift_rotations`).
- **Closed-loop SNR.** SNR is calibrated against an effective AC-power
  metric (`tensoreeg.calc_ac_power`) measured after a 4th-order
  Butterworth high-pass at 0.1 Hz, so slow drift cannot inflate
  apparent neural power.

```python
sim = te.sim_eeg_master(
    n_trials=20, n_time=500, n_channels=64, n_sources=10,
    fs=250.0, snr_neural_db=5.0, target_freqs=None,
    seed=42, verbose=False,
)
sim["data"].shape    # (500, 64, 20)
sim["audit"]         # list of dicts with realised SNR per trial
```

### Augment

Five covariance-level routines that share the same SPD-manifold
definitions but differ in *where* they place synthetic samples:

| Code | Function | Where it places synthetic samples |
|---|---|---|
| **E0** | `augment_e0` | Off-manifold Euclidean perturbation, calibrated so the median anchor distance matches G0's --- the "fair" off-manifold control |
| **G0** | `augment_g0` | Isotropic tangent jitter at each anchor; legacy log-Euclidean baseline |
| **G1** | `augment_g1` | Class-aware Gaussian in tangent coordinates with Ledoit--Wolf shrinkage |
| **G2** | `augment_g2` | Same-class log-Euclidean geodesic interpolation; the most fidelity-preserving variant |
| **A0** | `augment_a0` | Transductive Riemannian alignment of source covariances to the (unlabeled) target session mean |

E0/G0/G1/G2 are non-transductive: they need only labelled source
covariances and a seed. A0 is transductive and consumes the unlabeled
target stack.

### Audit

```python
res = te.audit_covariance_fidelity(anchors, aug_g2["cov"], aug_g2["anchor"])
res.keys()
# dict_keys(['n_synthetic',
#            'log_euclidean_to_reference',
#            'affine_invariant_to_reference',
#            'eigenvalue_correlation',
#            'trace_ratio', 'condition_number_ratio',
#            'anchor_perturbation_distance',
#            'normalized'])
res["normalized"].keys()
# dict_keys(['log_euclidean', 'affine_invariant',
#            'anchor_perturbation', 'feature_dim'])
```

Every distance is reported dimension-normalised by
:math:`\sqrt{p(p+1)/2}` so audits across datasets with different
channel counts can be pooled honestly. The five standalone metric
functions (`cov_logeuclidean_distance`, `cov_affine_invariant_distance`,
`cov_eigenvalue_correlation`, `cov_trace_ratio`, `cov_condition_ratio`,
`cov_anchor_perturbation_distance`) are exported individually if you
want to mix them into a custom pipeline.

### Replay

```python
manifest = te.read_calibration_manifest(te.data.example_manifest_path())
g2_group = [r for r in manifest if r["method_code"] == "G2"]

result = te.replay_from_manifest(
    g2_group,
    source_cov_list=anchors,
    source_labels=labels,
    ratio_per_anchor=3,
    g2_beta_alpha=1.0,
)
# result["cov"] matches augment_g2(..., seed=g2_group[0]["method_seed"])["cov"]
# byte-for-byte within Python.
```

A0 is intentionally not replayable from manifest seeds alone (it needs
the target session covariances), so it raises `NotImplementedError`
with a pointer to `augment_a0`.

---

## Examples

```bash
python examples/covariance_audit.py    # E0/G0/G1/G2 fidelity ranking demo
python examples/manifest_replay.py     # cross-language reproducibility demo
python examples/sim_eeg_demo.py        # SimEEG simulator + multirun session
```

The first script reproduces the four-method audit from Shen & Degras
(2026) on the bundled 8-anchor stack. The second reads the bundled
manifest CSV and verifies byte-equivalent replay against direct seeded
calls. The third runs the full simulation pipeline, optionally
producing a matplotlib validation dashboard if matplotlib is
installed.

---

## Testing

```bash
pip install -e .[test]
pytest -q
```

Currently 65 tests across the SPD, augmentation, fidelity, manifest,
simulation, and end-to-end modules. The end-to-end suite exercises the
full `sim_eeg_master() -> tensor_to_cov() -> 4 augmentations ->
audit_covariance_fidelity()` pipeline and verifies byte-equivalent
manifest replay for every non-A0 method.

---

## Cross-language with TensorEEG (R)

| What you want | Where it lives |
|---|---|
| The covariance-audit pipeline in R | [TensorEEG R package](https://github.com/Yiming-S/TensorEEG) |
| The same pipeline in Python | this package |
| Python protocol driver that produces `calibration_manifest.csv` | `scripts/protocol/` of the manuscript repository |
| The paper, with the multi-dataset audit results | [paper directory](https://github.com/Yiming-S/TensorEEG) |

The two language ports keep their public function names aligned
(`augment_cov_*`, `audit_covariance_fidelity`,
`read_calibration_manifest`, `replay_from_manifest`,
`sim_eeg_master`, ...). Manifest CSVs round-trip without translation.
A reviewer can run the R audit, send you the manifest, and you can
replay it in Python and check fidelity numbers without leaving your
environment.

---

## Citation

```bibtex
@article{ShenDegras2026,
  author = {Shen, Yiming and Degras, David},
  title  = {Covariance Geometry as a Safety Constraint for
            Cross-Session BCI Augmentation: A Multi-Dataset
            Non-Inferiority and Fidelity Audit},
  year   = {2026}
}

@software{ShenTensorEEG2026,
  author  = {Shen, Yiming},
  title   = {TensorEEG: A covariance-aware EEG simulation and
             augmentation toolkit (R + Python)},
  version = {0.10.0},
  year    = {2026},
  url     = {https://github.com/Yiming-S/TensorEEG}
}
```

---

## References

1. Harshman, R. A. (1972). *PARAFAC2: Mathematical and technical
   notes*. UCLA Working Papers in Phonetics.
2. Pennec, X., Fillard, P., Ayache, N. (2006). A Riemannian framework
   for tensor computing. *Int. J. Comput. Vis.*, 66(1), 41--66.
3. Arsigny, V., Fillard, P., Pennec, X., Ayache, N. (2007). Geometric
   means in a novel vector space structure on symmetric positive-
   definite matrices. *SIAM J. Matrix Anal. Appl.*, 29(1), 328--347.
4. Barachant, A., Bonnet, S., Congedo, M., Jutten, C. (2011).
   Multiclass brain-computer interface classification by Riemannian
   geometry. *IEEE Transactions on Biomedical Engineering*, 59(4),
   920--928.
5. Mandelbrot, B. B., Van Ness, J. W. (1968). Fractional Brownian
   motions, fractional noises and applications. *SIAM Review*, 10(4),
   422--437.
6. Ledoit, O., Wolf, M. (2004). A well-conditioned estimator for
   large-dimensional covariance matrices. *Journal of Multivariate
   Analysis*, 88(2), 365--411.
7. Higham, N. J. (2008). *Functions of Matrices: Theory and
   Computation*. SIAM.
8. Chung, F. R. K. (1997). *Spectral Graph Theory*. American
   Mathematical Society.
9. Nunez, P. L., Srinivasan, R. (2006). *Electric Fields of the
   Brain: The Neurophysics of EEG* (2nd ed.). Oxford University Press.

---

## License

MIT. See [LICENSE](LICENSE).
