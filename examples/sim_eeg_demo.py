"""SimEEG simulator demonstration: generate a tensor and validate it.

Mirrors the ``tensoreeg-validation`` vignette of the R package: produce
a small synthetic tensor with controlled SNR, and report the realised
SNR audit, the spatial / temporal / spectral structure, and whether
the closed-loop SNR control hit its target.

Run::

    python examples/sim_eeg_demo.py
"""

from __future__ import annotations

import numpy as np

import tensoreeg as te


def main() -> None:
    print("Generating a 6-trial / 8-channel / 4-source tensor at 5 dB...")
    sim = te.sim_eeg_master(
        n_trials=6,
        n_time=400,
        n_channels=8,
        n_sources=4,
        fs=250.0,
        snr_neural_db=5.0,
        snr_artifact_db=0.0,
        drift_power_ratio=0.5,
        target_freqs=[10, 12, 11, 9],
        seed=42,
        verbose=False,
    )
    print(f"  data shape:  {sim['data'].shape}")
    print(f"  labels:      {sim['labels'].tolist()}")
    print(f"  fs:          {sim['params']['fs']:.0f} Hz")

    snr = np.array([row["Realized_SNR_Neural"] for row in sim["audit"]])
    print(f"  realised SNR (dB): {np.round(snr, 3).tolist()}  "
          f"(target = 5.0)")

    # Convert to covariance and run a quick augmentation + audit.
    cov = te.tensor_to_cov(sim["data"], ridge=1e-6, centre=True)
    print(f"\n  cov stack:   {cov.shape} (one SPD covariance per trial)")
    aug_g2 = te.augment_g2(cov, sim["labels"], n_aug=3, beta_alpha=1.0,
                            seed=99)
    audit = te.audit_covariance_fidelity(cov, aug_g2["cov"], aug_g2["anchor"])
    print(
        f"  G2 audit:    LE={audit['log_euclidean_to_reference']:.4f}, "
        f"AI={audit['affine_invariant_to_reference']:.4f}, "
        f"anchor={audit['anchor_perturbation_distance']:.4f}"
    )

    # Multi-run session.
    print("\nGenerating a 2-run session with 3 trials per run, 2-trial gap...")
    multi = te.sim_multirun_session(
        n_runs=2, trials_per_run=3, gap_trials=2,
        n_time=200, n_channels=4, n_sources=2,
        target_freqs=[10, 12], seed=7, verbose=False,
    )
    print(f"  trials:      {len(multi['x'])}")
    print(f"  run ids:     {multi['run'].tolist()}")
    print(f"  fs:          {multi['fs']:.0f} Hz")

    # Optional: matplotlib dashboard if available.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("\n(matplotlib not installed -> skipping the validation dashboard)")
        return
    from tensoreeg.visualization import plot_sim_dashboard
    fig = plot_sim_dashboard(sim, channel=0, fmax=60.0)
    out_path = "examples/sim_dashboard.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nDashboard saved to {out_path}")


if __name__ == "__main__":
    main()
