# Resource-adaptive QR3 slice

Run from a checkout with the `photometry` extra and development dependencies installed:

```sh
PYTHONPATH=. uv run --no-sync python -m examples.resource_adaptive.run_qr3 \
  --output reports/resource-adaptive-qr3
```

The output directory must not already exist. Three pinned QR3 D1 cutouts are used. The
shallow budget detects/measures the same pixels in all three exposures. The deeper budget
adds up to 12 held-out one/multiple finite-ePSF profile diagnostics for proximity proposals.
Their wavelengths come from the held-out exposure's calibrated CWAVE map; different spectral
channels are not treated as time-variable observations. A deterministic priority heuristic
allocates those diagnostic tasks and reserves exploration slots for weaker candidates.

`sieve.execution.WorkUnit` pins hashes and sky/exposure/wavelength/investigation partitions.
`ResourceBudget` limits task count, worker count, estimated CPU/RSS/storage/input, and elapsed
time. `run_work` uses bounded local threads, per-unit atomic JSON checkpoints and hash-checked
resume. Estimates gate admission; actual worker CPU, wall time, process peak RSS, logical input
bytes and serialized checkpoint bytes are recorded. Runtime, RSS and output caps are checked
after a unit completes, so they are cooperative limits rather than operating-system isolation.
Workers must be deterministic, side-effect-free JSON functions. Remote workers and cluster
coordination are not implemented.

The existing evaluator independently controls injections from the production detection code
with one-to-one positional matching and DAO/IRAF baselines. Its model-matched injections use
one real field and independent archive pixel variance; they are not sky sensitivity or
identity truth. Held-out profile delta-BIC is also diagnostic only. No validated identity gain,
false-discovery rate, or discovery claim is supported by this run.

`results/qr3-budget-run-20261015.json` retains the original scheduler run. The follow-up
`results/qr3-edge-fit-followup-20261015.json` records the cutout-edge fitter change: 61 of
the prior 66 conditional fit failures are now measurements with the unrenormalized visible
ePSF profile; 5 remain below the declared PSF-information threshold. The 10-sigma paired
injection delta checks template response only and cancels pre-existing scene flux, so it is
not evidence of independent source-flux accuracy. The recorded single two-worker timing is
descriptive and does not establish scalable performance.
