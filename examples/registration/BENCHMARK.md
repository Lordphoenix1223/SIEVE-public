# WCS benchmark — experimental, not merge-ready science

Source snapshot: public `main` at `9d937f9aaaa4bd7f537ba2fb2d5b76862cdef2a2`;
registration module adapted from research commit `f5a51661d48a6c5fcfe9354c08f61acb4301642c`.
The public branch has a separate history; no private commits or archives are included.

| Comparison | Existing SIEVE | Guarded WCS path | Limit |
|---|---:|---:|---|
| QR3 detected peaks | 418 / 412 | 418 / 412 | Same detector and inputs |
| QR3 sky-radius pairs | 364 | 364 | All 364 pairs identical; not truth labels |
| Motion / flux claim | Pilot INCONCLUSIVE | REFUSED / REFUSED | Guarded CLI and pilot; other entry points unchanged |
| Controlled true pairs | 24 / 24 | 24 / 24 | Six noise-free representations of four sources |
| Controlled incorrect pairs | 0 | 0 | Same WCS constructs the truth; not a sky-accuracy claim |

Astropy `search_around_sky` is the original baseline and also the upgraded search;
Astropy's nearest-coordinate interface is a reference implementation, not an
independent astrometric truth source. The six controlled cases cover 30/90/180-degree
rotation, scale, reflection and SIP distortion. Regression tests also cover partial
coverage, RA wrap, polar geometry, invalid WCS and ambiguous matching. The fixtures
are too small and dependent to report real-sky accuracy or a meaningful confidence interval.

The Gaia DR3 QR3 reference query returned 473 rows; 282 candidate proximity pairs were
found against 415 measured peaks, of which 278 were singleton pairs and four were
ambiguous. With 155 singleton candidates used to fit a constant shift, the 123 held-out
candidate residuals have median 0.657 arcsec before and 0.642 arcsec after, p95 1.325
arcsec after. Those are *candidate* catalog links, not independently certified labels;
Gaia proper motion is applied, but spacecraft parallax and SIEVE covariance are not.
The fit is diagnostic and is never applied to science outputs.

The 2MASS/MSX field contains 1,217 / 26 exploratory peaks and one cross-instrument
singleton pair, separated by 9.33 arcsec. A bounded 337-row Gaia query yields one
shared proximity candidate: 1.28 arcsec from the 2MASS peak and 8.22 arcsec from the
MSX peak. Image epochs and observer geometry are unavailable; unknown proper motion,
blending and very different passbands mean this is not a validated identity or a
qualified cross-instrument registration solution.

One 20-repeat matching-only run gave a median 1.53 ms for the original Astropy sky
search and 1.77 ms for the upgraded candidate/ambiguity search; process peak RSS was
about 156 MiB on macOS, including common detection and imports. This is a single-field
microbenchmark, not end-to-end throughput. The full pilot includes injection and
identity work, so its elapsed time cannot be compared to the narrower CLI.

## Reproduce

```sh
uv sync --frozen --extra catalog --extra specialist --extra photometry --extra registration
mkdir -p reports
uv run --no-sync pytest -q tests/registration
PYTHONPATH=. uv run --no-sync python -m examples.qualification.run_experiment --output reports/qr3-pilot
uv run --no-sync sieve compare-qr3 --first examples/qualification/qr3-d1-exposure-1.fits --second examples/qualification/qr3-d1-exposure-2.fits --output reports/compare-qr3.json
PYTHONPATH=. uv run --no-sync python examples/registration/benchmark_qr3.py --baseline reports/qr3-pilot --upgraded reports/compare-qr3.json --output reports/benchmark-qr3.json
PYTHONPATH=. uv run --no-sync python examples/registration/benchmark_gaia.py --measurements reports/qr3-pilot/measurements.json --output reports/benchmark-gaia.json
PYTHONPATH=. uv run --no-sync python examples/registration/benchmark_geometry.py --output reports/benchmark-geometry.json
PYTHONPATH=. uv run --no-sync python examples/registration/benchmark_timing.py --output reports/benchmark-timing.json
```

The pilot's original matching code is unchanged apart from a preflight guard and a
registration receipt; its pairing uses the same Astropy call as public `main`.
Stored benchmark outputs and Gaia query snapshots are in this directory. The two
Gaia query URLs, exact ADQL, cutout hashes and limitations are retained in their
machine-readable reports. Cross-instrument data download commands are in [README.md](README.md);
then run `benchmark_cross_gaia.py --data-dir reports/cross-input --output reports/cross-gaia.json`.

**Recommendation: keep experimental.** The `compare-qr3` CLI and guarded pilot refuse
motion and flux-variability claims, but the older identity and triage entry points do
not enforce this new gate. There is no independent qualified astrometric covariance,
validated physical identity, trained anomaly model, PSF-matched difference image,
or validated multi-telescope false-match rate. A sampled overlap is not an exact
footprint; Reproject output is visualization only, with no propagated covariance.
