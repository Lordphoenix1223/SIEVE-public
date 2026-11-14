# WCS registration diagnostics

This experimental layer preserves native measurements and reports geometry; it does
not establish accurate astrometry, physical identity, motion, variability, or a valid
difference image. FITS metadata are normalized through Astropy WCS/ICRS and exposed
with the original input SHA-256. Missing metadata remain unknown. The WCS object is
used in memory (including supported distortions); the original FITS is the authoritative
WCS record. Round-trip accuracy checks numerical self-consistency, not truth on the sky.

The sampled rectangular footprint test can miss small or irregular intersections; a
`NO_SAMPLED_COMMON_COVERAGE` result is not proof of zero overlap. SOURCE detections
are paired in sky coordinates within a caller-supplied gate. Competing pairings are
retained as ambiguous; neither unique proximity nor unmatched points establish identity
or an anomaly. A median-shift diagnostic uses one half of supplied reference candidates
and evaluates the other half. These candidates are not independently identified stars,
so status remains `REGISTRATION_UNQUALIFIED`; the shift is not applied to any image.

The comparison gate is task-specific: duplicate identifiers, missing timestamps,
median wavelength intervals, masks and calibration limits are reported separately.
Image-level variability and source-motion verdicts remain closed. SPHEREx CWAVE/CBAND
medians are approximate context, not local calibrated filter responses. Reproject is
used for visualization only; bitwise flags are used to mask the native input, output
coverage is retained, and uncertainty is deliberately not propagated. Interpolation
correlates noise; the output is not suitable for photometry or image differencing.

From the repository root (Python 3.12 and uv):

```sh
uv sync --frozen --extra catalog --extra specialist --extra photometry --extra registration
uv run --no-sync pytest -q tests/registration
uv run --no-sync python examples/registration/run_qr3.py --output reports/qr3-wcs.json
```

Use a new output path. For the bounded cross-instrument geometry example, retrieve
Astropy's public 2MASS K and MSX E images and execute:

```sh
mkdir -p reports/cross-input
curl -fsSL https://www.astropy.org/astropy-data/galactic_center/gc_2mass_k.fits -o reports/cross-input/2mass.fits
curl -fsSL https://www.astropy.org/astropy-data/galactic_center/gc_msx_e.fits -o reports/cross-input/msx.fits
uv run --no-sync python examples/registration/run_cross_instrument.py --data-dir reports/cross-input --output reports/cross-wcs.json
```

Their SHA-256 values are pinned in the script. The two exposures lack adequate metadata
and aligned uncertainty/quality planes; their point-source proposals, one proximity
pair, and common geometry do not validate cross-instrument astrometric registration or
photometric comparability. They use different wavelengths and angular resolutions.

References: [Astropy WCS](https://docs.astropy.org/en/stable/wcs/),
[Reproject](https://reproject.readthedocs.io/) (resampling assumes correct WCS),
[SPHEREx data products](https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-products/),
and [proper image subtraction](https://arxiv.org/abs/1601.02655), which requires
appropriate calibration, PSFs and registration beyond this demonstration.

See [BENCHMARK.md](BENCHMARK.md) for baseline comparisons, Gaia cohorts, limitations and the experimental CLI.
