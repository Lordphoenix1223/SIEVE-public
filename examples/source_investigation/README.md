# Bounded QR3 source investigation

This is a research demonstration, not a discovery, confirmed identity, calibrated
artifact classifier, or Harness verdict. `evidence.json` is a validated example for
Aladin: `cases[].ra_deg` and `dec_deg` are ICRS degrees. Case A is a tentative Gaia
association to real QR3 pixels. B is a labeled one-pixel injection in Gaussian noise,
with sky coordinates projected through the real image WCS; it is not a real detection.
C asks a different question about the same real source as A and is inconclusive.

## One-command source demonstration

From a checkout with the pinned environment installed (`uv sync --frozen --extra
photometry`), run this single command with a **new** output directory:

```sh
PYTHONPATH=. uv run --no-sync python examples/source_investigation/run_demo.py --output reports/source-demo
```

The command verifies and loads the two real FITS cutouts, detects and measures the
selected source, associates it with the retained ESA Gaia DR3 cone result, recomputes
the independent centroid check, and validates and exports the evidence. It writes
`evidence.json` (the inspector record is case A), `evidence.schema.json`,
`aladin-marker.json` and its schema, the exact catalog input, retained artifact results,
and `run-manifest.json` with output hashes and integration statuses. The artifact case
is an existing labeled simulation; this command does not rerun its experiment.
The marker's `inspector_file` and `inspector_case_id` identify the source inspector.

To query Gaia live instead of using the retained real query result, append `--live-gaia`.
The live query must return the same fixed cohort; a network failure or changed result
stops the run rather than falling back silently. Neither path runs SearchX Identify or
Harness. The source remains tentative and the independent check inconclusive.
The command refuses to overwrite an existing output directory. To inspect the output,
load `aladin-marker.json`, place its ICRS `ra_deg`/`dec_deg` in Aladin, and load case A
from `evidence.json` for the inspector.

## Reproduce the underlying experiments offline

Use Python 3.12 and the repository's pinned dependencies and photometry extra:

```sh
uv sync --frozen --extra photometry
PYTHONPATH=. uv run --no-sync python examples/source_investigation/artifact_experiment.py
PYTHONPATH=. uv run --no-sync python examples/source_investigation/build_evidence.py
uv run --no-sync pytest -q tests/photometry/test_source_investigation.py
```

Both scripts regenerate their output files; preserve copies before running if needed.
`Package.model_validate_json` validates the evidence, and `evidence.schema.json` is its
JSON Schema Draft 2020-12. For independent JSON Schema validation, with `jsonschema`
installed in your own environment, use:

```sh
python -m jsonschema -i examples/source_investigation/evidence.json examples/source_investigation/evidence.schema.json
```

The checked-in input cutouts, their SHA-256, the immutable parent URLs, processing version,
and calibration inputs are pinned in `../qualification/source.json`. The code verifies
cutout hashes before analysis. The Gaia query and service are stored in `evidence.json`;
`gaia-dr3-cone.csv` is the actual ESA query result retained for offline reproduction.
The search is limited to 20 arcsec around the detection at catalog epoch 2016; sources
outside that cone with very high proper motion are not enumerated. The catalog row is
propagated to the observation epoch without distance or radial velocity, and covariance
is not propagated. SPHEREx position uncertainty is unavailable, so separation is not a
significance or identity probability.

`artifact_experiment.py` freezes a 95th-percentile concentration threshold on 32
source-injection development sites and applies it to 32 disjoint held-out sites. All
sites share one image, variance model, ePSF calibration and instrument; injections are
not independent real artifact labels. The real-image arm can match pre-existing sources.
The output labels a controlled artifact `ARTIFACT_UNRESOLVED`, even if the exploratory
feature flags it: this experiment is not a production classifier.

The independent check uses Photutils intensity moments and a border-median background
on two real images, rather than the generation finder's centroid or the PSF flux fit.
It clips negative residuals and has no calibrated positional/registration covariance;
its numerical separation cannot determine consistency. It is not the external Harness.

## Integration handoff

For Identify, supply the case A coordinate, UTC MJD, source and exposure IDs, flags,
full input lineage, catalog query and candidate list; retain unknown astrometric
covariance and allow no-match and ambiguity. The actual SearchX Identify API remains
unknown: the implementation was not accessible for this release. Repository read
access or a versioned interface specification is required; this package does not
replace it.

Harness at `3b28b39` accepts `schema_version: 1`, `case_id`, and backend-specific
`backends` blocks, not this evidence package. Its HelioLinX adapter requires a detection
CSV, column mapping, Earth ephemeris, observatory codes, heliocentric hypotheses and
reference MJD. THOR needs prepared observations and seed orbits; its linker comparison
needs at least two working linkers on shared IDs. This single source has no validated
moving-object track or backend verdict. Do not map a numerical centroid check to a
Harness acceptance claim.

### SearchX Identify evidence

The demo also writes `searchx-identifications.json` automatically (requires Node.js 20+ on PATH).
Each row is a detection inspector record with its original measurement, FITS provenance,
and independent checks. Aladin can place a marker at `detection.ra_deg` / `detection.dec_deg`
and use `detection.id` to find the inspector record. An inconclusive association can
contain a catalog counterpart; it does not confirm its identity. For the included
QR3 observations, calibrated positional uncertainty is unavailable and is left null.
The original independent checks are retained; Harness was not run.

The unmodified MIT matcher in `identify/engine.mjs` is pre-existing SearchX Identify
code from commit `7c24aeead8587215f9f9c73c0db51e2e17043cc8`; its license is included.

The demo validates the two original FITS WCS inputs and writes
`wcs-diagnostics.json`. Each Identify record links that file and explicitly refuses
motion and flux-variability claims: registration covariance and calibration are not
qualified. The marker's `identify_file` points to per-detection inspector records;
join by the marker `id` and record `detection.id`. The source case's tentative Gaia
association and Identify's **inconclusive** result use different evidence criteria;
neither confirms identity. Catalog input is the retained narrow cone, not a
field-wide survey. Harness remains `NOT_RUN`.

The bridge sends all detection jobs through one Node process while preserving the
original matcher's separate per-detection, epoch-propagated catalog inputs.
