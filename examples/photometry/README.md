# Pinned SPHEREx QR3 image scan

This is a real-data engineering demonstration. `qr3-d1-cutout.fits` is a 256 × 256
pixel excerpt from one public SPHEREx Quick Release 3 (QR3) D1 spectral image. The
original product identity, retrieval links, exact bytes hash, crop indices, and retained
FITS extensions are recorded in [`source.json`](source.json). The reduced FITS sample
is checked in so a fresh clone can run the example offline.

Reproduce it from the repository root:

```sh
uv sync --frozen --no-dev --extra photometry
uv run --no-sync python examples/photometry/run_demo.py
```

The command writes `reports/qr3-photometry/scan.json`. The destination must not exist;
choose another `--output` path for another run. `expected-scan.json` is the retained
measurement output from the pinned input and environment.

This minimal runtime setup and all 409 measurement rows were reproduced from a fresh
remote-main clone at `528399d87357725300ace9cad21d8bef7d2f78e9`. `--no-dev` leaves out
development tools; install the development group and other extras to run the full suite.

On the recorded run, Photutils 3.0.0 found 418 point-like peaks, rejected 9 apertures
that crossed the cutout edge, and measured 409 positions. All 418 peak centers carry the
upstream SPHEREx `SOURCE` flag; excluding those known-source-mask pixels leaves zero peaks
at the configured threshold. The main detector retained SOURCE pixels and did not
receive catalog source positions. The zero-count secondary search is a novelty diagnostic,
not evidence of detector failure or catalog-driven peak placement. These are not new objects;
completeness and false detections were not qualified by that scan. The background estimate
was 0.2785301246 MJy/sr; its median background RMS
estimate was 0.01919147291 MJy/sr. One retained example has sky position
(232.7528113°, -10.52481581°), aperture sum 0.6500992 MJy/sr pixel, and propagated
archive-variance aperture error 0.1066937 MJy/sr pixel. These numbers describe this
configured extraction only. The aperture sum is not
converted to source flux density because the detector pixel solid-angle calibration map
is not applied. The reported S/N is a diagnostic from the released variance plane, not a
calibrated detection probability.

136 of the 409 retained rows have masked pixels inside their apertures and are marked
`aperture_usable=false`; their sums omit those pixels. The other 273 have complete pixel
coverage under this mask policy, which still does not qualify the scientific errors or flux.

The processing chain uses Photutils `Background2D`, `DAOStarFinder`, and exact-overlap
circular aperture photometry. It uses the SPHEREx IMAGE, FLAGS, VARIANCE, and celestial
WCS products. Documented serious detector flags are masked; the known-source mask bit is
excluded from background fitting and separately counted. The finder uses a circular 2-pixel FWHM
kernel and a 5 × archive per-pixel standard-deviation threshold, with DAOFIND's default
kernel threshold scaling. A 3-pixel radius aperture is used for
the measured rows. The scan JSON retains coordinates, pixel values, aperture values and
errors, variance-based S/N diagnostics, centroid flags, release/exposure identity, exact
input hash, code/source-tree identity, environment hash, and limitations.

The measurement method is intentionally a small baseline. The SPHEREx Source Discovery
Tool and Spectrophotometry Tool already provide mission-specific source discovery and
PSF-based forced photometry. SIEVE's adapter does not replace them: it does not fit the
position-dependent ePSF, convert MJy/sr with the solid-angle map, deblend crowded fields,
derive calibrated probabilities, evaluate completeness/purity, generate cross-exposure
links, test physical hypotheses, or establish a moving object. See the [release provenance](../../RELEASE_PROVENANCE.md) for scientific limits.

The product can be fetched again from its `original_product_url`; the full 65,851,200-byte
product SHA-256 and SIA2 query are in `source.json`. The checked-in sample was produced
with Astropy Cutout2D using the stored pixel slices. Rebuild it from the downloaded parent:

```sh
uv run --no-sync python examples/photometry/rebuild_sample.py --parent /path/to/parent.fits --output /path/to/new-sample.fits
```

The rebuild verifies the parent and output hashes. It reconstructs only the celestial WCS;
the original WCS-WAVE table is retained as context and is not qualified for spectral use in
the reduced file. IRSA's guidance explains the QR3
collections and source-specific spectral-image structure: [SPHEREx data access](https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-access/).
