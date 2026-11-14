# SIEVE — Searching the Unknown

SIEVE is an experimental astronomy pipeline for traceable SPHEREx measurements and
competing source hypotheses. This public release demonstrates a real, bounded QR3
source investigation; it does not establish a discovery or a validated identity.

## Quick start

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync --frozen --extra photometry
PYTHONPATH=. uv run --no-sync python examples/source_investigation/run_demo.py --output reports/source-demo
```

The command reads two included real SPHEREx QR3 FITS cutouts, measures a source,
associates it with a retained ESA Gaia DR3 query, recomputes a separate centroid
check, and writes evidence and an Aladin-ready marker. Append `--live-gaia` to query
ESA directly; the command stops if the fixed catalog result changes or the query fails.
The output directory must not already exist.

Open `reports/source-demo/evidence.json` for the source inspector (case A) and
`reports/source-demo/aladin-marker.json` for its ICRS position and inspector reference.
Schemas and a hash manifest are generated alongside them. For an example without
running code, see [evidence.json](examples/source_investigation/evidence.json) and
[example-aladin-marker.json](examples/source_investigation/example-aladin-marker.json).
[Detailed instructions](examples/source_investigation/README.md) explain the interface,
data sources, and limitations.

The Gaia association is **tentative**, and the independent check is **inconclusive**:
qualified positional and registration uncertainties are unavailable. The artifact
example is a labeled simulation, not a real detection or a validated classifier.
The demo also runs SearchX Identify and writes per-detection inspector evidence
to `searchx-identifications.json`. Node.js 20+ is required. Harness is not run.

## Science and provenance

FITS ingestion, source detection, and measurements are implemented in
[`src/sieve/photometry`](src/sieve/photometry). The [QR3 input manifest](examples/qualification/source.json)
contains public parent-product URLs and SHA-256 hashes; included cutouts are checked
before processing. The repository also retains the broader identity, catalog and
triage implementation and its tests. Scientific results, limitations, and the origin
of this snapshot are described in [RELEASE_PROVENANCE.md](RELEASE_PROVENANCE.md).

The original SIEVE code is MIT licensed; see [LICENSE](LICENSE). SPHEREx data are
NASA/JPL/Caltech products distributed by IPAC; Gaia data are ESA products processed
by DPAC. See the [SPHEREx acknowledgment guidance](https://irsa.ipac.caltech.edu/Missions/spherex.html)
and [Gaia DR3 credit instructions](https://gea.esac.esa.int/archive/documentation/GDR3/Miscellaneous/sec_credit_and_citation_instructions/)
for published use.
