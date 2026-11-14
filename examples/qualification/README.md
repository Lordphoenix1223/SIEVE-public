# QR3 qualification inputs

The three real SPHEREx QR3 cutouts and their verified public parent-product URLs,
processing versions, calibration inputs, and SHA-256 hashes are in [source.json](source.json).
The offline pilot can be run from the repository root with:

```sh
PYTHONPATH=. uv run --no-sync python -m examples.qualification.run_experiment --output reports/qr3-pilot
```

Its results are conditional and scientifically inconclusive; no moving-source truth,
qualified positional covariance, independent identity cohort, or validated false-alert
rate is provided. The historical retained run archives are intentionally excluded from
this public snapshot; see [release provenance](../../RELEASE_PROVENANCE.md). The
source-investigation [demonstration](../source_investigation/README.md) is the shorter
frontend-oriented path.
