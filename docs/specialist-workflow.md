# Kepler RR Lyrae retrieval specialist v1

The implemented task is **retrieve archive label 7 (RR Lyrae) among the top ceil(5%) of
KeplerLightCurves benchmark rows**. This is supervised retrieval of a known archive class,
not general anomaly detection. The original catalog Isolation Forest remains a separate
unsupervised workflow. Comparing them measures different training information as well as
algorithms.

The [official archive description](https://timeseriesclassification.com/description.php?Dataset=KeplerLightCurves)
specifies 920 training rows, 399 test rows, seven classes, 4,767 samples per curve, and
expert-assigned labels. Missing samples were interpolated before release. The public
[Zenodo release](https://zenodo.org/records/11218015) is attributed under CC BY 4.0.
The archive's multiclass accuracy results are not comparable to this binary review-budget
metric. No state-of-the-art claim is supported by this implementation.

## Reproduce acquisition and tuning

```sh
uv sync --frozen --extra catalog --extra specialist
uv run --no-sync python examples/specialist/acquire_kepler.py --output reports/kepler-input-v1
uv run --no-sync python -m evaluation.kepler_specialist \
  --inputs reports/kepler-input-v1 --output reports/kepler-specialist-v1
```

Acquisition is capped at exact published asset sizes and verified against SHA-256 pins.
Total download size is 133,644,813 bytes. Real data, models, detailed evaluation labels and
large exports remain under ignored `reports/`; they are not included in the wheel.

The evaluator freezes the task, features, 11 candidate settings, package versions and code
hashes in `protocol.json`. Three-fold stratified cross-validation groups identical curves
by content hash. Fold-local scaler fitting uses scikit-learn Pipelines. Selection is mean
fold recall within the fixed 5% budget, then mean average precision, then stable candidate
ID. Selected settings, every configuration and every fold are saved in `selection.json`
before any test rows are parsed. The selected estimator is fitted once on all training rows.

A local exposure registry records the test asset hash and protocol hash before test parsing.
Changed protocols refuse that already-used holdout. Identical-protocol reruns are allowed
for reproducibility; they do not constitute fresh evidence. The registry is mutable local
bookkeeping, not cryptographic protection against a researcher rewriting history. A failed
benchmark directory can contain partial outputs and is not a completed run without
`benchmark.json` and `receipt.json`. Use a new output directory rather than overwriting it.

## Run the trained model without labels

Read the profile anchor from the completed benchmark report and retain it independently.
For an inference smoke run on its separated, label-free test asset:

```sh
PROFILE_SHA=$(uv run --no-sync python -c 'import json; print(json.load(open("reports/kepler-specialist-v1/benchmark.json"))["profile_sha256"])')
uv run --no-sync sieve curves --input reports/kepler-specialist-v1/test.manifest.json \
  --model reports/kepler-specialist-v1 --profile-sha256 "$PROFILE_SHA" \
  --output reports/kepler-curves-v1 --budget 20 --json-array
```

Each JSONL input row has exactly `id` and a numeric `values` array. Labels and extra columns
are refused. IDs name input rows, not established physical objects. The manifest pins the
asset SHA-256 and declares the exact supported domain, source URI, preprocessing reference
and limitations. Inference cannot verify an honest domain declaration from a numeric array;
`domain_status=DECLARED_NOT_VERIFIED` makes that limitation explicit. A raw Fornax curve does
not become qualified by changing its manifest string. Irregular sampling, other cadence,
missing data or preprocessing changes need a new adapter and external qualification.

Output contains all rows in `records.jsonl`, optional streamed `records.json`, bounded
`cases.json`, a SQLite inventory, manifest/profile copies, summary and hashed receipt.
Invalid rows, duplicate IDs and deferred rows remain in the audit. No raw light-curve arrays
are duplicated in every output record: row hashes and the pinned original asset retain
traceability. Ties use stable row IDs. The model requires an independently supplied profile
hash, exact feature implementation and dependency versions; model hashes and unreviewed
skops types are checked before loading. The only explicit reviewed type is
`sklearn.tree._tree.Tree`, needed for Extra Trees; other unknown types are refused.
No pickle is accepted.

Processing uses batches of 64, a 500,000-byte line cap, a default 100,000-row cap, 2 GB input
cap, 1,000-case review cap, 64 KB profile cap and 100 MB serialized-model cap. Row counts,
scores and review cases use SQLite rather than an in-memory collection of all curves.
These are resource limits, not measured million-row scalability. The receipt checks local
bytes; authenticity requires retaining its hash externally. The input/model assets remain
external to the run directory. Remote CI is unrun.

## Nine rules enforced or exposed

1. Freeze the task, review budget, search, seed and feature implementation before the test.
2. Fit population preprocessing and classifiers only inside training folds.
3. Group exact duplicate curves inside cross-validation; refuse cross-split duplicates.
4. Persist training-based selection before parsing test labels; reject changed exposed protocols.
5. Report fixed supervised, unsupervised, simple and random baselines plus label permutation.
6. Report class counts, recovered counts, precision, AP, ROC AUC and conditional recall intervals.
7. Keep labels in the evaluator and refuse them in inference input.
8. Anchor model/profile/input hashes and versions; fail on unsupported shapes/domains/types.
9. Preserve every accepted or rejected input row while bounding the human review queue.

Exact duplicates are only one leakage mechanism. Original star and observing-group IDs are
unavailable, so astrophysical independence cannot be certified. Archive expert labels remain
proxies rather than authoritative physical identity. Scores remain `RANKING_ONLY`; outputs
remain `INSUFFICIENT_EVIDENCE` or `UNSUPPORTED_REGIME`, with `NOT_ESTABLISHED` identity.

## Reuse and remaining gaps

The classifiers, pipelines, splits and metrics reuse maintained scikit-learn implementations;
model persistence reuses skops. [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)
underlies fold-local fitting. [Its persistence guidance](https://scikit-learn.org/stable/model_persistence.html)
explains version compatibility and trusted-model handling.

[aeon's MiniRocket](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.transformations.collection.convolution_based.MiniRocket.html)
is a credible subsequent equal-length time-series contender, with reference-only fitted
convolution features. It is **not tested here**, so beating the current baselines does not
establish superiority over it. Adding it after this holdout has been exposed requires a
fresh qualifying test set. The feature summaries here deliberately exclude absolute flux
scale; FFT frequencies are cycles per sample, not measured physical periods.

[Fornax's light-curve classifier notebook](https://nasa-fornax.github.io/fornax-demo-notebooks/light-curve-classifier/)
addresses a different CLAGN/QSO task. [AnomalyMatch](https://github.com/ESA/AnomalyMatch)
addresses images and active learning. Those published task scopes are transfer gaps, not
claims that those systems are defective. Their results cannot justify this model on irregular
Fornax curves. The next scientific work is a separately acquired, source-grouped Fornax
holdout with a documented cadence/missingness adapter, review adjudication and injection
controls that preserve observational nuisance structure. That evidence is currently absent.

Improved implementation brief: “Optimize archive-label RR Lyrae retrieval at a fixed 5%
review budget on pinned KeplerLightCurves. Select on training-only grouped CV, freeze before
the official test, compare fixed controls, export label-free ranking artifacts, and qualify
transfer separately. Claim only what the named benchmark demonstrates.”

The [measured v1 report](kepler-benchmark-report.md) records the frozen selection, held-out
comparisons, local resource measurements and verified export accounting.
