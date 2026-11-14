# Kepler RR Lyrae retrieval: measured v1 result

The frozen Extra Trees specialist retrieved **8/8 archive-labeled RR Lyrae curves within
20 reviews out of 399 held-out rows**, with average precision **0.986111**. The fixed logistic
baseline also retrieved 8/8. This is a measured narrow retrieval result; it does not support
claiming global benchmark leadership, a discovery or calibrated sensitivity.

The [official dataset](https://timeseriesclassification.com/description.php?Dataset=KeplerLightCurves)
is a multiclass, interpolated, equal-length archive. This experiment derives a binary
review-budget task from its published train/test split. It is not a submitted result on the
archive's multiclass accuracy leaderboard. Expert archive labels are evaluation proxies,
not authoritative physical truth. See [the protocol and workflow](specialist-workflow.md).

## Frozen search and test comparison

Training used 920 rows, including 17 positive archive labels. Three fixed training-only
folds had 307/6, 307/6 and 306/5 total/positive validation rows. The 11 preregistered
configurations all achieved mean validation recall of 1.0 at their ceil(5%) budgets; ten
also achieved validation AP of 1.0. Thus the primary tuning target was already saturated.
Extra Trees, depth 8, minimum leaf size 1, 300 estimators won the predeclared AP/ID tiebreak.
It was not selected using its test result. The search does not show that extensive tuning
was necessary: a fixed logistic baseline ties primary held-out recall.

| Frozen method | Recovered / 8 | Recall at 20 | Precision at 20 | Average precision |
|---|---:|---:|---:|---:|
| Training-CV-selected Extra Trees | 8 | 1.000 | 0.400 | 0.986111 |
| Fixed balanced logistic, C=1 | 8 | 1.000 | 0.400 | 0.958333 |
| Reference-only Isolation Forest | 3 | 0.375 | 0.150 | 0.134840 |
| Spectral concentration | 1 | 0.125 | 0.050 | 0.075558 |
| Seed-42 random ranking | 0 | 0.000 | 0.000 | 0.019077 |
| Selected architecture with shuffled training labels | 2 | 0.250 | 0.100 | 0.262200 |

Supervised methods receive class labels; Isolation Forest does not. This is an operational
baseline comparison, not a controlled attribution of improvements to architecture alone.
The single shuffled-label control does not estimate the full null distribution. No seed
shopping, model revision, test-based threshold change or holdout-based second search was
performed. The selected model's held-out ROC AUC is 0.999680; this secondary metric does not
replace the bounded-review task.

A conditional Wilson 95% interval for 8 successes out of 8 is approximately 67.6–100%,
assuming independent positive examples. Source-level independence cannot be established
from this archive. That interval is not a mission sensitivity interval. Eight positives
are too few to establish small differences between strong classifiers or transfer to an
external observing regime. The next useful experiment is a new, source-grouped external
holdout; repeatedly tuning this exposed test would weaken the evidence.

## Artifacts and runtime

Measured on 2026-10-06 in the canonical local checkout:

- Frozen training/evaluation source: `e81a9c9ce2c21b37fa4c10edcf30ad2c998912a0`, clean.
- Protocol SHA-256: `1f5f35867f7724f4ce768ef41596f102266e9f1a54517013897981eeac4a35ea`.
- Trained profile anchor: `860521ae18c2e51d13ada0a9dbd43ebc7fc92f66f261437f3091e590a2c41b36`.
- Model file: 5,581,580 bytes, persisted with skops; labels remain outside inference.
- Training/CV/control/test-export wall interval reported by evaluator:
  7.47 seconds **after acquisition** (downloads are excluded).
- Inference source: `6800270fdfd29044d161a8bf5ca902d303d7b4a7`, clean. This adds a fixed
  reviewed scikit-learn tree-type allowlist and persistence regression coverage; it does
  not alter features, model selection or test results.
- End-to-end persisted-model CLI inference: 11.50 seconds; child peak RSS 220,119,040 bytes,
  measured with Python `resource.getrusage` on Darwin. These are one-run local diagnostics.
- Label-free input JSONL: 42,373,762 bytes. Complete result JSON: 141,594 bytes; raw arrays
  remain in the pinned input. Output accounting: 399 ranked, 20 surfaced, 379 deferred,
  zero rejected. Every inference score exactly matches its evaluator counterpart.

Ignored local artifacts:

```text
reports/kepler-input-v1/acquisition.json
reports/kepler-specialist-v1/protocol.json
reports/kepler-specialist-v1/selection.json
reports/kepler-specialist-v1/benchmark.json
reports/kepler-specialist-v1/model.skops
reports/kepler-specialist-v1/profile.json
reports/kepler-specialist-v1/test.curves.jsonl
reports/kepler-specialist-v1/test.evaluator-labels.jsonl
reports/kepler-specialist-v1/receipt.json
reports/kepler-curves-v1/records.jsonl
reports/kepler-curves-v1/records.json
reports/kepler-curves-v1/cases.json
reports/kepler-curves-v1/receipt.json
reports/kepler-inference-resource.json
```

Both receipts' artifact hashes were independently checked locally. Complete JSON and JSONL
records match, and the bounded inference review contains all eight evaluator positives.
The supplied profile anchor is retained in this report; a receipt file by itself does not
prove authenticity. The local exposure registry now contains this test hash and refuses
changed protocols. Identical-protocol reproduction is permitted, not a fresh holdout.

## Verification and boundaries

149 local tests pass, including model round trips for all three search families, fixed
unknown-type refusal, feature/version/hash checks, complete export accounting, rejected rows,
duplicate IDs, domain/shape limits, deterministic budgets and holdout-registry refusal.
Ruff, formatting and strict mypy pass. The wheel builds offline and excludes evaluation
code and labels. Remote CI remains unrun. No push or merge was performed.

Real SPHEREx sensitivity, calibrated anomaly probabilities, case-level FDR, physical
classification, raw Fornax transfer, million-detection scalability and standards compliance
remain unestablished. MiniRocket and other published time-series contenders were not
compared, and no global superiority claim is justified.
