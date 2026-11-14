# Paired evaluator arithmetic fixture

These are **manually authored synthetic predictions**, not outputs from a SPHEREx factor,
trained model, Harness or empirical astrometry linkage implementation. The treatment is
assigned the intended partition to test evaluator arithmetic. Eight artificial source groups
and a perfect reduction are deliberately constructed; they provide **no thesis evidence**.

Both thresholds in `protocol.json` come from the two validation groups. The held-out fixture
contains eight separate synthetic source/dependence groups, with a clean case and a mixed
case in each group. Repeated clean/mixed appearances must be grouped for uncertainty.
`qualification=CONTRACT_ONLY` prevents a numerical success from becoming a scientific go.

From repository root:

```sh
uv run --no-sync python -m evaluation.identity_cli freeze --predictions evaluation/fixtures/identity-paired/val-baseline.json --truth evaluation/fixtures/identity-paired/val-truth.json --target-recall 1 --output reports/baseline-threshold.json
uv run --no-sync python -m evaluation.identity_cli freeze --predictions evaluation/fixtures/identity-paired/val-treatment.json --truth evaluation/fixtures/identity-paired/val-truth.json --target-recall 1 --output reports/treatment-threshold.json
shasum -a 256 evaluation/fixtures/identity-paired/protocol.json
uv run --no-sync python -m evaluation.identity_cli compare --baseline evaluation/fixtures/identity-paired/test-baseline.json --treatment evaluation/fixtures/identity-paired/test-treatment.json --truth evaluation/fixtures/identity-paired/test-truth.json --protocol evaluation/fixtures/identity-paired/protocol.json --protocol-sha256 PROTOCOL_FILE_SHA256 --output reports/paired-contract-demo
```

Replace `PROTOCOL_FILE_SHA256` with the printed digest. All output paths must be new; create
`reports/` first when running only the threshold commands. The comparison emits paired metrics,
all 16 cases, eight constructed baseline failure examples, input copies and code/environment/
artifact hashes. Its expected verdict is `CONTRACT_CHECK_PASSED`, with claim status
`NO_THESIS_EVIDENCE_CONTRACT_ONLY`. Inputs/truth are excluded from the inference wheel.

For real research, assemble a reviewed `ComparisonProtocol` using the two validation-only
frozen thresholds, qualified source/dependency splits, exact one-factor/control byte hashes,
independently justified power/effect targets and external evaluation. Publishing a protocol
hash and supplying sealed predictions is a procedural requirement: this local command does
not stop an operator from previously reading labels or rerunning a different experiment.
