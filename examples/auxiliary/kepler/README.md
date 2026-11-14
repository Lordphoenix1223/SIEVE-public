# Frozen auxiliary benchmark

This experiment is **AUXILIARY catalog/review-budget ranking evidence only**. It is not
SIEVE's physical-identity test. Do not retune it, change its target, make SOTA/global
leaderboard claims, or treat its candidate rankings as discovery evidence.

The preserved `protocol.json`, `selection.json`, `benchmark.json`, `profile.json`,
`receipt.json` and all 399 `test.scores.jsonl` records are copied byte-for-byte from the
original local run. `frozen-inventory.json` pins both published metadata and original local
artifacts, including inputs/model/rankings. Original data/model files remain under ignored
`reports/`; the Zenodo CC BY 4.0 data can be reacquired using its pinned acquisition example.
No original artifact or experiment source file is modified by the identity work.

The source namespace `sieve.specialist` is legacy **auxiliary-only** and retained to keep
existing source/profile hashes valid. Identity/contracts/provenance/triage import none of
it. The shared runtime receipt helper lives in `sieve.provenance.runtime`. The CLI's
`curves` command is explicitly labeled auxiliary; it is not the identity funnel.

For historical reproduction see `docs/specialist-workflow.md` and
`docs/kepler-benchmark-report.md`. Reproducing the original frozen protocol is not new
held-out evidence. Use its exact locked dependencies and retained source revisions;
never use the already-exposed test to design another search. Published rankings permit
independent comparison without re-running any parameter search.
