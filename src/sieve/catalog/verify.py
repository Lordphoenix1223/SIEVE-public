"""Independent local byte/accounting verification; no scientific authority."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from sieve.catalog.io import file_hash, strict_json
from sieve.catalog.models import CatalogConfig
from sieve.contracts.models import Score, ScoreSemantics


def verify_run(directory: Path, expected_receipt_hash: str | None = None) -> dict[str, Any]:
    try:
        return _verify_run(directory, expected_receipt_hash)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed catalog receipt or output structure: {exc}") from exc


def _verify_run(directory: Path, expected_receipt_hash: str | None) -> dict[str, Any]:
    receipt_path = directory / "receipt.json"
    actual_receipt_hash = file_hash(receipt_path)
    if expected_receipt_hash is not None and actual_receipt_hash != expected_receipt_hash:
        raise ValueError("receipt SHA256 does not match supplied trust anchor")
    receipt = strict_json(receipt_path.read_bytes())
    required = {
        "records.jsonl",
        "rejected.jsonl",
        "cases.json",
        "inventory.sqlite",
        "model.json",
        "summary.json",
        "reference.manifest.json",
        "target.manifest.json",
    }
    if not isinstance(receipt, dict) or not isinstance(receipt.get("artifacts"), dict):
        raise ValueError("invalid catalog receipt")
    if not required <= receipt["artifacts"].keys():
        raise ValueError("receipt is missing required catalog artifacts")
    for name, artifact in receipt["artifacts"].items():
        if Path(name).name != name or name in {".", "..", "receipt.json"}:
            raise ValueError("artifact reference must be a local basename")
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or symlinked artifact: {name}")
        if path.stat().st_size != artifact["bytes"] or file_hash(path) != artifact["sha256"]:
            raise ValueError(f"artifact bytes/hash mismatch: {name}")
    summary = strict_json((directory / "summary.json").read_bytes())
    if summary["status"] != "COMPLETE_CATALOG_RANKING":
        raise ValueError("run does not declare complete catalog processing")
    config = CatalogConfig.model_validate(summary["configuration"])
    if summary["review_budget"] != config.review_budget:
        raise ValueError("summary review budget differs from validated configuration")
    counts: Counter[str] = Counter()
    rows = 0
    surfaced: dict[str, Any] = {}
    with (directory / "records.jsonl").open() as stream:
        for line in stream:
            row = strict_json(line)
            rows += 1
            counts[row["decision"]] += 1
            if row["object_identity_status"] != "NOT_ESTABLISHED" or row[
                "scientific_disposition"
            ] not in {"INSUFFICIENT_EVIDENCE", "UNSUPPORTED_REGIME"}:
                raise ValueError("catalog record overstates scientific qualification")
            if row["score"] is not None:
                score = Score.model_validate(row["score"])
                if score.semantics != ScoreSemantics.RANKING_ONLY or score.direction != "higher":
                    raise ValueError("catalog score is not an uncalibrated higher-first ranking")
            if row["decision"] == "SURFACED":
                if len(surfaced) >= summary["review_budget"]:
                    raise ValueError("surface budget exceeded")
                if row["record_id"] in surfaced:
                    raise ValueError("duplicate surfaced source")
                surfaced[row["record_id"]] = row
    cases = strict_json((directory / "cases.json").read_bytes())["cases"]
    if (
        rows != summary["counts"]["target"]["input_rows"]
        or dict(counts) != summary["counts"]["output"]
    ):
        raise ValueError("row/decision accounting differs from summary")
    if len(cases) != len(surfaced) or {r["record_id"]: r for r in cases} != surfaced:
        raise ValueError("case packet differs from surfaced inventory rows")
    return {
        "status": "VERIFIED_LOCAL_ARTIFACT_BYTES_AND_ACCOUNTING",
        "receipt_sha256": actual_receipt_hash,
        "receipt_trust": "MATCHES_SUPPLIED_HASH"
        if expected_receipt_hash
        else "UNANCHORED_LOCAL_RECEIPT",
        "records_checked": rows,
        "surfaced_checked": len(surfaced),
        "artifacts_checked": len(receipt["artifacts"]),
        "limitations": [
            "Checks output consistency, not original input availability or scientific correctness",
            "An unanchored receipt can be edited together with artifacts; no authenticity claim",
            "Input hashes do not establish calibration, physical identity or discovery",
        ],
    }
