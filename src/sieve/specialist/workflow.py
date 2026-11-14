"""Stream label-free curves into complete audit JSONL and bounded review cases."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from sieve.specialist.features import DOMAIN, features
from sieve.specialist.model import digest, load_model, write_json


class CurveManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["sieve.curves.v1"] = "sieve.curves.v1"
    asset: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    domain: Literal["KeplerLightCurves-11218015/interpolated-equal-length-v1"]
    preprocessing_ref: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)


def ranking_scores(model: Any, x: NDArray[np.float64]) -> NDArray[np.float64]:
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=np.float64)
    else:
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("model lacks positive class")
        scores = np.asarray(model.predict_proba(x)[:, classes.index(1)], dtype=np.float64)
    if scores.shape != (len(x),) or not np.isfinite(scores).all():
        raise ValueError("invalid model ranking output")
    return scores


def _constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON: {value}")


def _keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def run_curves(
    manifest_path: Path,
    model_dir: Path,
    output: Path,
    profile_sha256: str,
    budget: int = 20,
    json_array: bool = False,
    max_rows: int = 100_000,
    max_bytes: int = 2_000_000_000,
) -> dict[str, Any]:
    if not 1 <= budget <= 1000 or not 1 <= max_rows <= 1_000_000:
        raise ValueError("invalid review budget or row cap")
    if not 1 <= max_bytes <= 4_000_000_000:
        raise ValueError("invalid input byte cap")
    manifest = CurveManifest.model_validate_json(manifest_path.read_bytes())
    asset = manifest_path.parent / manifest.asset
    if asset.stat().st_size > max_bytes or digest(asset) != manifest.sha256:
        raise ValueError("input size cap or hash mismatch")
    model, profile = load_model(model_dir, profile_sha256)
    if manifest.sha256 == profile.training_sha256:
        raise ValueError("training asset cannot be target asset")
    if output.exists():
        raise ValueError("output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".curves-", dir=output.parent))
    try:
        result = _stream(asset, model, staging, budget, max_rows, json_array)
        from sieve.cli import source_identity

        commit, tree, dirty, env = source_identity()
        result.update(
            schema_version="sieve.curves.run.v1",
            input_sha256=manifest.sha256,
            profile_sha256=profile_sha256,
            domain=DOMAIN,
            score_semantics="RANKING_ONLY",
            domain_status="DECLARED_NOT_VERIFIED",
            code_commit=commit,
            code_tree_sha256=tree,
            dirty=dirty,
            environment_id=env,
            limitations=list(profile.limitations) + list(manifest.limitations),
        )
        write_json(staging / "summary.json", result)
        shutil.copyfile(manifest_path, staging / "input.manifest.json")
        shutil.copyfile(model_dir / "profile.json", staging / "profile.json")
        hashes = {p.name: digest(p) for p in sorted(staging.iterdir()) if p.is_file()}
        write_json(staging / "receipt.json", {"artifacts": hashes, "summary": result})
        staging.rename(output)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _stream(
    asset: Path, model: Any, staging: Path, budget: int, max_rows: int, json_array: bool
) -> dict[str, Any]:
    db = sqlite3.connect(staging / "inventory.sqlite")
    try:
        db.execute(
            "CREATE TABLE rows (ordinal INTEGER PRIMARY KEY, id TEXT UNIQUE, "
            "score REAL, payload TEXT NOT NULL)"
        )
        batch: list[tuple[int, str, dict[str, Any], NDArray[np.float64]]] = []
        counts = {"rows": 0, "ranked": 0, "rejected": 0}

        def flush() -> None:
            if not batch:
                return
            scores = ranking_scores(model, features(np.stack([item[3] for item in batch])))
            for (ordinal, _identifier, record, _), score in zip(batch, scores, strict=True):
                record.update(
                    score=float(score),
                    score_semantics="RANKING_ONLY",
                    disposition="INSUFFICIENT_EVIDENCE",
                    object_identity="NOT_ESTABLISHED",
                )
                db.execute(
                    "UPDATE rows SET score=?, payload=? WHERE ordinal=?",
                    (float(score), json.dumps(record, allow_nan=False), ordinal),
                )
                counts["ranked"] += 1
            batch.clear()

        with asset.open("rb") as stream:
            while True:
                raw = stream.readline(500_001)
                if not raw:
                    break
                if len(raw) > 500_000:
                    raise ValueError("curve JSONL line exceeds 500000 bytes")
                counts["rows"] += 1
                ordinal = counts["rows"]
                if ordinal > max_rows:
                    raise ValueError("row cap exceeded")
                record: dict[str, Any] = {
                    "row": ordinal,
                    "input_row_sha256": hashlib.sha256(raw).hexdigest(),
                }
                identifier = None
                try:
                    obj = json.loads(raw, parse_constant=_constant, object_pairs_hook=_keys)
                    if not isinstance(obj, dict) or set(obj) != {"id", "values"}:
                        raise ValueError("expected only id and values; labels are prohibited")
                    identifier = obj["id"]
                    if not isinstance(identifier, str) or not 1 <= len(identifier) <= 200:
                        raise ValueError("invalid source ID")
                    values = obj["values"]
                    if not isinstance(values, list) or any(
                        type(v) not in (int, float) for v in values
                    ):
                        raise ValueError("values must be a numeric array")
                    curve = np.asarray(values, dtype=np.float64)
                    features(curve[None, :])
                    record["id"] = identifier
                    db.execute(
                        "INSERT INTO rows VALUES (?, ?, NULL, ?)",
                        (ordinal, identifier, json.dumps(record)),
                    )
                    batch.append((ordinal, identifier, record, curve))
                except (ValueError, TypeError, sqlite3.IntegrityError) as exc:
                    record.update(
                        id=identifier if isinstance(identifier, str) else None,
                        score=None,
                        disposition="UNSUPPORTED_REGIME",
                        reason=str(exc),
                        object_identity="NOT_ESTABLISHED",
                    )
                    db.execute(
                        "INSERT INTO rows VALUES (?, NULL, NULL, ?)",
                        (ordinal, json.dumps(record, allow_nan=False)),
                    )
                    counts["rejected"] += 1
                if len(batch) == 64:
                    flush()
        flush()
        # Fixed integer budget, including deterministic tie handling.
        selected = [
            row[0]
            for row in db.execute(
                "SELECT ordinal FROM rows WHERE score IS NOT NULL "
                "ORDER BY score DESC, id ASC, ordinal ASC LIMIT ?",
                (budget,),
            )
        ]
        selected_set = set(selected)
        cases: list[dict[str, Any]] = []
        with (staging / "records.jsonl").open("w") as out:
            array = (staging / "records.json").open("w") if json_array else None
            try:
                if array:
                    array.write("[\n")
                for index, (ordinal, payload) in enumerate(
                    db.execute("SELECT ordinal, payload FROM rows ORDER BY ordinal")
                ):
                    record = json.loads(payload)
                    record["review_status"] = (
                        "SURFACED"
                        if ordinal in selected_set
                        else "DEFERRED"
                        if record.get("score") is not None
                        else "REJECTED"
                    )
                    text = json.dumps(record, allow_nan=False)
                    out.write(text + "\n")
                    if array:
                        array.write((",\n" if index else "") + text)
                    if ordinal in selected_set:
                        cases.append(record)
                if array:
                    array.write("\n]\n")
            finally:
                if array:
                    array.close()
        cases.sort(key=lambda record: (-record["score"], record["id"], record["row"]))
        write_json(staging / "cases.json", cases)
        db.commit()
        counts["surfaced"] = len(cases)
        counts["deferred"] = counts["ranked"] - len(cases)
        return {
            "counts": counts,
            "review_budget": budget,
            "batch_rows": 64,
            "max_rows": max_rows,
            "fdr_status": "NOT_ESTABLISHED",
        }
    finally:
        db.close()


def review_count(n: int, fraction: float = 0.05) -> int:
    if n <= 0 or not 0 < fraction <= 1:
        raise ValueError("invalid review population or fraction")
    return max(1, math.ceil(n * fraction))
